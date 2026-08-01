#!/usr/bin/env python3
"""Annotate a local corpus with the Co-LMLM ModernBERT fact-span tagger.

RunPod / CLI port of Co_LMLM_annotate_ModernBERT.ipynb. Reads shards from a
local directory (no Google Drive / Colab), writes zstd JSONL annotations, and
supports multi-worker sharding via --worker-index / --num-workers so each
single-GPU job owns a disjoint subset of input files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
import time
from pathlib import Path

import torch
import zstandard
from transformers import AutoModelForTokenClassification, AutoTokenizer

LABELS = ["O", "B-FACT", "I-FACT"]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}

READABLE_SUFFIXES = (
    ".jsonl.zst",
    ".ndjson.zst",
    ".json.zst",
    ".jsonl.gz",
    ".ndjson.gz",
    ".json.gz",
    ".jsonl",
    ".ndjson",
    ".json",
    ".parquet",
    ".csv",
    ".tsv",
    ".txt",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Folder with config.json, model.safetensors, tokenizer.*",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Corpus root (walked recursively for readable shards)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where *.annotations.jsonl.zst and _manifest.json are written",
    )
    p.add_argument(
        "--path-prefix",
        default=None,
        help="Only annotate files whose relative path starts with this prefix",
    )
    p.add_argument("--text-field", default="text")
    p.add_argument(
        "--id-field",
        default="doc_id",
        help="Stable id field (FineWeb raw uses doc_id; falls back to <stem>-<n>)",
    )
    p.add_argument("--source-field", default="source")
    p.add_argument("--source-default", default="fineweb-edu")
    p.add_argument(
        "--include-text",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--zstd-level", type=int, default=10)
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap input files this worker processes (after worker filter)",
    )
    p.add_argument(
        "--max-docs-per-file",
        type=int,
        default=None,
        help="Cap docs read from each input file (smoke / trial)",
    )
    p.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="This job's index in [0, num-workers)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total parallel jobs; files are assigned by sorted index %% num-workers",
    )
    p.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After annotate, check span offsets on the newest non-empty shard",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run a one-sentence tag demo and exit (no corpus)",
    )
    return p.parse_args()


def list_input_files(input_dir: Path, path_prefix: str | None) -> list[Path]:
    files: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if not name.endswith(READABLE_SUFFIXES):
            continue
        rel = path.relative_to(input_dir).as_posix()
        if path_prefix and not rel.startswith(path_prefix.replace("\\", "/")):
            continue
        files.append(path)
    # Largest first so a trial run hits a real shard rather than a tiny sidecar.
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files


def assign_files(files: list[Path], worker_index: int, num_workers: int) -> list[Path]:
    if num_workers < 1:
        raise SystemExit("--num-workers must be >= 1")
    if not (0 <= worker_index < num_workers):
        raise SystemExit(f"--worker-index must be in [0, {num_workers})")
    # Stable assignment on sorted-by-name order so workers don't all pick the
    # same largest file when max-files is set.
    ordered = sorted(files, key=lambda p: p.as_posix())
    return [p for i, p in enumerate(ordered) if i % num_workers == worker_index]


def load_model(model_dir: Path):
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"No config.json under {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device == "cuda"
        else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = None
    for attn in ("sdpa", "eager"):
        try:
            model = AutoModelForTokenClassification.from_pretrained(
                model_dir, attn_implementation=attn
            )
            break
        except Exception as exc:  # noqa: BLE001 — try next attn backend
            print(f"attn={attn} failed: {exc}", flush=True)
    if model is None:
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    if hasattr(model.config, "reference_compile"):
        model.config.reference_compile = False
    # Cast after load: avoids torch_dtype/dtype kwarg rename across transformers.
    model.to(device=device, dtype=dtype).eval()
    print(
        f"loaded {model_dir} on {device} ({dtype}), labels={model.config.id2label}",
        flush=True,
    )
    return tokenizer, model, device


def make_tag_batch(tokenizer, model, device, max_length: int):
    @torch.inference_mode()
    def tag_batch(texts: list[str]) -> list[list[dict]]:
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        preds = model(input_ids=input_ids, attention_mask=attn).logits.argmax(-1).cpu().tolist()
        offsets = enc["offset_mapping"].tolist()
        special = enc["special_tokens_mask"].tolist()
        out: list[list[dict]] = []
        for b, text in enumerate(texts):
            spans: list[list[int]] = []
            cur = None
            for k, tid in enumerate(preds[b]):
                s, e = offsets[b][k]
                if special[b][k] or e <= s:
                    if cur:
                        spans.append(cur)
                        cur = None
                    continue
                if tid == LABEL2ID["B-FACT"]:
                    if cur:
                        spans.append(cur)
                    cur = [s, e]
                elif tid == LABEL2ID["I-FACT"] and cur:
                    cur[1] = e
                else:
                    # Orphan I-FACT closes like O (notebook decoder; MD notes a
                    # train-time "treat as start" variant — keep Colab behavior).
                    if cur:
                        spans.append(cur)
                        cur = None
            if cur:
                spans.append(cur)
            recs = []
            for s, e in spans:
                # ModernBERT byte-level BPE folds the preceding space into the
                # token offset; trim so span == text[char_start:char_end].
                while s < e and text[s].isspace():
                    s += 1
                while e > s and text[e - 1].isspace():
                    e -= 1
                if s < e:
                    recs.append(
                        {
                            "span": text[s:e],
                            "char_start": s,
                            "char_end": e,
                            "faithful": True,
                        }
                    )
            out.append(recs)
        return out

    return tag_batch


def _iter_jsonl(fh):
    for line in fh:
        line = line.strip()
        if line:
            yield json.loads(line)


def read_records(path: Path, name: str, text_field: str):
    low = name.lower()
    if low.endswith(".zst"):
        with open(path, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as r:
            text = io.TextIOWrapper(r, encoding="utf-8")
            if low.endswith((".jsonl.zst", ".ndjson.zst")):
                yield from _iter_jsonl(text)
            elif low.endswith(".json.zst"):
                obj = json.load(text)
                yield from (obj if isinstance(obj, list) else obj.get("data", [obj]))
            else:
                raise ValueError(f"unsupported compressed type: {name}")
    elif low.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            if low.endswith((".jsonl.gz", ".ndjson.gz")):
                yield from _iter_jsonl(fh)
            elif low.endswith(".json.gz"):
                obj = json.load(fh)
                yield from (obj if isinstance(obj, list) else obj.get("data", [obj]))
            else:
                raise ValueError(f"unsupported compressed type: {name}")
    elif low.endswith((".jsonl", ".ndjson")):
        with open(path, encoding="utf-8") as fh:
            yield from _iter_jsonl(fh)
    elif low.endswith(".json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        yield from (obj if isinstance(obj, list) else obj.get("data", [obj]))
    elif low.endswith(".parquet"):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                yield row
    elif low.endswith((".csv", ".tsv")):
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if low.endswith(".tsv") else ",")
            yield from reader
    elif low.endswith(".txt"):
        yield {text_field: path.read_text(encoding="utf-8")}
    else:
        raise ValueError(f"unsupported file type: {name}")


def has_documents(path: Path, name: str, text_field: str, probe: int = 5):
    keys: list[str] = []
    try:
        for i, rec in enumerate(read_records(path, name, text_field)):
            if isinstance(rec, dict):
                keys = sorted(rec.keys())
                val = rec.get(text_field)
                if isinstance(val, str) and val.strip():
                    return True, keys
            if i + 1 >= probe:
                break
    except Exception as e:  # noqa: BLE001
        return False, [f"unreadable: {type(e).__name__}: {e}"]
    return False, keys


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"files": [], "workers": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {"files": raw, "workers": {}}
    raw.setdefault("files", [])
    raw.setdefault("workers", {})
    return raw


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


class ShardWriter:
    def __init__(self, output_dir: Path, stem: str, zstd_level: int):
        self.output_dir = output_dir
        self.stem = stem
        self.zstd_level = zstd_level
        self.path: Path | None = None
        self._fout = self._w = None

    def _open(self) -> None:
        self.path = self.output_dir / f"{self.stem}.annotations.jsonl.zst"
        self._fout = open(self.path, "wb")
        self._w = zstandard.ZstdCompressor(level=self.zstd_level).stream_writer(self._fout)

    def write(self, rec: dict) -> None:
        if self._w is None:
            self._open()
        self._w.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))

    def close(self) -> Path | None:
        if self._w is None:
            return None
        self._w.close()
        self._fout.close()
        finished = self.path
        self._w = self._fout = self.path = None
        return finished


def stem_for(name: str) -> str:
    stem = name
    for suffix in (
        ".gz",
        ".zst",
        ".jsonl",
        ".ndjson",
        ".json",
        ".parquet",
        ".csv",
        ".tsv",
        ".txt",
    ):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def annotate_file(
    path: Path,
    *,
    tag_batch,
    args: argparse.Namespace,
) -> tuple[Path | None, int, int, list]:
    name = path.name
    ok, keys = has_documents(path, name, args.text_field)
    if not ok:
        return None, 0, 0, keys

    writer = ShardWriter(args.output_dir, stem_for(name), args.zstd_level)
    n_docs = n_spans = 0
    texts: list[str] = []
    metas: list[dict] = []
    t0 = time.time()
    stem = stem_for(name)

    def flush() -> None:
        nonlocal n_docs, n_spans
        if not texts:
            return
        for meta, spans in zip(metas, tag_batch(texts)):
            rec = {"id": meta["id"], "source": meta["source"], "annotations": spans}
            if args.include_text:
                rec["text"] = meta["text"]
            writer.write(rec)
            n_docs += 1
            n_spans += len(spans)
        texts.clear()
        metas.clear()

    for n, rec in enumerate(read_records(path, name, args.text_field)):
        if args.max_docs_per_file is not None and n_docs + len(texts) >= args.max_docs_per_file:
            break
        text = rec.get(args.text_field) if isinstance(rec, dict) else None
        if not text or not isinstance(text, str):
            continue
        texts.append(text[: args.max_length * 12])
        raw_id = rec.get(args.id_field) if isinstance(rec, dict) else None
        source = (
            rec.get(args.source_field, args.source_default)
            if isinstance(rec, dict)
            else args.source_default
        )
        metas.append(
            {
                # Keep doc_id=0 (falsy) — only fall back when the field is absent.
                "id": str(raw_id) if raw_id is not None else f"{stem}-{n}",
                "source": source,
                "text": text,
            }
        )
        if len(texts) >= args.batch:
            flush()
            if n_docs and n_docs % (args.batch * 20) == 0:
                rate = n_docs / max(time.time() - t0, 1e-9)
                print(
                    f"    {name}: {n_docs:,} docs, {n_spans:,} spans, {rate:.0f} docs/s",
                    flush=True,
                )
    flush()
    out_path = writer.close()
    return out_path, n_docs, n_spans, keys


def verify_output(output_dir: Path, limit: int = 2000) -> int:
    shards = sorted(output_dir.glob("*.annotations.jsonl.zst"))
    print(f"{len(shards)} output shard(s):", flush=True)
    for s in shards:
        print(f"  {s.name}  {s.stat().st_size / 1e6:.2f} MB", flush=True)
    candidates = [s for s in reversed(shards) if s.stat().st_size > 0]
    if not candidates:
        print("no non-empty shards to verify", flush=True)
        return 1
    shard = candidates[0]
    print(f"\nverifying {shard.name}", flush=True)
    bad = 0
    n = -1
    with open(shard, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as r:
        text = io.TextIOWrapper(r, encoding="utf-8")
        for n, line in enumerate(text):
            rec = json.loads(line)
            if "text" in rec:
                for a in rec["annotations"]:
                    if rec["text"][a["char_start"] : a["char_end"]] != a["span"]:
                        bad += 1
            if n < 2:
                print(f"\n{rec['id']}  ({len(rec['annotations'])} spans)", flush=True)
                for a in rec["annotations"][:8]:
                    print(f"  [{a['char_start']:>5}] {a['span']!r}", flush=True)
            if n >= limit:
                break
    print(f"\nchecked {n + 1} records, {bad} offset mismatches", flush=True)
    return 0 if bad == 0 and n >= 0 else 1


def main() -> int:
    args = parse_args()
    print("torch", torch.__version__, flush=True)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0), flush=True)
        print("bf16 supported:", torch.cuda.is_bf16_supported(), flush=True)
    else:
        print("WARNING: no GPU — tagging will be slow", flush=True)

    tokenizer, model, device = load_model(args.model_dir)
    tag_batch = make_tag_batch(tokenizer, model, device, args.max_length)

    demo = "The Eiffel Tower was completed in 1889 and stands 330 metres tall in Paris."
    print("demo:", tag_batch([demo])[0], flush=True)
    if args.demo:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.input_dir.is_dir():
        raise SystemExit(f"input dir not found: {args.input_dir}")

    all_files = list_input_files(args.input_dir, args.path_prefix)
    mine = assign_files(all_files, args.worker_index, args.num_workers)
    print(
        f"input: {len(all_files)} readable file(s); "
        f"worker {args.worker_index}/{args.num_workers} owns {len(mine)}",
        flush=True,
    )
    for f in mine[:20]:
        rel = f.relative_to(args.input_dir).as_posix()
        print(f"  {rel}  {f.stat().st_size / 1e6:.2f} MB", flush=True)
    if len(mine) > 20:
        print(f"  ... and {len(mine) - 20} more", flush=True)

    manifest_path = args.output_dir / "_manifest.json"
    manifest = load_manifest(manifest_path)
    # Resume key includes worker so parallel jobs don't clobber each other.
    done_key = f"w{args.worker_index}"
    done = set(manifest.get("workers", {}).get(done_key, manifest.get("files", [])))
    candidates = [f for f in mine if f.name not in done]
    if args.max_files is not None:
        candidates = candidates[: args.max_files]

    grand_docs = grand_spans = 0
    annotated = 0
    start = time.time()
    print(f"{len(candidates)} candidate(s) ({len(done)} already done)\n", flush=True)

    for f in candidates:
        print(f"[{annotated + 1}] {f.name} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)
        out_path, nd, ns, keys = annotate_file(f, tag_batch=tag_batch, args=args)
        if out_path is None:
            print(f"    skipped: no {args.text_field!r} in records; keys seen: {keys}", flush=True)
            continue
        annotated += 1
        grand_docs += nd
        grand_spans += ns
        done.add(f.name)
        manifest.setdefault("workers", {})[done_key] = sorted(done)
        # Keep legacy top-level list as union for humans reading the file.
        union = set()
        for names in manifest["workers"].values():
            union.update(names)
        manifest["files"] = sorted(union)
        save_manifest(manifest_path, manifest)
        print(
            f"    -> {out_path.name}: {nd:,} docs, {ns:,} spans "
            f"({ns / max(nd, 1):.1f}/doc)  [{(time.time() - start) / 60:.1f} min]",
            flush=True,
        )

    print(
        f"\ndone: {grand_docs:,} docs, {grand_spans:,} spans "
        f"[{(time.time() - start) / 60:.1f} min]",
        flush=True,
    )
    if grand_docs == 0:
        print(f"nothing annotated — check TEXT_FIELD={args.text_field!r} / input-dir", flush=True)
        return 2

    if args.verify:
        return verify_output(args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
