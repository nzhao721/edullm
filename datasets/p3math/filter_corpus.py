#!/usr/bin/env python3
"""Filter downloaded PP2 subsets into P3Math jsonl.zst outputs; copy Lean4 as-is."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import zstandard as zstd

# Allow `python datasets/p3math/filter_corpus.py` from repo root or package dir.
_HERE = Path(__file__).resolve().parent
_DATASETS = _HERE.parent
if str(_DATASETS) not in sys.path:
    sys.path.insert(0, str(_DATASETS))

from p3math.filters import (  # noqa: E402
    ARXIV_MAX_CHARS,
    ARXIV_MIN_CHARS,
    ARXIV_MIN_PROOF_ENVS,
    count_proof_envs,
    is_english_arxiv_record,
    keep_algebraic_stack_p3,
    keep_arxiv_p3_text,
    keep_arxiv_pure_math_categories,
    keep_openwebmath_p3_record,
    normalize_arxiv_id,
    parse_categories,
    parse_meta,
    prepare_arxiv_text,
    primary_is_math,
)


def _iter_jsonl_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pat in ("**/*.jsonl.zst", "**/*.jsonl.gz", "**/*.jsonl", "**/*.parquet"):
        paths.extend(sorted(root.glob(pat)))
    # De-dupe while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _open_text_lines(path: Path) -> Iterator[str]:
    name = path.name.lower()
    if name.endswith(".jsonl.zst"):
        with path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                import io

                text = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text:
                    yield line
    elif name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield line
    elif name.endswith(".jsonl"):
        with path.open("rt", encoding="utf-8") as fh:
            for line in fh:
                yield line
    else:
        raise ValueError(f"unsupported jsonl path: {path}")


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".parquet" or path.name.endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        for batch in table.to_batches(max_chunksize=1024):
            rows = batch.to_pydict()
            n = len(next(iter(rows.values()))) if rows else 0
            keys = list(rows.keys())
            for i in range(n):
                yield {k: rows[k][i] for k in keys}
        return
    for line in _open_text_lines(path):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            yield obj


class ZstdJsonlWriter:
    def __init__(self, path: Path, level: int = 3) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("wb")
        self._cctx = zstd.ZstdCompressor(level=level)
        self._writer = self._cctx.stream_writer(self._fh)
        self.docs = 0
        self.chars = 0

    def write(self, obj: dict[str, Any]) -> None:
        text = obj.get("text") or ""
        line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self._writer.write(line)
        self.docs += 1
        self.chars += len(text)

    def close(self) -> None:
        # zstandard StreamWriter: close() finalizes the frame; flush(end_frame=...)
        # is not portable across versions.
        try:
            self._writer.flush()
        except Exception:
            pass
        self._writer.close()
        self._fh.close()


def load_arxiv_primary_math_ids(meta_root: Path) -> set[str]:
    """Load arXiv IDs whose primary category starts with math."""
    ids: set[str] = set()
    paths = _iter_jsonl_paths(meta_root)
    if not paths:
        paths = sorted(
            p
            for p in meta_root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".json", ".jsonl", ".parquet"}
            and ".cache" not in p.parts
        )
    print(f"[arxiv-meta] scanning {len(paths)} files under {meta_root}", flush=True)
    for path in paths:
        print(f"[arxiv-meta] reading {path}", flush=True)
        if path.suffix == ".parquet" or path.name.endswith(".parquet"):
            for row in _iter_records(path):
                cats = row.get("categories") or row.get("category")
                if not primary_is_math(cats):
                    continue
                aid = normalize_arxiv_id(row.get("id") or row.get("arxiv_id"))
                if aid:
                    ids.add(aid)
            continue
        # Prefer line-delimited JSON (Cornell OAI snapshot is JSONL with a .json suffix).
        n = 0
        try:
            if path.name.endswith(".gz"):
                fh_ctx = gzip.open(path, "rt", encoding="utf-8")
            else:
                fh_ctx = path.open("rt", encoding="utf-8")
            with fh_ctx as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line in {"[", "]", ","}:
                        continue
                    if line.endswith(","):
                        line = line[:-1]
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    n += 1
                    cats = row.get("categories") or row.get("category")
                    if not primary_is_math(cats):
                        continue
                    aid = normalize_arxiv_id(row.get("id") or row.get("arxiv_id"))
                    if aid:
                        ids.add(aid)
                    if n % 200_000 == 0:
                        print(f"[arxiv-meta] scanned {n:,} rows; math ids {len(ids):,}", flush=True)
        except Exception as exc:
            print(f"[arxiv-meta] stream failed on {path}: {exc}; trying whole-file JSON", flush=True)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc2:
                print(f"[arxiv-meta] skip {path}: {exc2}", flush=True)
                continue
            if isinstance(data, list):
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    cats = row.get("categories") or row.get("category")
                    if not primary_is_math(cats):
                        continue
                    aid = normalize_arxiv_id(row.get("id") or row.get("arxiv_id"))
                    if aid:
                        ids.add(aid)
        print(f"[arxiv-meta] finished {path.name}: scanned≈{n:,}, math ids so far {len(ids):,}", flush=True)
    print(f"[arxiv-meta] primary math.* ids: {len(ids):,}", flush=True)
    return ids


def load_arxiv_categories_by_id(meta_root: Path) -> dict[str, str]:
    """Map normalized arXiv id -> raw ``categories`` string from OAI snapshot."""
    paths = _iter_jsonl_paths(meta_root)
    if not paths:
        paths = sorted(
            p
            for p in meta_root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".json", ".jsonl", ".parquet"}
            and ".cache" not in p.parts
        )
    out: dict[str, str] = {}
    print(f"[arxiv-meta-cats] scanning {len(paths)} files under {meta_root}", flush=True)
    for path in paths:
        print(f"[arxiv-meta-cats] reading {path}", flush=True)
        n = 0
        if path.suffix == ".parquet" or path.name.endswith(".parquet"):
            for row in _iter_records(path):
                cats = row.get("categories") or row.get("category")
                if not cats:
                    continue
                aid = normalize_arxiv_id(row.get("id") or row.get("arxiv_id"))
                if aid:
                    out[aid] = cats if isinstance(cats, str) else " ".join(str(c) for c in cats)
            continue
        try:
            if path.name.endswith(".gz"):
                fh_ctx = gzip.open(path, "rt", encoding="utf-8")
            else:
                fh_ctx = path.open("rt", encoding="utf-8")
            with fh_ctx as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line in {"[", "]", ","}:
                        continue
                    if line.endswith(","):
                        line = line[:-1]
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    n += 1
                    cats = row.get("categories") or row.get("category")
                    if not cats:
                        continue
                    aid = normalize_arxiv_id(row.get("id") or row.get("arxiv_id"))
                    if aid:
                        out[aid] = cats if isinstance(cats, str) else " ".join(str(c) for c in cats)
                    if n % 500_000 == 0:
                        print(f"[arxiv-meta-cats] scanned {n:,}; mapped {len(out):,}", flush=True)
        except Exception as exc:
            print(f"[arxiv-meta-cats] skip {path}: {exc}", flush=True)
            continue
        print(f"[arxiv-meta-cats] finished {path.name}: scanned≈{n:,}, mapped so far {len(out):,}", flush=True)
    print(f"[arxiv-meta-cats] category map size: {len(out):,}", flush=True)
    return out


def filter_arxiv_pure(in_path: Path, out_path: Path, cats_by_id: dict[str, str]) -> dict[str, Any]:
    """Third pass: cross-list purity (all math.*) + pure-math subcategory allowlist."""
    writer = ZstdJsonlWriter(out_path)
    seen = kept = 0
    drop_missing_id = drop_missing_cats = drop_crosslist = drop_subcat = 0
    t0 = time.time()
    print(f"[arxiv-pure] reading {in_path}", flush=True)
    for rec in _iter_records(in_path):
        seen += 1
        aid = rec.get("arxiv_id") or parse_meta(rec).get("arxiv_id")
        aid = normalize_arxiv_id(aid) if aid else None
        if not aid:
            drop_missing_id += 1
            continue
        cats = cats_by_id.get(aid)
        if not cats:
            drop_missing_cats += 1
            continue
        parts = parse_categories(cats)
        if not all(p.lower().startswith("math.") for p in parts):
            drop_crosslist += 1
            continue
        if not keep_arxiv_pure_math_categories(cats):
            drop_subcat += 1
            continue
        meta = parse_meta(rec)
        meta["arxiv_id"] = aid
        meta["categories"] = cats
        out_rec: dict[str, Any] = {
            "text": rec.get("text") or "",
            "meta": meta,
            "source": "arxiv",
            "arxiv_id": aid,
            "categories": cats,
        }
        writer.write(out_rec)
        kept += 1
        if kept % 20_000 == 0:
            print(f"[arxiv-pure] kept {kept:,} / seen {seen:,}", flush=True)
    writer.close()
    return {
        "domain": "arxiv-pure",
        "seen_docs": seen,
        "kept_docs": kept,
        "drop_missing_id": drop_missing_id,
        "drop_missing_cats": drop_missing_cats,
        "drop_crosslist_non_math": drop_crosslist,
        "drop_subcat_not_allowlisted": drop_subcat,
        "kept_chars": writer.chars,
        "category_map_size": len(cats_by_id),
        "in": str(in_path),
        "out": str(out_path),
        "seconds": round(time.time() - t0, 1),
    }


def _record_arxiv_id(record: dict[str, Any]) -> str | None:
    meta = parse_meta(record)
    for key in ("arxiv_id", "id", "paper_id"):
        if key in meta:
            return normalize_arxiv_id(meta.get(key))
        if key in record:
            return normalize_arxiv_id(record.get(key))
    url = meta.get("url") or record.get("url") or ""
    if isinstance(url, str) and "arxiv.org" in url:
        # https://arxiv.org/abs/1234.5678
        tail = url.rstrip("/").split("/")[-1]
        return normalize_arxiv_id(tail)
    return None


def filter_openwebmath(raw_dir: Path, out_path: Path) -> dict[str, Any]:
    writer = ZstdJsonlWriter(out_path)
    seen = kept = 0
    t0 = time.time()
    for path in _iter_jsonl_paths(raw_dir):
        print(f"[owm] reading {path}", flush=True)
        for rec in _iter_records(path):
            seen += 1
            text = rec.get("text") or ""
            if not text or not keep_openwebmath_p3_record(rec, text):
                continue
            writer.write({"text": text, "meta": parse_meta(rec), "source": "open-web-math"})
            kept += 1
            if kept % 50_000 == 0:
                print(f"[owm] kept {kept:,} / seen {seen:,}", flush=True)
    writer.close()
    return {
        "domain": "open-web-math",
        "seen_docs": seen,
        "kept_docs": kept,
        "kept_chars": writer.chars,
        "out": str(out_path),
        "seconds": round(time.time() - t0, 1),
    }


def filter_algebraic_stack(raw_dir: Path, out_path: Path) -> dict[str, Any]:
    from p3math.filters import algebraic_stack_language

    writer = ZstdJsonlWriter(out_path)
    seen = kept = 0
    by_lang: dict[str, int] = {}
    t0 = time.time()
    for path in _iter_jsonl_paths(raw_dir):
        print(f"[alg] reading {path}", flush=True)
        for rec in _iter_records(path):
            seen += 1
            text = rec.get("text") or ""
            if not text or not keep_algebraic_stack_p3(rec, text, source_name=path.name):
                continue
            meta = parse_meta(rec)
            lang = algebraic_stack_language(rec, text, source_name=path.name) or "unknown"
            by_lang[lang] = by_lang.get(lang, 0) + 1
            writer.write({"text": text, "meta": meta, "source": "algebraic-stack", "language": lang})
            kept += 1
            if kept % 50_000 == 0:
                print(f"[alg] kept {kept:,} / seen {seen:,}", flush=True)
    writer.close()
    return {
        "domain": "algebraic-stack",
        "seen_docs": seen,
        "kept_docs": kept,
        "kept_chars": writer.chars,
        "by_language": by_lang,
        "out": str(out_path),
        "seconds": round(time.time() - t0, 1),
    }

def filter_arxiv(raw_dir: Path, out_path: Path, math_ids: set[str]) -> dict[str, Any]:
    writer = ZstdJsonlWriter(out_path)
    seen = kept = missing_id = 0
    drop_lang = drop_len = drop_proof = 0
    t0 = time.time()
    for path in _iter_jsonl_paths(raw_dir):
        print(f"[arxiv] reading {path}", flush=True)
        for rec in _iter_records(path):
            seen += 1
            aid = _record_arxiv_id(rec)
            if aid is None:
                missing_id += 1
                continue
            if aid not in math_ids:
                continue
            if not is_english_arxiv_record(rec):
                drop_lang += 1
                continue
            text = prepare_arxiv_text(rec.get("text") or "")
            if not keep_arxiv_p3_text(text):
                n = len(text)
                if n < ARXIV_MIN_CHARS or n > ARXIV_MAX_CHARS:
                    drop_len += 1
                elif count_proof_envs(text) < ARXIV_MIN_PROOF_ENVS:
                    drop_proof += 1
                else:
                    drop_len += 1
                continue
            meta = parse_meta(rec)
            meta["arxiv_id"] = aid
            writer.write({"text": text, "meta": meta, "source": "arxiv", "arxiv_id": aid})
            kept += 1
            if kept % 20_000 == 0:
                print(f"[arxiv] kept {kept:,} / seen {seen:,}", flush=True)
    writer.close()
    return {
        "domain": "arxiv",
        "seen_docs": seen,
        "kept_docs": kept,
        "missing_arxiv_id": missing_id,
        "drop_non_english": drop_lang,
        "drop_length": drop_len,
        "drop_proof_envs": drop_proof,
        "kept_chars": writer.chars,
        "math_id_universe": len(math_ids),
        "out": str(out_path),
        "seconds": round(time.time() - t0, 1),
    }


def refine_arxiv(in_path: Path, out_path: Path) -> dict[str, Any]:
    """Second pass on already math.*-filtered arXiv: English + cleanup + length/proof gates."""
    writer = ZstdJsonlWriter(out_path)
    seen = kept = 0
    drop_lang = drop_len = drop_proof = 0
    t0 = time.time()
    print(f"[arxiv-refine] reading {in_path}", flush=True)
    for rec in _iter_records(in_path):
        seen += 1
        if not is_english_arxiv_record(rec):
            drop_lang += 1
            continue
        # Input may already be biblio-stripped; re-run full prepare for noise strip.
        text = prepare_arxiv_text(rec.get("text") or "")
        if not keep_arxiv_p3_text(text):
            n = len(text)
            if n < ARXIV_MIN_CHARS or n > ARXIV_MAX_CHARS:
                drop_len += 1
            elif count_proof_envs(text) < ARXIV_MIN_PROOF_ENVS:
                drop_proof += 1
            else:
                drop_len += 1
            continue
        meta = parse_meta(rec)
        aid = rec.get("arxiv_id") or meta.get("arxiv_id")
        if aid:
            meta["arxiv_id"] = aid
        out_rec: dict[str, Any] = {"text": text, "meta": meta, "source": "arxiv"}
        if aid:
            out_rec["arxiv_id"] = aid
        writer.write(out_rec)
        kept += 1
        if kept % 20_000 == 0:
            print(f"[arxiv-refine] kept {kept:,} / seen {seen:,}", flush=True)
    writer.close()
    return {
        "domain": "arxiv-refine",
        "seen_docs": seen,
        "kept_docs": kept,
        "drop_non_english": drop_lang,
        "drop_length": drop_len,
        "drop_proof_envs": drop_proof,
        "kept_chars": writer.chars,
        "min_chars": ARXIV_MIN_CHARS,
        "max_chars": ARXIV_MAX_CHARS,
        "min_proof_envs": ARXIV_MIN_PROOF_ENVS,
        "in": str(in_path),
        "out": str(out_path),
        "seconds": round(time.time() - t0, 1),
    }


def copy_lean4(raw_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.name not in {".gitattributes"}:
            rel = path.relative_to(raw_dir)
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied.append(str(rel))
    return {"domain": "lean4-mathlib", "copied_files": len(copied), "out": str(out_dir)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch-root", type=Path, required=True)
    p.add_argument(
        "--domains",
        nargs="*",
        default=["open-web-math", "algebraic-stack", "arxiv", "lean4"],
        help="Domains to filter/copy (also: arxiv-refine, arxiv-pure)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a domain if its filtered output already exists and is non-empty",
    )
    args = p.parse_args(argv)

    root = args.scratch_root.resolve()
    raw = root / "raw"
    out = root / "filtered"
    out.mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {}
    domains = set(args.domains)

    def _exists(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    if "open-web-math" in domains:
        owm_out = out / "open-web-math" / "open-web-math.jsonl.zst"
        if args.skip_existing and _exists(owm_out):
            stats["open-web-math"] = {"domain": "open-web-math", "skipped": True, "out": str(owm_out)}
        else:
            stats["open-web-math"] = filter_openwebmath(raw / "proof-pile-2" / "open-web-math", owm_out)
        print(json.dumps(stats["open-web-math"], indent=2), flush=True)

    if "algebraic-stack" in domains:
        alg_out = out / "algebraic-stack" / "algebraic-stack.jsonl.zst"
        if args.skip_existing and _exists(alg_out):
            stats["algebraic-stack"] = {"domain": "algebraic-stack", "skipped": True, "out": str(alg_out)}
        else:
            stats["algebraic-stack"] = filter_algebraic_stack(raw / "proof-pile-2" / "algebraic-stack", alg_out)
        print(json.dumps(stats["algebraic-stack"], indent=2), flush=True)

    if "arxiv" in domains:
        arxiv_out = out / "arxiv" / "arxiv-math.jsonl.zst"
        if args.skip_existing and _exists(arxiv_out):
            stats["arxiv"] = {"domain": "arxiv", "skipped": True, "out": str(arxiv_out)}
        else:
            math_ids = load_arxiv_primary_math_ids(raw / "arxiv-metadata")
            id_path = root / "manifests" / "arxiv_primary_math_ids.txt"
            id_path.write_text("\n".join(sorted(math_ids)) + "\n", encoding="utf-8")
            stats["arxiv"] = filter_arxiv(raw / "proof-pile-2" / "arxiv", arxiv_out, math_ids)
        print(json.dumps(stats["arxiv"], indent=2), flush=True)

    if "arxiv-refine" in domains:
        arxiv_in = out / "arxiv" / "arxiv-math.jsonl.zst"
        arxiv_ref_out = out / "arxiv" / "arxiv-math-refined.jsonl.zst"
        if not _exists(arxiv_in):
            raise FileNotFoundError(f"arxiv-refine needs existing input: {arxiv_in}")
        if args.skip_existing and _exists(arxiv_ref_out):
            stats["arxiv-refine"] = {
                "domain": "arxiv-refine",
                "skipped": True,
                "out": str(arxiv_ref_out),
            }
        else:
            stats["arxiv-refine"] = refine_arxiv(arxiv_in, arxiv_ref_out)
        refine_path = root / "manifests" / "arxiv_refine_summary.json"
        refine_path.write_text(json.dumps(stats["arxiv-refine"], indent=2) + "\n", encoding="utf-8")
        print(json.dumps(stats["arxiv-refine"], indent=2), flush=True)

    if "arxiv-pure" in domains:
        arxiv_in = out / "arxiv" / "arxiv-math-refined.jsonl.zst"
        arxiv_pure_out = out / "arxiv" / "arxiv-math-pure.jsonl.zst"
        if not _exists(arxiv_in):
            raise FileNotFoundError(f"arxiv-pure needs existing input: {arxiv_in}")
        if args.skip_existing and _exists(arxiv_pure_out):
            stats["arxiv-pure"] = {
                "domain": "arxiv-pure",
                "skipped": True,
                "out": str(arxiv_pure_out),
            }
        else:
            cats_by_id = load_arxiv_categories_by_id(raw / "arxiv-metadata")
            cat_path = root / "manifests" / "arxiv_categories_by_id.json"
            # Store only ids we need would be huge; write size note instead of full map.
            cat_path.write_text(
                json.dumps({"entries": len(cats_by_id), "source": str(raw / "arxiv-metadata")}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            stats["arxiv-pure"] = filter_arxiv_pure(arxiv_in, arxiv_pure_out, cats_by_id)
        pure_path = root / "manifests" / "arxiv_pure_summary.json"
        pure_path.write_text(json.dumps(stats["arxiv-pure"], indent=2) + "\n", encoding="utf-8")
        print(json.dumps(stats["arxiv-pure"], indent=2), flush=True)

    if "lean4" in domains:
        lean_out = out / "lean4-mathlib"
        marker = lean_out / "data" / "train-00000-of-00001.parquet"
        if args.skip_existing and _exists(marker):
            stats["lean4"] = {"domain": "lean4-mathlib", "skipped": True, "out": str(lean_out)}
        else:
            stats["lean4"] = copy_lean4(raw / "lean4-mathlib", lean_out)
        print(json.dumps(stats["lean4"], indent=2), flush=True)

    summary_path = root / "manifests" / "filter_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
