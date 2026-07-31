#!/usr/bin/env python3
"""Score one RegMix document chunk with RefHQ early and late-average checkpoints."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("WANDB_DISABLED", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import torch.nn.functional as F

# Compat for torch<2.7: olmo-core passes reason= to torch.compiler.disable.
_orig_torch_compiler_disable = torch.compiler.disable


def _torch_compiler_disable_compat(fn=None, recursive=True, **kwargs):
    kwargs.pop("reason", None)
    if fn is None:
        return lambda f: _orig_torch_compiler_disable(f, recursive=recursive)
    return _orig_torch_compiler_disable(fn, recursive=recursive)


torch.compiler.disable = _torch_compiler_disable_compat  # type: ignore[assignment]

from olmo.config import TrainConfig
from olmo.tokenizer import Tokenizer
from olmo.torch_util import get_local_rank
from olmo.util import add_cached_path_clients, prepare_cli_environment
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig

log = logging.getLogger("label_regmix_doc_lm")

EMBEDDING_SIZE = 100_352
IGNORE_INDEX = -100


@dataclass
class DocAccumulator:
    row: dict[str, Any]
    remaining_segments: int
    early_loss: float = 0.0
    late_loss: float = 0.0
    n_loss_tokens: int = 0


def stable_doc_id(domain: str, rel_path: str, line_index: int, text: str) -> str:
    digest = hashlib.sha1()
    digest.update(domain.encode())
    digest.update(b"\0")
    digest.update(rel_path.encode())
    digest.update(b"\0")
    digest.update(str(line_index).encode())
    digest.update(b"\0")
    digest.update(text.encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()


def shard_stem(index: int, path: str) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return f"lm-{index:05d}-{digest}"


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping) and value:
                return {str(k): v.detach().cpu() for k, v in value.items() if torch.is_tensor(v)}
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return {str(k): v.detach().cpu() for k, v in payload.items()}
    raise TypeError(f"could not find model tensor state in {path}")


def build_model() -> torch.nn.Module:
    try:
        backend = AttentionBackendName.torch
    except Exception:
        backend = None
    kwargs: dict[str, Any] = {"vocab_size": EMBEDDING_SIZE}
    if backend is not None:
        kwargs["attn_backend"] = backend
    cfg = TransformerConfig.olmo2_370M(**kwargs)
    model = cfg.build(init_device="cuda")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model()
    state = load_model_state(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("Missing %d keys from %s (showing 8): %s", len(missing), checkpoint, missing[:8])
    if unexpected:
        log.warning(
            "Unexpected %d keys from %s (showing 8): %s",
            len(unexpected),
            checkpoint,
            unexpected[:8],
        )
    if len(missing) > max(4, 0.05 * (len(state) + len(missing))):
        raise RuntimeError(f"too many missing keys for {checkpoint}: {len(missing)}")
    model.to(device)
    return model


def build_tokenizer(base_config: Path) -> Tokenizer:
    cfg = TrainConfig.load(str(base_config), [])
    cfg.evaluators = []
    return Tokenizer.from_train_config(cfg)


def tokenizer_encode(tokenizer: Tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "ids"):
        ids = ids.ids
    return [int(x) for x in ids]


def pad_id_from_tokenizer(tokenizer: Tokenizer) -> int:
    for attr in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            return value
    return 100_277


def iter_chunk_docs(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for chunk_row, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if isinstance(text, str) and text:
                yield chunk_row, int(row["source_line"]), int(row["source_doc"]), text


def make_segments(ids: list[int], seq_len: int) -> list[list[int]]:
    if len(ids) < 2:
        return []
    out: list[list[int]] = []
    start = 0
    while start < len(ids) - 1:
        end = min(len(ids), start + seq_len)
        out.append(ids[start:end])
        if end >= len(ids):
            break
        start = end - 1
    return out


def model_token_losses(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        out = model(
            input_ids,
            labels=labels,
            ignore_index=IGNORE_INDEX,
            loss_reduction="none",
            return_logits=False,
        )
    ce = getattr(out, "ce_loss", None)
    if torch.is_tensor(ce):
        return ce

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        logits_out = model(input_ids, return_logits=True)
    logits = logits_out.logits if hasattr(logits_out, "logits") else logits_out
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    vocab = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)
    return F.pad(loss, (0, 1), value=0.0)


def finite_exp(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if value > 80.0:
        return float("inf")
    return math.exp(value)


def finalize_doc(acc: DocAccumulator) -> dict[str, Any]:
    row = acc.row
    n = acc.n_loss_tokens
    if n > 0:
        early = acc.early_loss / n
        late = acc.late_loss / n
        learnability = late - early
    else:
        early = late = learnability = None
    return {
        **row,
        "n_loss_tokens": n,
        "early_step250_avg_nll": early,
        "early_step250_perplexity": finite_exp(early),
        "late_avg_steps_1000_1125_1315_avg_nll": late,
        "late_avg_steps_1000_1125_1315_perplexity": finite_exp(late),
        "avg_nll": late,
        "avg_perplexity": finite_exp(late),
        "learnability_late_minus_early_avg_nll": learnability,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--early-checkpoint", type=Path, required=True)
    parser.add_argument("--late-avg-checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lines = [line for line in args.work_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.index < 0 or args.index >= len(lines):
        print(f"index {args.index} out of range 0..{len(lines) - 1}", file=sys.stderr)
        return 2
    item = json.loads(lines[args.index])
    chunk_path = Path(item["path"])
    if not chunk_path.exists():
        print(f"missing source chunk: {chunk_path}", file=sys.stderr)
        return 1

    domain = item["domain"]
    source_rel_path = item["source_rel_path"]
    stem = shard_stem(args.index, str(chunk_path))
    docs_dir = args.out_root / "docs" / domain
    metrics_dir = args.out_root / "metrics" / domain
    docs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    docs_out = docs_dir / f"{stem}.jsonl.gz"
    metrics_out = metrics_dir / f"{stem}.metrics.jsonl.gz"
    done_marker = docs_dir / f"{stem}.done"
    if done_marker.exists() and docs_out.exists() and metrics_out.exists():
        print(json.dumps({"event": "skip_existing", "index": args.index, "docs": str(docs_out)}))
        return 0

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    prepare_cli_environment()
    add_cached_path_clients()
    device = torch.device("cuda")

    log.info("Loading tokenizer and models")
    tokenizer = build_tokenizer(args.base_config)
    pad_id = pad_id_from_tokenizer(tokenizer)
    early_model = load_model(args.early_checkpoint, device)
    late_model = load_model(args.late_avg_checkpoint, device)

    pending: list[tuple[str, list[int]]] = []
    accumulators: dict[str, DocAccumulator] = {}
    finished_rows: list[dict[str, Any]] = []
    n_docs = 0
    n_scored_tokens = 0

    def flush() -> None:
        nonlocal pending, n_scored_tokens
        if not pending:
            return
        max_len = max(len(ids) for _, ids in pending)
        batch = torch.full((len(pending), max_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full_like(batch, IGNORE_INDEX)
        masks: list[torch.Tensor] = []
        for row_idx, (_doc_key, ids) in enumerate(pending):
            seq = torch.tensor(ids, dtype=torch.long, device=device)
            length = seq.numel()
            batch[row_idx, :length] = seq
            if length > 1:
                labels[row_idx, : length - 1] = seq[1:]
            masks.append(labels[row_idx] != IGNORE_INDEX)

        early_ce = model_token_losses(early_model, batch, labels).float()
        late_ce = model_token_losses(late_model, batch, labels).float()

        for row_idx, (doc_key, _ids) in enumerate(pending):
            mask = masks[row_idx]
            count = int(mask.sum().item())
            acc = accumulators[doc_key]
            if count > 0:
                acc.early_loss += float(early_ce[row_idx][mask].sum().item())
                acc.late_loss += float(late_ce[row_idx][mask].sum().item())
                acc.n_loss_tokens += count
                n_scored_tokens += count
            acc.remaining_segments -= 1
            if acc.remaining_segments == 0:
                finished_rows.append(finalize_doc(acc))
                del accumulators[doc_key]
        pending = []

    def maybe_flush() -> None:
        total_tokens = sum(len(ids) for _, ids in pending)
        if total_tokens >= args.batch_tokens:
            flush()

    tmp_docs = Path(str(docs_out) + ".tmp")
    tmp_metrics = Path(str(metrics_out) + ".tmp")

    for chunk_row, source_line, source_doc, text in iter_chunk_docs(chunk_path):
        ids = tokenizer_encode(tokenizer, text)
        segments = make_segments(ids, args.seq_len)
        doc_key = f"{args.index}:{chunk_row}"
        base_row = {
            "id": stable_doc_id(domain, source_rel_path, source_line, text),
            "domain": domain,
            "source_path": source_rel_path,
            "source_line": source_line,
            "source_doc": source_doc,
            "chunk_index": item["chunk_index"],
            "chunk_row": chunk_row,
            "text": text,
            "n_tokens": len(ids),
        }
        if not segments:
            finished_rows.append(finalize_doc(DocAccumulator(base_row, remaining_segments=0)))
        else:
            accumulators[doc_key] = DocAccumulator(base_row, remaining_segments=len(segments))
            for seg in segments:
                pending.append((doc_key, seg))
                maybe_flush()
        n_docs += 1

    flush()
    if accumulators:
        raise RuntimeError(f"unfinished docs after flush: {len(accumulators)}")

    with gzip.open(tmp_docs, "wt", encoding="utf-8") as docs_handle, gzip.open(
        tmp_metrics, "wt", encoding="utf-8"
    ) as metrics_handle:
        for row in finished_rows:
            docs_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_handle.write(
                json.dumps({k: v for k, v in row.items() if k != "text"}, ensure_ascii=False) + "\n"
            )

    tmp_docs.replace(docs_out)
    tmp_metrics.replace(metrics_out)
    summary = {
        "event": "labeled_lm_chunk",
        "index": args.index,
        "domain": domain,
        "source_chunk": str(chunk_path),
        "docs": n_docs,
        "scored_tokens": n_scored_tokens,
        "docs_out": str(docs_out),
        "metrics_out": str(metrics_out),
        "seq_len": args.seq_len,
        "batch_tokens": args.batch_tokens,
    }
    done_marker.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
