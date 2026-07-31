#!/usr/bin/env python3
"""Single-GPU smoke-test pretrain for SmolLM2-135M.

Token data is resolved from published+validated ``s3://edullm-data/`` via
``edullm_data.read.dataset_paths`` / ``resolve_latest``, then staged under the
job's output/stage dir. No FarmShare scratch corpus, laptop-local memmaps, or
legacy ``s3://edullm-datasets/`` are required.

Scratch is treated as ephemeral: durable artifacts (throughput.json, optional
checkpoint) must be uploaded before exit via ``--s3-output`` and/or W&B.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]

# Smallest published pretrain corpus (~310 MiB train shard, byte ids 0–255).
# FineWeb-Edu / SmolLM2-tokenized data is not published under edullm-data.
DEFAULT_DATASET_ID = "pretrain/lean4-mathlib-bytes"
LEGACY_DATA_BUCKET = "edullm-datasets"


@dataclass
class ThroughputStats:
    steps: int
    tokens_seen: int
    wall_time_s: float
    tokens_per_sec: float
    steps_per_sec: float
    final_loss: float
    peak_tokens_per_sec: float
    batch_size: int
    seq_len: int
    model: str
    dataset_id: str
    dataset_version: str
    stage_dir: str
    device: str
    s3_output: str | None = None
    wandb_run_url: str | None = None


class MemmapChunkDataset(Dataset):
    """Random-access fixed-length chunks over one or more flat uint32 token memmaps."""

    def __init__(self, memmaps: list[np.memmap], seq_len: int) -> None:
        if not memmaps:
            raise ValueError("need at least one token memmap")
        self.memmaps = memmaps
        self.seq_len = seq_len
        self._chunk_ends: list[int] = []
        total = 0
        for mm in memmaps:
            n = (len(mm) - 1) // seq_len
            if n <= 0:
                raise ValueError(f"memmap too short for seq_len={seq_len}: {len(mm)} tokens")
            total += n
            self._chunk_ends.append(total)
        self.num_chunks = total

    def __len__(self) -> int:
        return self.num_chunks

    def _locate(self, idx: int) -> tuple[np.memmap, int]:
        if idx < 0:
            idx += self.num_chunks
        prev = 0
        for mm, end in zip(self.memmaps, self._chunk_ends):
            if idx < end:
                local = idx - prev
                return mm, local * self.seq_len
            prev = end
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mm, start = self._locate(idx)
        input_ids = np.asarray(
            mm[start : start + self.seq_len],
            dtype=np.int64,
        )
        return {"input_ids": torch.from_numpy(input_ids.copy())}


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _dtype_from_name(name: str | None) -> np.dtype:
    if not name:
        raise SystemExit("dataset_paths returned no dtype; refuse to guess (need uint32)")
    return np.dtype(name)


def _refuse_legacy_data_ref(value: str, *, flag: str) -> None:
    lowered = value.replace("\\", "/").lower()
    if LEGACY_DATA_BUCKET in lowered:
        raise SystemExit(
            f"refusing legacy {LEGACY_DATA_BUCKET} reference in {flag}={value!r}; "
            "pass a published edullm-data dataset id"
        )
    if re.search(r"(^|/)(fineweb|smollm2[-_]?tokens?|edullm-cache)(/|$)", lowered):
        raise SystemExit(
            f"refusing hardcoded scratch corpus path in {flag}={value!r}; "
            "stage from edullm-data via --dataset-id instead"
        )
    if re.match(r"^(s3://|/|~|[a-z]:/)", lowered) and "pretrain/" not in lowered:
        # Path-like args are not dataset ids.
        if flag == "--dataset-id":
            raise SystemExit(
                f"refusing path-like --dataset-id={value!r}; "
                "expected family/name (e.g. pretrain/lean4-mathlib-bytes)"
            )


def resolve_published_paths(
    dataset_id: str,
    version: str | None,
    split: str,
) -> tuple[str, list[str], str, int | None]:
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    ver = version or resolve_latest(dataset_id, s3=s3)
    if not ver:
        raise SystemExit(f"no published versions found for {dataset_id!r} in edullm-data catalog")
    resolved = dataset_paths(dataset_id, ver, split=split, s3=s3)
    if not resolved.paths:
        raise SystemExit(f"no shard paths for {dataset_id}/{ver} split={split}")
    if not resolved.dtype:
        raise SystemExit(f"ambiguous/missing dtype for {dataset_id}/{ver}")
    for uri in resolved.paths:
        if LEGACY_DATA_BUCKET in uri:
            raise SystemExit(f"dataset_paths returned legacy URI {uri!r}; refuse to train")
    return ver, list(resolved.paths), resolved.dtype, resolved.rows


def stage_shards(
    s3_uris: list[str],
    stage_dir: Path,
    *,
    max_shards: int | None,
) -> list[Path]:
    """Download missing shards under stage_dir; skip when local size matches HEAD."""
    import boto3

    client = boto3.client("s3", region_name="us-east-1")
    stage_dir.mkdir(parents=True, exist_ok=True)
    selected = s3_uris if max_shards is None else s3_uris[: max(1, max_shards)]
    local_paths: list[Path] = []

    for uri in selected:
        bucket, key = _parse_s3_uri(uri)
        if bucket == LEGACY_DATA_BUCKET:
            raise SystemExit(f"refusing to stage from legacy bucket: {uri}")
        # key = <family>/<name>/<version>/... → stage under <version-relative path>
        parts = key.split("/")
        rel = Path(*parts[3:]) if len(parts) >= 4 else Path(parts[-1])
        dest = stage_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        head = client.head_object(Bucket=bucket, Key=key)
        remote_size = int(head["ContentLength"])
        if dest.exists() and dest.stat().st_size == remote_size:
            print(f"stage skip (size match): {dest}", flush=True)
        else:
            tmp = dest.with_suffix(dest.suffix + ".partial")
            print(f"stage download: s3://{bucket}/{key} -> {dest} ({remote_size} bytes)", flush=True)
            client.download_file(bucket, key, str(tmp))
            got = tmp.stat().st_size
            if got != remote_size:
                tmp.unlink(missing_ok=True)
                raise SystemExit(f"download size mismatch for {uri}: got {got} want {remote_size}")
            tmp.replace(dest)
        local_paths.append(dest)

    return local_paths


def open_token_memmaps(paths: list[Path], dtype: np.dtype) -> list[np.memmap]:
    memmaps: list[np.memmap] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        nbytes = path.stat().st_size
        itemsize = int(dtype.itemsize)
        if nbytes % itemsize != 0:
            raise SystemExit(f"{path} size {nbytes} not divisible by dtype {dtype}")
        n = nbytes // itemsize
        memmaps.append(np.memmap(path, dtype=dtype, mode="r", shape=(n,)))
    return memmaps


def upload_tree_to_s3(local_dir: Path, s3_uri: str) -> None:
    """Upload every file under local_dir to s3_uri (upload-before-end durable save)."""
    import boto3

    bucket, prefix = _parse_s3_uri(s3_uri if s3_uri.endswith("/") else s3_uri + "/")
    if bucket == LEGACY_DATA_BUCKET:
        raise SystemExit(f"refusing durable upload to legacy bucket: {s3_uri}")
    client = boto3.client("s3", region_name="us-east-1")
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    if not files:
        raise SystemExit(f"no files to upload under {local_dir}")
    for path in files:
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix}{rel}"
        print(f"s3 upload: {path} -> s3://{bucket}/{key}", flush=True)
        client.upload_file(str(path), bucket, key)
    print(f"uploaded {len(files)} file(s) to s3://{bucket}/{prefix}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SmolLM2-135M single-GPU smoke train from published edullm-data."
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Published edullm-data id (default: {DEFAULT_DATASET_ID})",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Exact version (e.g. v3). Default: resolve_latest from catalog.",
    )
    parser.add_argument("--split", default="train", help="Partition name (default: train)")
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Job-scoped cache for fetched .u32le.bin shards (default: <output-dir>/staged-data)",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Stage/use only the first N train shards (smoke on large corpora).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--s3-output",
        default=None,
        help="Durable prefix (s3://...) for throughput.json + checkpoint upload-before-end.",
    )
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--wandb-project", default="edullm-smollm2-smoke")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="disabled",
        choices=("online", "offline", "disabled"),
        help="W&B is an allowed durable remote when online/offline (default: disabled).",
    )
    parser.add_argument(
        "--allow-local-only",
        action="store_true",
        help="Escape hatch: allow scratch-only artifacts (not for FarmShare ephemeral jobs).",
    )
    return parser.parse_args()


def lr_at_step(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def _assert_token_ids_in_vocab(memmaps: list[np.memmap], vocab_size: int, sample: int = 1_000_000) -> None:
    """Fail fast if published token ids exceed the model embedding table (e.g. dolma2 on SmolLM2)."""
    seen = 0
    max_id = -1
    for mm in memmaps:
        take = min(len(mm), max(0, sample - seen))
        if take <= 0:
            break
        chunk_max = int(np.max(mm[:take]))
        max_id = max(max_id, chunk_max)
        seen += take
    if max_id >= vocab_size:
        raise SystemExit(
            f"token id {max_id} >= model vocab_size {vocab_size}. "
            f"This corpus is not compatible with {vocab_size}-way embeddings. "
            f"Use a byte-token dataset (e.g. {DEFAULT_DATASET_ID}) or a matching tokenizer/model."
        )


def wandb_enabled(args: argparse.Namespace) -> bool:
    return args.wandb_mode != "disabled" and wandb is not None and bool(os.environ.get("WANDB_API_KEY"))


def init_wandb(args: argparse.Namespace, config: dict) -> object | None:
    if args.wandb_mode == "disabled":
        return None
    if wandb is None:
        print("wandb package missing; continuing without W&B", flush=True)
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY unset; continuing without W&B", flush=True)
        return None
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name,
        config=config,
        dir=str(args.output_dir / "wandb"),
        resume="allow",
    )
    print(f"wandb run={run.id} url={run.url}", flush=True)
    return run


def require_durable_sink(args: argparse.Namespace) -> None:
    has_s3 = bool(args.s3_output)
    has_wandb = args.wandb_mode != "disabled"
    if args.allow_local_only:
        print("WARN: --allow-local-only set; scratch artifacts may be lost", flush=True)
        return
    if has_s3 or has_wandb:
        return
    raise SystemExit(
        "ephemeral scratch requires a durable sink: pass --s3-output s3://... "
        "and/or --wandb-mode online|offline (with WANDB_API_KEY), "
        "or --allow-local-only for non-FarmShare debugging"
    )


def main() -> None:
    args = parse_args()
    _refuse_legacy_data_ref(args.dataset_id, flag="--dataset-id")
    if args.s3_output:
        _refuse_legacy_data_ref(args.s3_output, flag="--s3-output")
        _parse_s3_uri(args.s3_output)
    require_durable_sink(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = args.stage_dir or (args.output_dir / "staged-data")
    _refuse_legacy_data_ref(str(stage_dir), flag="--stage-dir")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for this smoke test")

    version, s3_uris, dtype_name, rows = resolve_published_paths(
        args.dataset_id, args.dataset_version, args.split
    )
    dtype = _dtype_from_name(dtype_name)
    print(
        f"resolved {args.dataset_id}/{version} split={args.split} "
        f"shards={len(s3_uris)} dtype={dtype_name} rows={rows}",
        flush=True,
    )

    local_paths = stage_shards(s3_uris, stage_dir, max_shards=args.max_shards)
    memmaps = open_token_memmaps(local_paths, dtype)
    total_tokens = sum(len(mm) for mm in memmaps)
    print(f"staged {len(local_paths)} shard(s), {total_tokens:,} tokens under {stage_dir}", flush=True)

    seq_len = int(args.seq_len)
    dataset = MemmapChunkDataset(memmaps, seq_len=seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    config = AutoConfig.from_pretrained(args.model_id)
    _assert_token_ids_in_vocab(memmaps, int(config.vocab_size))
    model = AutoModelForCausalLM.from_config(config)
    model.to(device=device, dtype=torch.bfloat16)
    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    wb_run = init_wandb(
        args,
        {
            "dataset_id": args.dataset_id,
            "dataset_version": version,
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "seq_len": seq_len,
            "max_steps": args.max_steps,
            "s3_output": args.s3_output,
        },
    )
    if args.wandb_mode != "disabled" and wb_run is None and not args.s3_output and not args.allow_local_only:
        raise SystemExit(
            "wandb requested but unavailable and no --s3-output; "
            "fix WANDB_API_KEY / install wandb, or pass --s3-output"
        )

    tokens_per_step = args.batch_size * seq_len
    step = 0
    tokens_seen = 0
    running_loss = 0.0
    step_times: list[float] = []
    peak_tps = 0.0
    train_start = time.perf_counter()
    data_iter = iter(loader)
    loss = torch.tensor(0.0)

    print(
        f"train_smollm2_135m_smoke: dataset={args.dataset_id}/{version} "
        f"chunks={len(dataset):,} seq_len={seq_len} batch={args.batch_size} "
        f"max_steps={args.max_steps} device={device}",
        flush=True,
    )
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        step_start = time.perf_counter()
        input_ids = batch["input_ids"].to(device=device, non_blocking=True)
        labels = input_ids.clone()

        for group in optimizer.param_groups:
            group["lr"] = lr_at_step(step, args.lr, args.warmup_steps)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1
        tokens_seen += tokens_per_step
        running_loss += float(loss.detach())
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        step_tps = tokens_per_step / max(step_time, 1e-9)
        peak_tps = max(peak_tps, step_tps)

        if step % args.log_every == 0 or step == 1 or step == args.max_steps:
            elapsed = time.perf_counter() - train_start
            avg_tps = tokens_seen / max(elapsed, 1e-9)
            recent = step_times[-args.log_every :]
            recent_tps = tokens_per_step / max(sum(recent) / len(recent), 1e-9)
            avg_loss = running_loss / min(args.log_every, step)
            print(
                f"step={step:5d} loss={avg_loss:.4f} "
                f"recent_tps={recent_tps:,.0f} avg_tps={avg_tps:,.0f} "
                f"step_ms={1000.0 * step_time:.1f} lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
            if wb_run is not None:
                wb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/recent_tps": recent_tps,
                        "train/avg_tps": avg_tps,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )
            running_loss = 0.0

    wall_time = time.perf_counter() - train_start
    ckpt_dir: Path | None = None
    if args.save_checkpoint:
        ckpt_dir = args.output_dir / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        model_to_save.save_pretrained(ckpt_dir)
        print(f"saved checkpoint to {ckpt_dir}", flush=True)

    stats = ThroughputStats(
        steps=step,
        tokens_seen=tokens_seen,
        wall_time_s=wall_time,
        tokens_per_sec=tokens_seen / max(wall_time, 1e-9),
        steps_per_sec=step / max(wall_time, 1e-9),
        final_loss=float(loss.detach()),
        peak_tokens_per_sec=peak_tps,
        batch_size=args.batch_size,
        seq_len=seq_len,
        model=args.model_id,
        dataset_id=args.dataset_id,
        dataset_version=version,
        stage_dir=str(stage_dir),
        device=torch.cuda.get_device_name(device),
        s3_output=args.s3_output,
        wandb_run_url=getattr(wb_run, "url", None) if wb_run is not None else None,
    )
    stats_path = args.output_dir / "throughput.json"
    stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    print(json.dumps(asdict(stats), indent=2), flush=True)

    # Durable upload-before-end (scratch may be wiped after the job).
    if wb_run is not None:
        art = wandb.Artifact(name="smoke-throughput", type="metrics")
        art.add_file(str(stats_path), name=stats_path.name)
        wb_run.log_artifact(art)
        if ckpt_dir is not None:
            ckpt_art = wandb.Artifact(name="smoke-checkpoint", type="model")
            ckpt_art.add_dir(str(ckpt_dir))
            wb_run.log_artifact(ckpt_art)
            print(f"wandb uploaded checkpoint artifact {ckpt_art.name}", flush=True)
        wb_run.finish()

    if args.s3_output:
        # Upload metrics + checkpoint only (not staged token shards).
        durable_dir = args.output_dir / "_durable_upload"
        if durable_dir.exists():
            for p in durable_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
        durable_dir.mkdir(parents=True, exist_ok=True)
        (durable_dir / "throughput.json").write_text(stats_path.read_text(encoding="utf-8"), encoding="utf-8")
        if ckpt_dir is not None:
            import shutil

            shutil.copytree(ckpt_dir, durable_dir / "checkpoint", dirs_exist_ok=True)
        upload_tree_to_s3(durable_dir, args.s3_output)


if __name__ == "__main__":
    main()
