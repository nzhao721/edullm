#!/usr/bin/env python3
"""Multi-GPU DDP pretraining for SmolLM2-135M on published edullm-data token shards.

Resolves a validated ``s3://edullm-data/<dataset_id>/<version>/`` corpus via
``edullm_data.read.dataset_paths`` / ``resolve_latest``, stages ``.u32le.bin``
shards into ``--stage-dir`` (job-scoped scratch; default ``<output-dir>/staged-data``),
then memmaps them for training.

Ephemeral-runtime contract:
  - Does not assume FarmShare scratch, laptop memmaps, or prior slice dirs exist.
  - Never reads the legacy ``s3://edullm-datasets/`` bucket.
  - Checkpoints must be durable off-box: ``--checkpoint-s3-uri`` and/or W&B online
    artifact upload (``WANDB_API_KEY`` + ``--wandb-mode online``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from eval_arc_task_loss_smollm import run_suite

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]

DEFAULT_DATASET_ID = "pretrain/fineweb-edu-500m"
LEGACY_DATA_BUCKET = "edullm-datasets"
DATA_BUCKET = "edullm-data"
CHECKPOINT_BUCKET = "edullm-checkpoints"
STAGE_MARKER = "_STAGED.json"


class MemmapChunkDataset(Dataset):
    """Contiguous uint token stream across one or more local memmap shards."""

    def __init__(
        self,
        shard_paths: list[Path],
        *,
        dtype: np.dtype,
        seq_len: int,
        num_tokens: int | None = None,
    ) -> None:
        if not shard_paths:
            raise ValueError("no token shards to load")
        self.shards: list[np.memmap] = []
        self.cumlen: list[int] = [0]
        for path in shard_paths:
            n = path.stat().st_size // dtype.itemsize
            if n <= 0:
                raise ValueError(f"empty token shard: {path}")
            self.shards.append(np.memmap(path, dtype=dtype, mode="r", shape=(n,)))
            self.cumlen.append(self.cumlen[-1] + n)
        total = self.cumlen[-1]
        if num_tokens is not None:
            if num_tokens > total:
                raise ValueError(f"declared rows={num_tokens} exceed staged tokens={total}")
            total = int(num_tokens)
        if total <= seq_len:
            raise ValueError(f"token stream too short for seq_len={seq_len}: {total} tokens")
        self.total_tokens = total
        self.seq_len = seq_len
        self.num_chunks = (total - 1) // seq_len

    def __len__(self) -> int:
        return self.num_chunks

    def _read_slice(self, start: int, end: int) -> np.ndarray:
        out = np.empty(end - start, dtype=np.int64)
        pos = 0
        for shard, c0, c1 in zip(self.shards, self.cumlen[:-1], self.cumlen[1:]):
            if end <= c0 or start >= c1:
                continue
            lo = max(start, c0) - c0
            hi = min(end, c1) - c0
            n = hi - lo
            out[pos : pos + n] = np.asarray(shard[lo:hi], dtype=np.int64)
            pos += n
            if pos >= end - start:
                break
        if pos != end - start:
            raise RuntimeError(f"failed to read tokens[{start}:{end}] across shards")
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += self.num_chunks
        start = idx * self.seq_len
        input_ids = self._read_slice(start, start + self.seq_len)
        return {"input_ids": torch.from_numpy(input_ids.copy())}


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if bucket == LEGACY_DATA_BUCKET:
        raise ValueError(
            f"refusing legacy bucket s3://{LEGACY_DATA_BUCKET}/ — use published s3://{DATA_BUCKET}/ only"
        )
    if bucket != DATA_BUCKET:
        raise ValueError(f"expected s3://{DATA_BUCKET}/..., got {uri!r}")
    return bucket, key


def _dtype_from_name(name: str | None) -> np.dtype:
    if not name:
        raise ValueError("dataset_paths returned no dtype; refuse to default (must be explicit uint32)")
    return np.dtype(name)


def resolve_edullm_split(
    dataset_id: str,
    *,
    version: str | None,
    split: str,
):
    try:
        from edullm_data.read import dataset_paths, resolve_latest
        from edullm_data.s3 import Boto3S3
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "edullm-data package is required. Install with:\n"
            '  pip install "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"'
        ) from exc

    s3 = Boto3S3.default()
    ver = version or resolve_latest(dataset_id, s3=s3)
    if not ver:
        raise SystemExit(
            f"no published versions for {dataset_id!r} under s3://{DATA_BUCKET}/_catalog/. "
            "Publish+validate the FineWeb-Edu 500M SmolLM2-tokenized corpus first."
        )
    resolved = dataset_paths(dataset_id, ver, split=split, s3=s3)
    if not resolved.paths:
        raise SystemExit(f"{dataset_id}/{ver} split={split!r} resolved to zero shard paths")
    for uri in resolved.paths:
        _parse_s3_uri(uri)
    return resolved


def _stage_ready(stage_dir: Path, expected: dict) -> bool:
    marker = stage_dir / STAGE_MARKER
    if not marker.exists():
        return False
    try:
        got = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        got.get("dataset_id") != expected["dataset_id"]
        or got.get("version") != expected["version"]
        or got.get("split") != expected["split"]
        or got.get("dtype") != expected["dtype"]
    ):
        return False
    shards = got.get("shards") or []
    if len(shards) != len(expected["shards"]):
        return False
    for shard, exp in zip(shards, expected["shards"]):
        path = Path(shard["local"])
        if not path.is_file() or path.stat().st_size != int(exp["bytes"]):
            return False
    return True


def stage_edullm_shards(
    resolved,
    stage_dir: Path,
    *,
    force: bool = False,
) -> dict:
    """Download validated edullm-data shards into stage_dir (idempotent)."""
    import boto3

    stage_dir.mkdir(parents=True, exist_ok=True)
    dtype_name = resolved.dtype or "uint32"
    client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

    shard_meta: list[dict] = []
    for uri in resolved.paths:
        bucket, key = _parse_s3_uri(uri)
        head = client.head_object(Bucket=bucket, Key=key)
        size = int(head["ContentLength"])
        # Keep relative path under stage_dir (tokens/.../train-00000.u32le.bin).
        rel = key.split(f"{resolved.dataset_id}/{resolved.version}/", 1)[-1]
        local = stage_dir / rel
        shard_meta.append({"uri": uri, "local": str(local), "bytes": size, "key": key, "bucket": bucket})

    expected = {
        "dataset_id": resolved.dataset_id,
        "version": resolved.version,
        "split": resolved.split,
        "dtype": dtype_name,
        "rows": resolved.rows,
        "shards": [{"uri": s["uri"], "local": s["local"], "bytes": s["bytes"]} for s in shard_meta],
    }
    if not force and _stage_ready(stage_dir, expected):
        print(f"stage cache hit under {stage_dir} ({len(shard_meta)} shards)", flush=True)
        return expected

    for shard in shard_meta:
        local = Path(shard["local"])
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.is_file() and local.stat().st_size == int(shard["bytes"]):
            print(f"skip existing {local} ({shard['bytes']:,} B)", flush=True)
            continue
        tmp = local.with_suffix(local.suffix + ".partial")
        print(f"fetching {shard['uri']} -> {local}", flush=True)
        client.download_file(shard["bucket"], shard["key"], str(tmp))
        if tmp.stat().st_size != int(shard["bytes"]):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"size mismatch downloading {shard['uri']}")
        tmp.replace(local)

    marker = stage_dir / STAGE_MARKER
    tmp_marker = marker.with_suffix(".tmp")
    tmp_marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    tmp_marker.replace(marker)
    print(f"staged {len(shard_meta)} shards for {resolved.dataset_id}/{resolved.version}", flush=True)
    return expected


def load_staged_corpus(stage_meta: dict, seq_len: int) -> tuple[MemmapChunkDataset, dict]:
    dtype = _dtype_from_name(stage_meta.get("dtype"))
    paths = [Path(s["local"]) for s in stage_meta["shards"]]
    rows = stage_meta.get("rows")
    dataset = MemmapChunkDataset(paths, dtype=dtype, seq_len=seq_len, num_tokens=rows)
    meta = {
        "dataset_id": stage_meta["dataset_id"],
        "version": stage_meta["version"],
        "split": stage_meta["split"],
        "dtype": stage_meta["dtype"],
        "num_tokens": dataset.total_tokens,
        "seq_len": seq_len,
        "n_shards": len(paths),
        "stage_shards": [str(p) for p in paths],
    }
    return dataset, meta


def _parse_checkpoint_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3:// URI for checkpoints, got {uri!r}")
    bucket = parsed.netloc
    if bucket == LEGACY_DATA_BUCKET:
        raise ValueError(f"refusing legacy bucket s3://{LEGACY_DATA_BUCKET}/ for checkpoints")
    prefix = parsed.path.lstrip("/").rstrip("/")
    return bucket, prefix


def upload_dir_to_s3(local_dir: Path, s3_uri: str) -> str:
    """Upload a local directory to s3://bucket/prefix/<dirname>/ via boto3."""
    import boto3

    bucket, prefix = _parse_checkpoint_s3_uri(s3_uri)
    key_prefix = f"{prefix}/{local_dir.name}" if prefix else local_dir.name
    client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    uploaded = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{key_prefix}/{rel}"
        client.upload_file(str(path), bucket, key)
        uploaded += 1
    dest = f"s3://{bucket}/{key_prefix}/"
    print(f"uploaded {uploaded} file(s) to {dest}", flush=True)
    return dest


def download_s3_prefix(s3_uri: str, dest_dir: Path) -> Path:
    """Download s3://bucket/prefix/ into dest_dir (flat contents of that prefix)."""
    import boto3

    bucket, prefix = _parse_checkpoint_s3_uri(s3_uri)
    if not prefix:
        raise ValueError(f"checkpoint S3 URI must include an object prefix: {s3_uri!r}")
    client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix if prefix.endswith("/") else prefix + "/"):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            out = dest_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"fetch resume {key} -> {out}", flush=True)
            client.download_file(bucket, key, str(out))
            count += 1
    if count == 0:
        raise FileNotFoundError(f"no objects under {s3_uri}")
    return dest_dir


def durable_backend_ok(args: argparse.Namespace) -> tuple[bool, str]:
    """Require S3 checkpoint URI and/or live W&B so scratch wipe is survivable."""
    has_s3 = bool(args.checkpoint_s3_uri)
    has_wandb = (
        args.wandb_mode == "online"
        and wandb is not None
        and bool(os.environ.get("WANDB_API_KEY"))
    )
    if has_s3 or has_wandb:
        parts = []
        if has_s3:
            parts.append(f"s3={args.checkpoint_s3_uri}")
        if has_wandb:
            parts.append("wandb=online")
        return True, "+".join(parts)
    return False, (
        "durable save required for ephemeral scratch: set --checkpoint-s3-uri "
        f"(e.g. s3://{CHECKPOINT_BUCKET}/smollm2/<run>/) and/or WANDB_API_KEY with --wandb-mode online"
    )


def rendezvous_file(path: Path, *, create: bool, timeout_s: float = 10_800.0) -> None:
    """Filesystem rendezvous so long rank-0 evals do not sit in NCCL barriers."""
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("done\n", encoding="utf-8")
        tmp.replace(path)
        return
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout_s:
            raise TimeoutError(f"timed out waiting for rendezvous file {path}")
        time.sleep(1.0)


def sync_after_side_work(output_dir: Path, tag: str) -> None:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    flag = output_dir / "progress" / f".sync_{tag}"
    if is_main_process():
        rendezvous_file(flag, create=True)
    else:
        rendezvous_file(flag, create=False)
    # Short collective once everyone is past the long rank-0 work.
    dist.barrier()
    if is_main_process() and flag.exists():
        flag.unlink(missing_ok=True)


def clear_sync_flag(output_dir: Path, tag: str) -> None:
    if not is_main_process():
        return
    flag = output_dir / "progress" / f".sync_{tag}"
    flag.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolLM2-135M multi-GPU pretrain.")
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Published edullm-data id (default: {DEFAULT_DATASET_ID}).",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Exact version (e.g. v1). Default: resolve_latest().",
    )
    parser.add_argument("--split", default="train", help="Partition name from dataset.json.")
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Job-scoped scratch for fetched edullm-data shards (default: <output-dir>/staged-data).",
    )
    parser.add_argument(
        "--restage",
        action="store_true",
        help="Force re-download of shards even if stage marker matches.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-s3-uri",
        default=os.environ.get("CHECKPOINT_S3_URI"),
        help=(
            "Durable S3 prefix for checkpoints/evals "
            f"(e.g. s3://{CHECKPOINT_BUCKET}/smollm2/<run>/). "
            "Required unless W&B online+API key is available."
        ),
    )
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--run-name", default="smollm2-135m-500m-40ep")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--max-train-tokens", type=int, default=20_000_000_000)
    parser.add_argument("--checkpoint-interval-epochs", type=float, default=0.5)
    parser.add_argument("--eval-interval-epochs", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint-interval-tokens",
        type=int,
        default=None,
        help="If set, overrides --checkpoint-interval-epochs (e.g. 250000000).",
    )
    parser.add_argument(
        "--eval-interval-tokens",
        type=int,
        default=None,
        help="If set, overrides --eval-interval-epochs (e.g. 250000000).",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Local checkpoint dir (must already be on this machine; prefer --resume-from-s3).",
    )
    parser.add_argument(
        "--resume-from-s3",
        default=None,
        help="s3:// URI of a checkpoint prefix to download into <output-dir>/resume_ckpt before train.",
    )
    parser.add_argument("--wandb-project", default="edullm-smollm2")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", "online"),
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument(
        "--wandb-upload-existing",
        action="store_true",
        help="On start, upload existing local checkpoints/evals as W&B artifacts.",
    )
    return parser.parse_args()


def wandb_enabled(args: argparse.Namespace) -> bool:
    return (
        is_main_process()
        and args.wandb_mode != "disabled"
        and wandb is not None
        and bool(os.environ.get("WANDB_API_KEY"))
    )


def init_wandb(args: argparse.Namespace, run_meta: dict) -> object | None:
    if not wandb_enabled(args):
        if is_main_process() and args.wandb_mode != "disabled" and wandb is None:
            print("wandb package missing; continuing without W&B", flush=True)
        elif is_main_process() and args.wandb_mode != "disabled" and not os.environ.get("WANDB_API_KEY"):
            print("WANDB_API_KEY unset; continuing without W&B", flush=True)
        return None
    assert wandb is not None
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    id_path = args.output_dir / "wandb_run_id.txt"
    run_id = id_path.read_text(encoding="utf-8").strip() if id_path.exists() else None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or args.run_name,
        id=run_id,
        resume="allow" if run_id else None,
        config={
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            **run_meta,
        },
        dir=str(args.output_dir / "wandb"),
    )
    id_path.write_text(str(run.id), encoding="utf-8")
    print(f"wandb run={run.id} url={run.url}", flush=True)
    # Smoke-test alerts as soon as the run is live.
    run.alert(
        title="smollm2 train job started",
        text=(
            f"run={run.name} id={run.id} "
            f"slurm_job={os.environ.get('SLURM_JOB_ID', 'n/a')} "
            f"host={os.environ.get('SLURMD_NODENAME', os.environ.get('HOSTNAME', 'n/a'))}"
        ),
        level=wandb.AlertLevel.INFO,
    )
    return run


def wandb_log(run: object | None, metrics: dict, *, step: int) -> None:
    if run is None:
        return
    run.log(metrics, step=step)


def wandb_log_eval(run: object | None, payload: dict, *, step: int, eval_path: Path) -> None:
    if run is None:
        return
    metrics: dict[str, float] = {"eval/macro_bpb": float(payload["macro_mean"])}
    if "macro_mean_accuracy" in payload:
        metrics["eval/macro_acc"] = float(payload["macro_mean_accuracy"])
    for k, v in (payload.get("labels") or {}).items():
        metrics[f"eval/bpb/{k}"] = float(v)
    for k, v in (payload.get("accuracy_labels") or {}).items():
        metrics[f"eval/acc/{k}"] = float(v)
    for k, v in (payload.get("task_families") or {}).items():
        metrics[f"eval/family_bpb/{k}"] = float(v)
    for k, v in (payload.get("accuracy_families") or {}).items():
        metrics[f"eval/family_acc/{k}"] = float(v)
    wandb_log(run, metrics, step=step)
    art = wandb.Artifact(name=f"eval-step{step:07d}", type="eval")
    art.add_file(str(eval_path), name=eval_path.name)
    run.log_artifact(art)


def wandb_log_checkpoint(run: object | None, ckpt_dir: Path, *, step: int, epoch: float, tokens_seen: int) -> None:
    if run is None:
        return
    wandb_log(
        run,
        {
            "checkpoint/step": step,
            "checkpoint/epoch": epoch,
            "checkpoint/tokens_seen": tokens_seen,
        },
        step=step,
    )
    art = wandb.Artifact(
        name=f"checkpoint-step{step:07d}",
        type="model",
        metadata={"step": step, "epoch": epoch, "tokens_seen": tokens_seen},
    )
    art.add_dir(str(ckpt_dir))
    run.log_artifact(art)
    print(f"wandb uploaded checkpoint artifact {art.name}", flush=True)


def wandb_upload_existing(run: object | None, output_dir: Path) -> None:
    if run is None:
        return
    ckpt_root = output_dir / "checkpoints"
    if ckpt_root.exists():
        for ckpt_dir in sorted(ckpt_root.glob("step*")):
            if not (ckpt_dir / "trainer_state.pt").exists():
                continue
            step = int(ckpt_dir.name.replace("step", ""))
            state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
            wandb_log_checkpoint(
                run,
                ckpt_dir,
                step=step,
                epoch=float(state.get("epoch", step)),
                tokens_seen=int(state.get("tokens_seen", 0)),
            )
    eval_root = output_dir / "task_loss"
    if eval_root.exists():
        for eval_path in sorted(eval_root.glob("step*_task_loss.json")):
            step = int(eval_path.name.split("_")[0].replace("step", ""))
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            wandb_log_eval(run, payload, step=step, eval_path=eval_path)
    progress = output_dir / "progress" / "task_loss.jsonl"
    if progress.exists():
        art = wandb.Artifact(name="task-loss-curve", type="metrics")
        art.add_file(str(progress), name=progress.name)
        run.log_artifact(art)
    meta = output_dir / "run_meta.json"
    if meta.exists():
        art = wandb.Artifact(name="run-meta", type="config")
        art.add_file(str(meta), name=meta.name)
        run.log_artifact(art)


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    epoch: float,
    tokens_seen: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
) -> Path:
    ckpt_dir = output_dir / "checkpoints" / f"step{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(ckpt_dir)
    state = {
        "step": step,
        "epoch": epoch,
        "tokens_seen": tokens_seen,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    torch.save(state, ckpt_dir / "trainer_state.pt")
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt_dir), encoding="utf-8")
    if args.checkpoint_s3_uri:
        dest = upload_dir_to_s3(ckpt_dir, args.checkpoint_s3_uri.rstrip("/") + "/checkpoints")
        (output_dir / "latest_checkpoint_s3.txt").write_text(dest + "\n", encoding="utf-8")
    return ckpt_dir


def persist_eval_artifact(args: argparse.Namespace, eval_path: Path) -> None:
    if not args.checkpoint_s3_uri:
        return
    import boto3

    bucket, prefix = _parse_checkpoint_s3_uri(args.checkpoint_s3_uri)
    key = f"{prefix}/task_loss/{eval_path.name}" if prefix else f"task_loss/{eval_path.name}"
    boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")).upload_file(
        str(eval_path), bucket, key
    )
    print(f"uploaded eval to s3://{bucket}/{key}", flush=True)


EVAL_TASKS = ("ARC-Easy", "ARC-Challenge", "HellaSwag")


def run_task_eval(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    out_path: Path,
    run_name: str,
    device: torch.device,
    *,
    checkpoint: str | None = None,
) -> dict | None:
    """Run ARC+HellaSwag eval on all ranks (sharded); rank 0 writes JSON."""
    model.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    payload = run_suite(
        model,
        tokenizer,
        list(EVAL_TASKS),
        n_shot=5,
        device=device,
        seed=42,
        run_name=run_name,
        checkpoint=checkpoint,
        rank=rank,
        world_size=world_size,
    )
    if is_main_process():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    model.train()
    return payload if is_main_process() else None


def eval_payload_has_hellaswag(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    labels = payload.get("labels") or {}
    return "hellaswag_val_rc_5shot_bpb" in labels


def append_task_loss_curve(progress_dir: Path, step: int, payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    record = {
        "step": step,
        "task_loss_bpb": payload["labels"],
        "accuracy": payload.get("accuracy_labels", {}),
    }
    path = progress_dir / "task_loss.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("step", -1)) != step:
                existing.append(row)
    existing.append(record)
    path.write_text("".join(json.dumps(r) + "\n" for r in existing), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if re.search(r"edullm-datasets", args.dataset_id):
        raise SystemExit("refusing legacy edullm-datasets paths; pass an edullm-data dataset id")
    if args.checkpoint_s3_uri:
        _parse_checkpoint_s3_uri(args.checkpoint_s3_uri)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        # Default NCCL timeout is 10m; ARC+HellaSwag eval on rank 0 exceeds that.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if args.stage_dir is None:
        args.stage_dir = args.output_dir / "staged-data"

    ok, detail = durable_backend_ok(args)
    if not ok:
        raise SystemExit(detail)

    if is_main_process():
        print(f"durable backend: {detail}", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "progress").mkdir(parents=True, exist_ok=True)
        args.stage_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from_s3:
        if is_main_process():
            local_resume = args.output_dir / "resume_ckpt"
            download_s3_prefix(args.resume_from_s3, local_resume)
            args.resume_from = local_resume
        if world_size > 1:
            dist.barrier()
        if args.resume_from is None:
            args.resume_from = args.output_dir / "resume_ckpt"

    seq_len = int(args.seq_len)
    # Resolve + stage from edullm-data on rank 0; all ranks open the shared stage dir.
    stage_meta: dict | None = None
    if is_main_process():
        resolved = resolve_edullm_split(
            args.dataset_id,
            version=args.dataset_version,
            split=args.split,
        )
        stage_meta = stage_edullm_shards(resolved, args.stage_dir, force=args.restage)
    if world_size > 1:
        dist.barrier()
    if stage_meta is None:
        marker = args.stage_dir / STAGE_MARKER
        if not marker.exists():
            raise FileNotFoundError(
                f"rank {dist.get_rank() if dist.is_initialized() else 0} missing stage marker {marker}; "
                "ensure --stage-dir is on a shared filesystem for multi-node jobs"
            )
        stage_meta = json.loads(marker.read_text(encoding="utf-8"))
    dataset, meta = load_staged_corpus(stage_meta, seq_len=seq_len)

    corpus_tokens = int(meta["num_tokens"])

    global_batch_tokens = args.per_device_batch_size * world_size * seq_len
    steps_per_epoch = math.ceil(corpus_tokens / global_batch_tokens)
    total_steps = min(args.num_epochs * steps_per_epoch, math.ceil(args.max_train_tokens / global_batch_tokens))
    if args.checkpoint_interval_tokens is not None:
        checkpoint_every = max(1, round(args.checkpoint_interval_tokens / global_batch_tokens))
    else:
        checkpoint_every = max(1, round(args.checkpoint_interval_epochs * steps_per_epoch))
    if args.eval_interval_tokens is not None:
        eval_every = max(1, round(args.eval_interval_tokens / global_batch_tokens))
    else:
        eval_every = max(1, round(args.eval_interval_epochs * steps_per_epoch))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    if is_main_process():
        run_meta = {
            "run_name": args.run_name,
            "corpus_tokens": corpus_tokens,
            "num_epochs": args.num_epochs,
            "max_train_tokens": args.max_train_tokens,
            "global_batch_tokens": global_batch_tokens,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "checkpoint_every_steps": checkpoint_every,
            "eval_every_steps": eval_every,
            "checkpoint_interval_tokens": args.checkpoint_interval_tokens,
            "eval_interval_tokens": args.eval_interval_tokens,
            "world_size": world_size,
            "dataset_id": args.dataset_id,
            "dataset_version": meta.get("version"),
            "split": args.split,
            "stage_dir": str(args.stage_dir),
            "checkpoint_s3_uri": args.checkpoint_s3_uri,
            "data_meta": meta,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        print(json.dumps(run_meta, indent=2), flush=True)
    else:
        run_meta = {}

    wb_run = init_wandb(args, run_meta)
    if wb_run is not None and args.wandb_upload_existing:
        print("uploading existing checkpoints/evals to wandb...", flush=True)
        wandb_upload_existing(wb_run, args.output_dir)

    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    config = AutoConfig.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_config(config)
    model.gradient_checkpointing_enable()
    model.to(device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_step = 0
    tokens_seen = 0
    if args.resume_from is not None:
        if not (args.resume_from / "trainer_state.pt").exists():
            raise FileNotFoundError(
                f"resume checkpoint missing trainer_state.pt under {args.resume_from}; "
                "fetch from --resume-from-s3 or push a durable checkpoint first"
            )
        state = torch.load(args.resume_from / "trainer_state.pt", map_location="cpu", weights_only=False)
        model_to_load = model.module if hasattr(model, "module") else model
        loaded = AutoModelForCausalLM.from_pretrained(args.resume_from, torch_dtype=torch.bfloat16)
        model_to_load.load_state_dict(loaded.state_dict())
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        tokens_seen = int(state.get("tokens_seen", start_step * global_batch_tokens))

    step = start_step
    running_loss = 0.0
    train_start = time.perf_counter()
    data_iter = iter(loader)
    next_checkpoint = ((start_step // checkpoint_every) + 1) * checkpoint_every
    next_eval = ((start_step // eval_every) + 1) * eval_every

    # On resume, re-run eval at the checkpoint step if HellaSwag (or any suite) is missing.
    if args.resume_from is not None:
        eval_out = args.output_dir / "task_loss" / f"step{start_step:07d}_task_loss.json"
        need_eval = False
        if is_main_process():
            need_eval = not eval_payload_has_hellaswag(eval_out)
        if world_size > 1:
            flag = torch.tensor([1 if need_eval else 0], device=device, dtype=torch.int32)
            dist.broadcast(flag, src=0)
            need_eval = bool(int(flag.item()))
        if need_eval:
            if is_main_process():
                print(f"running sharded task eval at resumed step {start_step}", flush=True)
            eval_model = model.module if hasattr(model, "module") else model
            run_task_eval(
                eval_model,
                tokenizer,
                eval_out,
                args.run_name,
                device,
                checkpoint=str(args.resume_from),
            )
            if is_main_process():
                append_task_loss_curve(args.output_dir / "progress", start_step, eval_out)
                payload = json.loads(eval_out.read_text(encoding="utf-8"))
                wandb_log_eval(wb_run, payload, step=start_step, eval_path=eval_out)
                persist_eval_artifact(args, eval_out)
                print(f"task loss eval wrote {eval_out}", flush=True)
            if world_size > 1:
                dist.barrier()
        next_eval = ((start_step // eval_every) + 1) * eval_every

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step // max(len(loader), 1))
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device=device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, labels=input_ids).loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1
        tokens_seen += global_batch_tokens
        running_loss += float(loss.detach())

        if step % args.log_every == 0 and is_main_process():
            elapsed = time.perf_counter() - train_start
            avg_tps = (tokens_seen - start_step * global_batch_tokens) / max(elapsed, 1e-9)
            epoch = step / steps_per_epoch
            loss_avg = running_loss / args.log_every
            lr = scheduler.get_last_lr()[0]
            print(
                f"step={step:7d}/{total_steps} epoch={epoch:6.3f} "
                f"loss={loss_avg:.4f} tokens_seen={tokens_seen:,} "
                f"avg_tps={avg_tps:,.0f} lr={lr:.2e}",
                flush=True,
            )
            wandb_log(
                wb_run,
                {
                    "train/loss": loss_avg,
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/tokens_seen": tokens_seen,
                    "train/avg_tps": avg_tps,
                },
                step=step,
            )
            running_loss = 0.0

        if step >= next_checkpoint:
            if is_main_process():
                epoch = step / steps_per_epoch
                ckpt_dir = save_checkpoint(
                    args.output_dir,
                    step=step,
                    epoch=epoch,
                    tokens_seen=tokens_seen,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                )
                print(f"saved checkpoint {ckpt_dir}", flush=True)
                wandb_log_checkpoint(wb_run, ckpt_dir, step=step, epoch=epoch, tokens_seen=tokens_seen)
            if world_size > 1:
                dist.barrier()
            next_checkpoint += checkpoint_every

        if step >= next_eval:
            if is_main_process():
                print(f"running sharded task eval at step {step}", flush=True)
            eval_out = args.output_dir / "task_loss" / f"step{step:07d}_task_loss.json"
            eval_model = model.module if hasattr(model, "module") else model
            run_task_eval(eval_model, tokenizer, eval_out, args.run_name, device)
            if is_main_process():
                append_task_loss_curve(args.output_dir / "progress", step, eval_out)
                payload = json.loads(eval_out.read_text(encoding="utf-8"))
                wandb_log_eval(wb_run, payload, step=step, eval_path=eval_out)
                persist_eval_artifact(args, eval_out)
                print(f"task loss eval wrote {eval_out}", flush=True)
            if world_size > 1:
                dist.barrier()
            next_eval += eval_every

    if is_main_process():
        final_ckpt = save_checkpoint(
            args.output_dir,
            step=step,
            epoch=step / steps_per_epoch,
            tokens_seen=tokens_seen,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )
        print(f"finished training at step={step}, checkpoint={final_ckpt}", flush=True)
        wandb_log_checkpoint(
            wb_run,
            final_ckpt,
            step=step,
            epoch=step / steps_per_epoch,
            tokens_seen=tokens_seen,
        )
        if args.checkpoint_s3_uri:
            meta_path = args.output_dir / "run_meta.json"
            if meta_path.exists():
                import boto3

                bucket, prefix = _parse_checkpoint_s3_uri(args.checkpoint_s3_uri)
                key = f"{prefix}/run_meta.json" if prefix else "run_meta.json"
                boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")).upload_file(
                    str(meta_path), bucket, key
                )
                print(f"uploaded run_meta to s3://{bucket}/{key}", flush=True)
        if wb_run is not None:
            wb_run.finish()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
