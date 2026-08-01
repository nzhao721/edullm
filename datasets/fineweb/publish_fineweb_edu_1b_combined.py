#!/usr/bin/env python3
"""Publish FineWeb-Edu 1B as one dataset: tokens (pretrain-tokens) + raw (vendored)."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

RUN = Path(os.environ["RUN_DIR"])
STAGE_DIR = Path(os.environ["STAGE_DIR"])  # original publish-stage with tokens+text
TOKENIZED_ROOT = Path(
    os.environ.get(
        "TOKENIZED_ROOT",
        "/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-tokenized",
    )
)
RAW_ROOT = Path(
    os.environ.get("RAW_ROOT", "/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-raw")
)
SESSION = Path(os.environ["AWS_SESSION_ENV"])
# Concrete HF commit for HuggingFaceFW/fineweb-edu @ main (API).
HF_REVISION = os.environ.get(
    "HF_FINEWEB_EDU_REVISION",
    "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
)
COMBINED = RUN / "publish-stage-combined"
SESSION_RELOAD_SECONDS = 300


def apply_session(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("export "):
            continue
        key, val = line[len("export ") :].split("=", 1)
        os.environ[key] = shlex.split(val)[0] if val else ""


class RefreshingBoto3S3:
    def __init__(self, session_env: Path) -> None:
        import threading

        import boto3
        from edullm_data.s3 import Boto3S3

        self._boto3 = boto3
        self._Boto3S3 = Boto3S3
        self.session_env = session_env
        self._lock = threading.Lock()
        self._inner = None
        self._loaded_at = 0.0
        self._mtime = 0.0
        self._rebuild(force=True)

    def _rebuild(self, *, force: bool = False) -> None:
        now = time.time()
        mtime = self.session_env.stat().st_mtime if self.session_env.is_file() else 0.0
        if (
            not force
            and self._inner is not None
            and (now - self._loaded_at) < SESSION_RELOAD_SECONDS
            and mtime <= self._mtime
        ):
            return
        apply_session(self.session_env)
        client = self._boto3.client(
            "s3",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["AWS_SESSION_TOKEN"],
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        self._inner = self._Boto3S3(client)
        self._loaded_at = now
        self._mtime = mtime
        print(f"S3 creds rotated (...{(os.environ.get('AWS_ACCESS_KEY_ID') or '')[-4:]})", flush=True)

    def _call(self, method: str, *args, **kwargs):
        with self._lock:
            self._rebuild()
            fn = getattr(self._inner, method)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "expired" not in msg and "invalidtoken" not in msg:
                raise
            with self._lock:
                self._rebuild(force=True)
                fn = getattr(self._inner, method)
            return fn(*args, **kwargs)

    def get(self, *a, **k):
        return self._call("get", *a, **k)

    def get_range(self, *a, **k):
        return self._call("get_range", *a, **k)

    def head(self, *a, **k):
        return self._call("head", *a, **k)

    def list(self, *a, **k):
        return self._call("list", *a, **k)

    def hash_object(self, *a, **k):
        return self._call("hash_object", *a, **k)

    def put(self, *a, **k):
        return self._call("put", *a, **k)

    def put_file(self, *a, **k):
        return self._call("put_file", *a, **k)

    def copy(self, *a, **k):
        return self._call("copy", *a, **k)

    def delete(self, *a, **k):
        return self._call("delete", *a, **k)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import edullm_data
    from edullm_data.contracts import validate_dataset_id
    from edullm_data.publish import publish

    print("edullm_data", getattr(edullm_data, "__version__", "?"), flush=True)
    if str(getattr(edullm_data, "__version__", "")).startswith("0.2"):
        raise SystemExit("refuse ancient edullm-data 0.2.x — install newest wheel")

    if len(HF_REVISION) < 12 or HF_REVISION.count("f0") > 3:
        raise SystemExit(f"HF_FINEWEB_EDU_REVISION looks placeholder: {HF_REVISION}")

    tokens_src = RUN / "publish-stage-tokens" / "tokens"
    if not tokens_src.is_dir():
        tokens_src = STAGE_DIR / "tokens"
    text_src = STAGE_DIR / "text" / "fineweb-edu"
    if not text_src.is_dir():
        raise SystemExit(f"missing text stage: {text_src}")
    if not tokens_src.is_dir():
        raise SystemExit(f"missing tokens stage: {tokens_src}")

    if COMBINED.exists():
        shutil.rmtree(COMBINED)
    COMBINED.mkdir(parents=True)
    shutil.copytree(tokens_src, COMBINED / "tokens")
    vendor_dst = COMBINED / "vendor"
    vendor_dst.mkdir(parents=True)
    upstream_files: list[dict] = []
    for src in sorted(text_src.glob("train-*.jsonl.gz")):
        dst = vendor_dst / src.name
        shutil.copy2(src, dst)
        digest = sha256_file(dst)
        upstream_files.append(
            {"path": src.name, "bytes": dst.stat().st_size, "sha256": digest}
        )
        print(f"vendor witness {src.name} bytes={dst.stat().st_size} sha256={digest[:12]}…", flush=True)
    if not upstream_files:
        raise SystemExit("no raw shards staged")

    tok_meta = json.loads((TOKENIZED_ROOT / "meta.json").read_text(encoding="utf-8"))
    raw_meta = json.loads((RAW_ROOT / "meta.json").read_text(encoding="utf-8"))
    retrieved_at = datetime.now(timezone.utc).isoformat()

    apply_session(SESSION)
    s3 = RefreshingBoto3S3(SESSION)
    validate_dataset_id("pretrain/fineweb-edu-1b")

    # edullm-data 0.6.3: publish(tokenizer=...) attaches depends_on to every key in
    # group_meta (not to pretrain-tokens groups). Include "tokens" so Gate A can derive
    # vocab_size on the token group. Vendor may also get the pin; that is harmless.
    plan = publish(
        COMBINED,
        dataset_id="pretrain/fineweb-edu-1b",
        purpose=(
            "1B-token FineWeb-Edu corpus with SmolLM2 token shards and a vendored raw JSONL "
            "companion for SmolLM2-135M ladder runs and FineWeb baselines"
        ),
        profile={"tokens": "pretrain-tokens/v1", "vendor": "vendored/v1"},
        tokenizer="tokenizer/smollm2-bpe",
        s3=s3,
        created_at=retrieved_at,
        hash_workers=8,
        copy_workers=8,
        group_meta={
            "tokens": {},
            "vendor": {
                "vendor_root": "vendor",
                "sentinels": [],
                "upstream": {
                    "name": "HuggingFaceFW/fineweb-edu",
                    "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
                    "revision": HF_REVISION,
                    "retrieved_at": retrieved_at,
                    "transport": {
                        "name": "HuggingFaceFW/fineweb-edu",
                        "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
                        "revision": HF_REVISION,
                    },
                },
                "upstream_files": upstream_files,
            },
        },
        about=(
            f"Token group: first {int(tok_meta['num_tokens']):,} SmolLM2 tokens from "
            f"{tok_meta['hf_path']} ({int(tok_meta['num_docs']):,} docs), nested "
            f"tokens/fineweb-edu/. Vendor group: id+text JSONL from "
            f"{raw_meta.get('hf_path')}/{raw_meta.get('hf_name')} "
            f"({int(raw_meta.get('num_docs') or 0):,} docs, ~1B-token budget). "
            "Tokenized and raw come from different FineWeb-Edu HF artifacts and are NOT "
            "document-aligned."
        ),
        sources=[
            {
                "name": "FineWeb-Edu 100BT-shuffled (tokenized prefix)",
                "tokens": int(tok_meta["num_tokens"]),
                "license": "ODC-By-1.0",
                "scope": "upstream-full-collection",
                "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT-shuffled",
            },
            {
                "name": "FineWeb-Edu sample-100BT (raw companion)",
                "tokens": int(raw_meta.get("num_tokens") or 0),
                "license": "ODC-By-1.0",
                "scope": "upstream-full-collection",
                "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
            },
        ],
        notes=(
            "Val carve: 0.15% into tokens/fineweb-edu/val-00000.u32le.bin. "
            "Raw companion uses vendored/v1 (text-corpus/v1 not in Gate A image). "
            f"Upstream revision pinned to {HF_REVISION}. "
            f"FarmShare legacy: {TOKENIZED_ROOT} + {RAW_ROOT}"
        ),
        license={"id": "ODC-By-1.0", "basis": "declared"},
    )
    out = {
        "dataset_id": plan.dataset_id,
        "version": plan.version,
        "payload_objects": len(plan.payload_keys),
    }
    print(json.dumps(out, indent=2), flush=True)
    (RUN / "combined_publish.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
