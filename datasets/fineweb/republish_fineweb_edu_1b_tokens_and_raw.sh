#!/usr/bin/env bash
# Republish FineWeb 1B: tokens-only with edullm-data 0.6.3, then raw text as vendor.
set -Eeuo pipefail
: "${RUN_DIR:?}" "${STAGE_DIR:?}" "${AWS_SESSION_ENV:?}"

TOKENIZED_ROOT="${TOKENIZED_ROOT:-/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-tokenized}"
RAW_ROOT="${RAW_ROOT:-/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-raw}"
VENV="${VENV:-${RUN_DIR}/venv}"
WHEEL_URI="${WHEEL_URI:-s3://edullm-landing/_dist/edullm_data-0.6.3-py3-none-any.whl}"

mkdir -p "${RUN_DIR}/logs"
cd "${RUN_DIR}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"
export PATH="${PATH}:${HOME}/.local/bin:${HOME}/tools/aws/bin"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
unset AWS_PROFILE || true

pip install -q -U pip wheel boto3
aws s3 cp "${WHEEL_URI}" "${RUN_DIR}/edullm_data-0.6.3-py3-none-any.whl"
pip install -q --force-reinstall "${RUN_DIR}/edullm_data-0.6.3-py3-none-any.whl"
python -c 'import edullm_data; print("edullm_data", getattr(edullm_data,"__version__", "?"), edullm_data.__file__)'
python -c 'import boto3; print(boto3.client("sts", region_name="us-east-1").get_caller_identity()["Arn"])'

# Tokens-only stage: keep existing staged tokens; park text aside
TOKENS_ONLY_STAGE="${RUN_DIR}/publish-stage-tokens"
rm -rf "${TOKENS_ONLY_STAGE}"
mkdir -p "${TOKENS_ONLY_STAGE}"
cp -a "${STAGE_DIR}/tokens" "${TOKENS_ONLY_STAGE}/tokens"

python - <<'PY'
from __future__ import annotations
import json, os, shlex, time
from datetime import datetime, timezone
from pathlib import Path

from edullm_data.contracts import validate_dataset_id
from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3

SESSION = Path(os.environ["AWS_SESSION_ENV"])
STAGE = Path(os.environ["TOKENS_ONLY_STAGE"])
RAW_ROOT = Path(os.environ["RAW_ROOT"])
TOKENIZED_ROOT = Path(os.environ["TOKENIZED_ROOT"])
TEXT_STAGE = Path(os.environ["STAGE_DIR"]) / "text"
RUN_DIR = Path(os.environ["RUN_DIR"])

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
        if not force and self._inner is not None and (now - self._loaded_at) < SESSION_RELOAD_SECONDS and mtime <= self._mtime:
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


tok_meta = json.loads((TOKENIZED_ROOT / "meta.json").read_text())
raw_meta = json.loads((RAW_ROOT / "meta.json").read_text())
apply_session(SESSION)
s3 = RefreshingBoto3S3(SESSION)

validate_dataset_id("pretrain/fineweb-edu-1b")
plan = publish(
    STAGE,
    dataset_id="pretrain/fineweb-edu-1b",
    purpose=(
        "1B-token FineWeb-Edu SmolLM2-tokenized corpus for SmolLM2-135M ladder runs "
        "and FineWeb baselines"
    ),
    profile="pretrain-tokens/v1",
    tokenizer="tokenizer/smollm2-bpe",
    s3=s3,
    created_at=datetime.now(timezone.utc).isoformat(),
    hash_workers=8,
    copy_workers=8,
    about=(
        f"First {int(tok_meta['num_tokens']):,} SmolLM2 tokens from "
        f"{tok_meta['hf_path']} ({int(tok_meta['num_docs']):,} docs). "
        "Shards nest as tokens/fineweb-edu/. Raw text companion is published separately "
        "as vendor/fineweb-edu-1b-raw (Gate A does not yet accept text-corpus/v1)."
    ),
    sources=[
        {
            "name": "FineWeb-Edu 100BT-shuffled (tokenized prefix)",
            "tokens": int(tok_meta["num_tokens"]),
            "license": "ODC-By-1.0",
            "scope": "upstream-full-collection",
            "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT-shuffled",
        }
    ],
    notes=(
        "Val carve: 0.15% into tokens/fineweb-edu/val-00000.u32le.bin. "
        "Raw JSONL companion: vendor/fineweb-edu-1b-raw (not document-aligned with these tokens). "
        f"FarmShare legacy: {TOKENIZED_ROOT}"
    ),
    license={"id": "ODC-By-1.0", "basis": "declared"},
)
print(json.dumps({"dataset_id": plan.dataset_id, "version": plan.version, "payload_objects": len(plan.payload_keys)}, indent=2), flush=True)
(RUN_DIR / "tokens_publish.json").write_text(
    json.dumps({"dataset_id": plan.dataset_id, "version": plan.version}, indent=2) + "\n"
)

# Raw text as vendored companion (text-corpus/v1 is not in the Gate A image).
if TEXT_STAGE.is_dir():
    import shutil

    vendor_stage = RUN_DIR / "publish-stage-text-vendor"
    if vendor_stage.exists():
        shutil.rmtree(vendor_stage)
    raw_root = vendor_stage / "raw"
    raw_root.mkdir(parents=True)
    for src in sorted((TEXT_STAGE / "fineweb-edu").glob("train-*.jsonl.gz")):
        shutil.copy2(src, raw_root / src.name)
    validate_dataset_id("vendor/fineweb-edu-1b-raw")
    apply_session(SESSION)
    s3 = RefreshingBoto3S3(SESSION)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    tplan = publish(
        vendor_stage,
        dataset_id="vendor/fineweb-edu-1b-raw",
        purpose=(
            "Byte-preserving FineWeb-Edu raw JSONL mirror (~1B-token budget) as the text "
            "companion for pretrain/fineweb-edu-1b"
        ),
        profile="vendored/v1",
        s3=s3,
        created_at=retrieved_at,
        hash_workers=8,
        copy_workers=8,
        group_meta={
            "raw": {
                "upstream": {
                    "name": "HuggingFaceFW/fineweb-edu",
                    "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
                    "retrieved_at": retrieved_at,
                    "transport": {
                        "name": "HuggingFaceFW/fineweb-edu",
                        "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
                        "revision": str(raw_meta.get("hf_name") or "sample-100BT"),
                    },
                },
                "vendor_root": "raw",
            }
        },
        about=(
            f"Raw documents from {raw_meta.get('hf_path')} / {raw_meta.get('hf_name')}: "
            f"{int(raw_meta.get('num_docs') or 0):,} docs (~1B-token budget), jsonl.gz with id+text. "
            "NOT document-aligned with pretrain/fineweb-edu-1b token shards "
            "(tokenized from fineweb_edu_100BT-shuffled)."
        ),
        sources=[
            {
                "name": "FineWeb-Edu sample-100BT",
                "tokens": int(raw_meta.get("num_tokens") or 0),
                "license": "ODC-By-1.0",
                "scope": "upstream-full-collection",
                "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
            }
        ],
        notes=(
            f"Companion to pretrain/fineweb-edu-1b. Records are id+text JSONL rewritten from the "
            f"FarmShare export. FarmShare legacy: {RAW_ROOT}"
        ),
        license={"id": "ODC-By-1.0", "basis": "declared"},
    )
    print(
        json.dumps(
            {
                "dataset_id": tplan.dataset_id,
                "version": tplan.version,
                "payload_objects": len(tplan.payload_keys),
            },
            indent=2,
        ),
        flush=True,
    )
    (RUN_DIR / "text_publish.json").write_text(
        json.dumps({"dataset_id": tplan.dataset_id, "version": tplan.version}, indent=2) + "\n"
    )
else:
    print("no text stage found; skipped vendor publish", flush=True)
PY
