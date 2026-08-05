#!/usr/bin/env python3
"""Concatenate tokenized part .npy files into <split>.npy per (source, domain).

Parts are headerless uint32 streams from TokenWriter, so byte-concat is valid.
Also mirrors merged outputs into tok/<source>/<domain>/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()


def _merge_parts(parts_dir: Path, out_npy: Path) -> dict:
    parts = sorted(parts_dir.glob("*.npy"))
    if not parts:
        raise SystemExit(f"no part npy under {parts_dir}")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    content_tokens = 0
    docs = 0
    with out_npy.open("wb") as out:
        for part in parts:
            data = part.read_bytes()
            if len(data) % 4 != 0:
                raise SystemExit(f"{part}: size {len(data)} not uint32-aligned")
            out.write(data)
            total_bytes += len(data)
            meta_path = part.with_suffix(".json")
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                content_tokens += int(meta.get("content_tokens") or 0)
                docs += int(meta.get("docs") or 0)
    stream_tokens = total_bytes // 4
    meta = {
        "parts": [str(p) for p in parts],
        "n_parts": len(parts),
        "docs": docs,
        "content_tokens": content_tokens,
        "stream_tokens_with_eos": stream_tokens,
        "tokenized_npy": str(out_npy),
    }
    out_npy.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def merge_all(plan: dict) -> dict:
    scratch = Path(plan["scratch_root"])
    summary: dict = {"merged": []}
    for source, src_plan in plan["sources"].items():
        tok_root = Path(src_plan["paths"]["tokenized"])
        if not tok_root.is_dir():
            continue
        for domain_dir in sorted(p for p in tok_root.iterdir() if p.is_dir()):
            domain = domain_dir.name
            for split in ("train", "val"):
                parts_dir = domain_dir / f"{split}.parts"
                if not parts_dir.is_dir():
                    continue
                out_npy = domain_dir / f"{split}.npy"
                print(f"merge {source}/{domain}/{split} from {parts_dir}", flush=True)
                meta = _merge_parts(parts_dir, out_npy)
                meta.update({"source": source, "domain": domain, "split": split})
                # Mirror for finalize_upload
                dest_dir = scratch / "tok" / source / domain
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_npy, dest_dir / out_npy.name)
                shutil.copy2(out_npy.with_suffix(".json"), dest_dir / f"{split}.json")
                summary["merged"].append(meta)
                print(
                    f"merged {source}/{domain}/{split} parts={meta['n_parts']} "
                    f"stream_tokens={meta['stream_tokens_with_eos']:,}",
                    flush=True,
                )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    summary = merge_all(plan)
    out = Path(plan["scratch_root"]) / "manifests" / "tokenize_merge_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"merged_count": len(summary["merged"]), "summary": str(out)}, indent=2))
    if not summary["merged"]:
        raise SystemExit("no tokenized parts found to merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
