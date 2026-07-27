"""Resolve Dolma HQ config templates for code reference filtering."""

from __future__ import annotations

import argparse
from pathlib import Path

_CODE_HQ_CONFIGS = {
    "taggers": "taggers-code-hq.yaml",
    "pre-mix": "pre-mix-code-hq.template.yaml",
    "mix": "mix-code-hq-domain.template.yaml",
}


def dolma_config_path(domain: str, kind: str, config_root: Path) -> Path:
    if kind not in _CODE_HQ_CONFIGS:
        raise ValueError(f"unknown dolma config kind: {kind}")
    if domain not in ("code-hq", "starcoder-hq"):
        raise ValueError(f"unsupported HQ dolma domain: {domain}")
    return config_root / _CODE_HQ_CONFIGS[kind]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("path",))
    parser.add_argument("--domain", default="code-hq")
    parser.add_argument("--kind", required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "path":
        print(dolma_config_path(args.domain, args.kind, args.config_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
