"""Resolve Dolma English config templates for refhq-new filtering."""

from __future__ import annotations

import argparse
from pathlib import Path

_ENGLISH_CONFIGS = {
    "taggers": "taggers-english.yaml",
    "pre-mix": "pre-mix-english.template.yaml",
}

ENGLISH_DOMAINS = ("english", "en")


def dolma_config_path(domain: str, kind: str, config_root: Path) -> Path:
    if kind not in _ENGLISH_CONFIGS:
        raise ValueError(f"unknown dolma config kind: {kind}")
    if domain not in ENGLISH_DOMAINS:
        raise ValueError(f"unsupported english dolma domain: {domain}")
    return config_root / _ENGLISH_CONFIGS[kind]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("path",))
    parser.add_argument("--domain", default="english")
    parser.add_argument("--kind", required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "path":
        print(dolma_config_path(args.domain, args.kind, args.config_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
