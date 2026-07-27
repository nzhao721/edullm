#!/usr/bin/env python3
"""Smoke-check that copyright span removal leaves syntactically intact Python samples."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
from pathlib import Path


def _strip_copyright_heuristic(text: str) -> str:
    """Approximate Dolma copyright span strip for offline smoke tests without Dolma."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    # Drop leading block/line comments that look like license headers.
    i = 0
    while i < len(lines):
        s = lines[i].lstrip()
        if s.startswith("#!") or s.startswith("# -*-") or s.startswith("# coding"):
            i += 1
            continue
        if s.startswith("#") and any(
            k in s.lower() for k in ("copyright", "license", "spdx", "apache", "mit ", "gpl")
        ):
            i += 1
            continue
        if s.startswith('"""') or s.startswith("'''"):
            quote = s[:3]
            if s.count(quote) >= 2 and any(
                k in s.lower() for k in ("copyright", "license", "apache", "mit", "gpl")
            ):
                i += 1
                continue
            # multi-line docstring header
            j = i + 1
            block = s
            while j < len(lines) and quote not in lines[j]:
                block += lines[j]
                j += 1
            if j < len(lines):
                block += lines[j]
                j += 1
            if any(k in block.lower() for k in ("copyright", "license", "apache", "mit license", "gpl")):
                i = j
                continue
        break
    return "".join(lines[i:])


def check_python_docs(docs: list[str], limit: int = 50) -> dict:
    checked = 0
    ok = 0
    failed: list[str] = []
    for text in docs:
        if "def " not in text and "class " not in text:
            continue
        stripped = _strip_copyright_heuristic(text)
        checked += 1
        try:
            ast.parse(stripped)
            ok += 1
        except SyntaxError as exc:
            failed.append(str(exc))
        if checked >= limit:
            break
    return {"checked": checked, "ok": ok, "failed": len(failed), "examples": failed[:5]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, default=None, help="Optional .json.gz shard to check")
    parser.add_argument("--fixture", action="store_true", help="Run built-in fixture checks")
    args = parser.parse_args()

    if args.fixture or args.shard is None:
        fixture = '''\
# Copyright (c) 2023 Example Authors
# SPDX-License-Identifier: MIT
"""MIT License header."""

def add(a, b):
    return a + b
'''
        result = check_python_docs([fixture], limit=1)
        assert result["checked"] == 1 and result["ok"] == 1, result
        # Broken strip should fail: dangling quote
        broken = '"""\ndef add(a, b):\n    return a + b\n'
        bad = check_python_docs([broken], limit=1)
        print(json.dumps({"fixture_ok": result, "broken_case": bad}, indent=2))
        if result["ok"] != 1:
            return 1

    if args.shard is not None:
        docs: list[str] = []
        opener = gzip.open if str(args.shard).endswith(".gz") else open
        with opener(args.shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
                if text:
                    docs.append(text)
        result = check_python_docs(docs)
        print(json.dumps(result, indent=2))
        if result["checked"] == 0:
            print("no python-like docs found; smoke skipped")
            return 0
        # Allow a small failure rate from non-python misclassified files.
        if result["ok"] / result["checked"] < 0.9:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
