#!/usr/bin/env python3
"""Normalize + metadata-filter one HF source into Dolma jsonl.gz docs by domain."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from refhq_new.domain_map import map_domain  # noqa: E402
from refhq_new.exclusion import keep_row, load_exclusion_rules, skip_smoltalk_config  # noqa: E402
from refhq_new_sources import DOCS_PER_SHARD  # noqa: E402

ROLE_LABELS: dict[str, str] = {
    "system": "System",
    "user": "User",
    "human": "User",
    "assistant": "Assistant",
    "gpt": "Assistant",
    "bot": "Assistant",
    "model": "Assistant",
    "tool": "Tool",
    "function": "Tool",
}


def _hf_token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _message_content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        content = message.get("value")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def flatten_conversation(row: Mapping[str, Any]) -> str:
    """Flatten messages/conversations to plain text for Dolma (one example = one doc)."""
    if isinstance(row.get("text"), str) and str(row["text"]).strip():
        # Prefer explicit text only when no chat turns exist.
        messages = row.get("messages") or row.get("conversations")
        if not messages:
            return str(row["text"]).strip()

    messages = row.get("messages") or row.get("conversations") or []
    if not isinstance(messages, list) or not messages:
        for key in ("prompt", "instruction", "input", "output", "response"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role_raw = str(message.get("role") or message.get("from") or "user").strip().lower()
        label = ROLE_LABELS.get(role_raw, role_raw.capitalize() or "User")
        content = _message_content(message)
        if not content:
            continue
        parts.append(f"{label}: {content}")
    return "\n\n".join(parts).strip()


class DomainShardWriter:
    """Write Dolma-shaped documents-*.jsonl.gz under docs/<source>/<domain>/."""

    def __init__(self, docs_root: Path, source: str, max_per_shard: int = DOCS_PER_SHARD) -> None:
        self.docs_root = docs_root
        self.source = source
        self.max_per_shard = max_per_shard
        self._handles: dict[str, Any] = {}
        self._counts: dict[str, int] = {}
        self._shard_idx: dict[str, int] = {}
        self._paths: dict[str, list[str]] = {}
        self.domain_counts: dict[str, int] = {}

    def _open_shard(self, domain: str) -> None:
        idx = self._shard_idx.get(domain, 0)
        out_dir = self.docs_root / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"documents-{idx:05d}.jsonl.gz"
        handle = gzip.open(path, "wt", encoding="utf-8")
        self._handles[domain] = handle
        self._counts[domain] = 0
        self._shard_idx[domain] = idx + 1
        self._paths.setdefault(domain, []).append(str(path))

    def write(self, domain: str, doc: dict[str, Any]) -> None:
        if domain not in self._handles or self._counts.get(domain, 0) >= self.max_per_shard:
            if domain in self._handles:
                self._handles[domain].close()
            self._open_shard(domain)
        handle = self._handles[domain]
        handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self._counts[domain] = self._counts.get(domain, 0) + 1
        self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1

    def close(self) -> dict[str, list[str]]:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        return dict(self._paths)


def _row_id(source: str, row: Mapping[str, Any], index: int, config: str | None = None) -> str:
    for key in ("id", "uuid", "conversation_id"):
        val = row.get(key)
        if val is not None and str(val).strip():
            base = str(val).strip()
            return f"{source}:{config}:{base}" if config else f"{source}:{base}"
    suffix = f"{config}-" if config else ""
    return f"{source}:{suffix}{index}"


def _iter_local_jsonl_rows(raw_dir: Path) -> Iterator[dict[str, Any]] | None:
    """If download stage left a json/jsonl snapshot, stream it without Hub features."""
    candidates: list[Path] = []
    for pattern in (
        "*.jsonl",
        "*.jsonl.gz",
        "data/*.jsonl",
        "data/*.jsonl.gz",
        "**/*dataset*.jsonl",
    ):
        candidates.extend(sorted(raw_dir.glob(pattern)))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    files: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        # Skip tiny metadata / readme-like files
        if path.stat().st_size < 1024:
            continue
        seen.add(resolved)
        files.append(path)
    if not files:
        return None

    def _gen() -> Iterator[dict[str, Any]]:
        for path in files:
            print(f"stream local file={path}", flush=True)
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                    if isinstance(row, dict):
                        yield row

    return _gen()


def _iter_hf_rows(
    *,
    repo_id: str,
    raw_dir: Path,
    token: str | None,
    config: str | None = None,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    # Prefer local jsonl from download stage (avoids Hub feature schemas like Image/Pillow).
    if config is None and raw_dir.is_dir():
        local = _iter_local_jsonl_rows(raw_dir)
        if local is not None:
            yield from local
            return

    kwargs: dict[str, Any] = {
        "path": repo_id,
        "split": "train",
        "streaming": True,
        "trust_remote_code": True,
    }
    if token:
        kwargs["token"] = token
    if config:
        kwargs["name"] = config

    try:
        ds = load_dataset(**kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        ds = load_dataset(**kwargs)

    for row in ds:
        if isinstance(row, dict):
            yield row


def _iter_source_rows(
    *,
    source: str,
    repo_id: str,
    raw_dir: Path,
    token: str | None,
    rules: Mapping[str, Any],
) -> Iterator[tuple[dict[str, Any], str | None]]:
    if source == "smoltalk":
        from datasets import get_dataset_config_names

        cfg_kwargs: dict[str, Any] = {"path": repo_id}
        if token:
            cfg_kwargs["token"] = token
        try:
            configs = get_dataset_config_names(**cfg_kwargs)
        except TypeError:
            cfg_kwargs.pop("token", None)
            configs = get_dataset_config_names(**cfg_kwargs)
        for config in sorted(str(c) for c in configs):
            if skip_smoltalk_config(config, rules):
                print(f"skip smoltalk config={config}", flush=True)
                continue
            print(f"stream smoltalk config={config}", flush=True)
            for row in _iter_hf_rows(repo_id=repo_id, raw_dir=raw_dir, token=token, config=config):
                yield row, config
        return

    print(f"stream source={source} repo={repo_id}", flush=True)
    for row in _iter_hf_rows(repo_id=repo_id, raw_dir=raw_dir, token=token, config=None):
        yield row, None


def _iter_fixture_rows(path: Path) -> Iterator[tuple[dict[str, Any], str | None]]:
    """Yield (row, smoltalk_config) from a local JSONL fixture (no HF)."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"fixture line {line_no}: expected object, got {type(row)}")
            config = row.pop("_smoltalk_config", None)
            if config is not None:
                config = str(config)
            yield row, config


def normalize_rows(
    *,
    source: str,
    repo_id: str,
    docs_root: Path,
    rows: Iterator[tuple[dict[str, Any], str | None]],
    max_docs: int | None = None,
) -> dict[str, Any]:
    """Normalize pre-loaded rows into Dolma docs (shared by HF stream + local fixtures)."""
    rules = load_exclusion_rules()
    writer = DomainShardWriter(docs_root, source)
    kept = 0
    dropped = 0
    empty_text = 0

    try:
        for index, (row, smoltalk_config) in enumerate(rows):
            if max_docs is not None and kept >= max_docs:
                break
            if not keep_row(source, row, rules, smoltalk_config=smoltalk_config):
                dropped += 1
                continue
            text = flatten_conversation(row)
            if not text:
                empty_text += 1
                continue
            domain = map_domain(source, row, smoltalk_config=smoltalk_config)
            doc = {
                "id": _row_id(source, row, index, smoltalk_config),
                "text": text,
                "source": source,
                "metadata": {
                    "domain": domain,
                    "hf_repo": repo_id,
                    "smoltalk_config": smoltalk_config,
                    "row_source": row.get("source") or row.get("dataset") or row.get("source_dataset"),
                },
            }
            writer.write(domain, doc)
            kept += 1
            if kept % 50_000 == 0:
                print(
                    f"progress source={source} kept={kept:,} dropped={dropped:,} empty={empty_text:,}",
                    flush=True,
                )
    finally:
        shard_paths = writer.close()

    return {
        "source": source,
        "hf_repo": repo_id,
        "kept": kept,
        "dropped_metadata": dropped,
        "dropped_empty_text": empty_text,
        "domain_counts": writer.domain_counts,
        "shards": shard_paths,
        "docs_root": str(docs_root),
    }


def normalize_source(
    *,
    source: str,
    repo_id: str,
    raw_dir: Path,
    docs_root: Path,
    token: str | None,
    max_docs: int | None = None,
) -> dict[str, Any]:
    rules = load_exclusion_rules()
    return normalize_rows(
        source=source,
        repo_id=repo_id,
        docs_root=docs_root,
        rows=_iter_source_rows(
            source=source,
            repo_id=repo_id,
            raw_dir=raw_dir,
            token=token,
            rules=rules,
        ),
        max_docs=max_docs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument(
        "--fixture-jsonl",
        type=Path,
        default=None,
        help="Local JSONL(.gz) rows instead of HF (for offline smoke)",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.source not in plan["sources"]:
        raise SystemExit(f"source {args.source} missing from plan")
    src_plan = plan["sources"][args.source]
    raw_dir = Path(src_plan["paths"]["raw"])
    docs_root = Path(src_plan["paths"]["docs"])
    docs_root.mkdir(parents=True, exist_ok=True)

    # Clear prior normalize outputs for this source.
    for old in docs_root.glob("*"):
        if old.is_dir():
            for shard in old.glob("documents-*.jsonl.gz"):
                shard.unlink()

    if args.fixture_jsonl is not None:
        if not args.fixture_jsonl.is_file():
            raise SystemExit(f"fixture not found: {args.fixture_jsonl}")
        print(f"normalize from fixture={args.fixture_jsonl}", flush=True)
        stats = normalize_rows(
            source=args.source,
            repo_id=src_plan["hf_repo"],
            docs_root=docs_root,
            rows=_iter_fixture_rows(args.fixture_jsonl),
            max_docs=args.max_docs,
        )
    else:
        marker = raw_dir / "_download_complete.json"
        if not marker.is_file():
            print(f"WARN: missing {marker}; normalize will stream from Hub", flush=True)
        stats = normalize_source(
            source=args.source,
            repo_id=src_plan["hf_repo"],
            raw_dir=raw_dir,
            docs_root=docs_root,
            token=_hf_token(args.token),
            max_docs=args.max_docs,
        )
    stats_path = Path(src_plan["paths"]["stats"]).with_name(f"{args.source}-normalize.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
