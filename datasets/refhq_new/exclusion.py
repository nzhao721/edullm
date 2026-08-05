"""Load and apply metadata exclusion rules for refhq-new."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RULES_PATH = PACKAGE_ROOT / "exclusion_rules.yaml"

SOURCE_IDS = (
    "tulu-v2",
    "openhermes-25",
    "tulu-3",
    "hermes-3",
    "smoltalk",
    "dolci",
)


def load_exclusion_rules(path: Path | None = None) -> dict[str, Any]:
    rules_path = path or DEFAULT_RULES_PATH
    raw = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "sources" not in raw:
        raise ValueError(f"exclusion rules must have a top-level 'sources' map: {rules_path}")
    sources = raw["sources"]
    if not isinstance(sources, dict):
        raise ValueError(f"'sources' must be a mapping: {rules_path}")
    missing = [name for name in SOURCE_IDS if name not in sources]
    if missing:
        raise ValueError(f"exclusion rules missing sources: {', '.join(missing)}")
    return raw


@lru_cache(maxsize=1)
def _default_rules() -> dict[str, Any]:
    return load_exclusion_rules()


def source_rules(source: str, rules: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    bundle = rules if rules is not None else _default_rules()
    sources = bundle["sources"]
    if source not in sources:
        raise ValueError(f"unknown source for exclusion rules: {source!r}")
    return sources[source]


def _haystack(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text.lower())
    return " ".join(parts)


def _contains_any(haystack: str, needles: list[str] | None) -> bool:
    if not needles or not haystack:
        return False
    return any(needle.lower() in haystack for needle in needles if needle)


def _is_non_null_tool_field(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def skip_smoltalk_config(config: str, rules: Mapping[str, Any] | None = None) -> bool:
    """Return True if this SmolTalk HF config should not be loaded."""
    cfg = source_rules("smoltalk", rules)
    skip = {str(name).strip().lower() for name in (cfg.get("skip_configs") or [])}
    return str(config).strip().lower() in skip


def keep_row(
    source: str,
    row: Mapping[str, Any],
    rules: Mapping[str, Any] | None = None,
    *,
    smoltalk_config: str | None = None,
) -> bool:
    """Return True if the row should be kept after metadata filters."""
    cfg = source_rules(source, rules)

    if source == "smoltalk":
        if smoltalk_config is not None and skip_smoltalk_config(smoltalk_config, rules):
            return False
        return True

    if source == "openhermes-25" and cfg.get("drop_non_english_language"):
        language = row.get("language")
        if language is not None and str(language).strip():
            allowed = {
                str(v).strip().lower()
                for v in (cfg.get("english_language_values") or ["en", "eng", "english"])
            }
            if str(language).strip().lower() not in allowed:
                return False

    if source == "tulu-3":
        haystack = _haystack(row.get("source"), row.get("id"))
        if _contains_any(haystack, cfg.get("drop_source_substrings")):
            return False
        return True

    if source == "dolci":
        domain = str(row.get("domain") or "").strip().lower()
        drop_domains = {str(d).strip().lower() for d in (cfg.get("drop_domains") or [])}
        if domain and domain in drop_domains:
            return False
        haystack = _haystack(row.get("source_dataset"), row.get("id"))
        if _contains_any(haystack, cfg.get("drop_source_dataset_substrings")):
            return False
        if cfg.get("drop_if_function_calls"):
            messages = row.get("messages") or row.get("conversations") or []
            if isinstance(messages, list):
                for message in messages:
                    if not isinstance(message, Mapping):
                        continue
                    if _is_non_null_tool_field(message.get("function_calls")):
                        return False
                    if _is_non_null_tool_field(message.get("functions")):
                        return False
        return True

    # tulu-v2 / hermes-3 / default: keep unless generic substring list matches.
    haystack = _haystack(row.get("source"), row.get("dataset"), row.get("source_dataset"), row.get("id"))
    if _contains_any(haystack, cfg.get("drop_source_substrings")):
        return False
    return True
