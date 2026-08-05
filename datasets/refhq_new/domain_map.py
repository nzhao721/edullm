"""Map instruct-row metadata to path domain labels for refhq-new."""

from __future__ import annotations

from typing import Any, Mapping

SOURCES: tuple[str, ...] = (
    "tulu-v2",
    "openhermes-25",
    "tulu-3",
    "hermes-3",
    "smoltalk",
    "dolci",
)

DOMAINS: tuple[str, ...] = ("general", "math", "code", "science", "chat")

DEFAULT_DOMAIN = "general"

# SmolTalk HF config → domain.
SMOLTALK_CONFIG_DOMAIN: dict[str, str] = {
    "numina-cot-100k": "math",
    "metamathqa-50k": "math",
    "self-oss-instruct": "code",
    "everyday-conversations": "chat",
    "systemchats-30k": "chat",
    "smol-magpie-ultra": "chat",
    "openhermes-100k": "general",
    "smol-summarize": "general",
    "smol-rewrite": "general",
    "explore-instruct-rewriting": "general",
    "longalign": "general",
}

# Substring → domain. First match wins within each list.
_TULU3_SOURCE_DOMAIN: tuple[tuple[str, str], ...] = (
    ("personas-math", "math"),
    ("personas_math", "math"),
    ("math-grade", "math"),
    ("personas-algebra", "math"),
    ("personas_algebra", "math"),
    ("numina", "math"),
    ("personas-code", "code"),
    ("personas_code", "code"),
    ("evol-codealpaca", "code"),
    ("codealpaca", "code"),
    ("sciriff", "science"),
    ("wildchat", "chat"),
    ("no_robots", "chat"),
    ("no-robots", "chat"),
    ("oasst", "chat"),
    ("flan", "general"),
    ("table-gpt", "general"),
    ("tablegpt", "general"),
    ("hard-coded", "general"),
    ("hard_coded", "general"),
)

_TULU2_DATASET_DOMAIN: tuple[tuple[str, str], ...] = (
    ("code_alpaca", "code"),
    ("codealpaca", "code"),
    ("science", "science"),
    ("sharegpt", "chat"),
    ("hard_coded", "chat"),
    ("cot", "general"),
    ("flan", "general"),
    ("open_orca", "general"),
    ("gpt4_alpaca", "general"),
    ("wizardlm", "general"),
    ("lima", "general"),
)

_OPENHERMES_CATEGORY_DOMAIN: tuple[tuple[str, str], ...] = (
    ("math", "math"),
    ("code", "code"),
    ("coding", "code"),
    ("science", "science"),
    ("roleplay", "chat"),
    ("general", "general"),
)

_DOLCI_DOMAIN_MAP: dict[str, str] = {
    "math": "math",
    "science": "science",
    "coding": "code",
    "code": "code",
    "chat": "chat",
    "other": "general",
    "general": "general",
}


def _first_substring_domain(haystack: str, rules: tuple[tuple[str, str], ...]) -> str | None:
    text = haystack.lower()
    for needle, domain in rules:
        if needle in text:
            return domain
    return None


def _normalize_domain(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    if key in DOMAINS:
        return key
    return _DOLCI_DOMAIN_MAP.get(key)


def map_domain(
    source: str,
    row: Mapping[str, Any] | None = None,
    *,
    smoltalk_config: str | None = None,
) -> str:
    """Return path domain label for one row (default ``general``)."""
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source!r}")
    row = row or {}

    if source == "smoltalk":
        config = (smoltalk_config or str(row.get("source") or "")).strip().lower()
        if config in SMOLTALK_CONFIG_DOMAIN:
            return SMOLTALK_CONFIG_DOMAIN[config]
        return DEFAULT_DOMAIN

    if source == "tulu-3":
        haystack = " ".join(
            str(row.get(key) or "") for key in ("source", "id")
        ).lower()
        mapped = _first_substring_domain(haystack, _TULU3_SOURCE_DOMAIN)
        return mapped or DEFAULT_DOMAIN

    if source == "tulu-v2":
        haystack = " ".join(
            str(row.get(key) or "") for key in ("dataset", "source", "id")
        ).lower()
        mapped = _first_substring_domain(haystack, _TULU2_DATASET_DOMAIN)
        return mapped or DEFAULT_DOMAIN

    if source == "dolci":
        mapped = _normalize_domain(str(row.get("domain") or "") or None)
        if mapped:
            return mapped
        haystack = str(row.get("source_dataset") or "").lower()
        if "math" in haystack:
            return "math"
        if "science" in haystack:
            return "science"
        if "code" in haystack or "python" in haystack or "coding" in haystack:
            return "code"
        if "wildchat" in haystack or "chat" in haystack:
            return "chat"
        return DEFAULT_DOMAIN

    if source == "openhermes-25":
        category = str(row.get("category") or row.get("source") or "").lower()
        mapped = _first_substring_domain(category, _OPENHERMES_CATEGORY_DOMAIN)
        return mapped or DEFAULT_DOMAIN

    # hermes-3: no reliable category column → general
    return DEFAULT_DOMAIN
