"""Heuristics for keeping mathematical text instead of file-path or import noise."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

_TEX_COMMAND = re.compile(
    r"\\(?:frac|sum|int|sqrt|mathcal|operatorname|bar|sigma|mu|alpha|beta|gamma|delta|theta|lambda|pi)\b"
)
_DOLLAR_MATH = re.compile(r"\$[^$\n]+\$")
_UNC_PATH = re.compile(r"\\\\[a-zA-Z0-9_.$-]+\\")
_WINDOWS_PATH = re.compile(r"[a-zA-Z]:\\")
_FILE_LABEL = re.compile(r"(?:^|\s)file\s+\d+:", re.IGNORECASE)
_IMPORT_LINE = re.compile(r"^(?:open\s+)?import\s+")
_MODULE_LINE = re.compile(r"^module\s+")


def _extraction_info(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("metadata")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            extraction = parsed.get("extraction_info")
            return extraction if isinstance(extraction, dict) else {}
        return {}
    if isinstance(raw, dict):
        extraction = raw.get("extraction_info")
        return extraction if isinstance(extraction, dict) else raw
    return {}


def count_tex_signals(text: str, extraction_info: Mapping[str, Any] | None = None) -> int:
    info = extraction_info or {}
    count = 0
    for key in (
        "script_math_tex",
        "mathjax_inline_tex",
        "mathjax_display_tex",
        "katex",
        "mathjax_tag",
        "align",
        "equation",
        "codecogs_latex",
        "wp_latex",
    ):
        count += int(info.get(key, 0) or 0)
    count += len(_DOLLAR_MATH.findall(text))
    count += len(_TEX_COMMAND.findall(text))
    return count


def is_file_path_noise(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    pathish = sum(
        1
        for line in lines
        if _UNC_PATH.search(line) or _WINDOWS_PATH.search(line) or _FILE_LABEL.search(line)
    )
    if pathish >= 5:
        return True
    return pathish >= 3 and pathish / len(lines) >= 0.15


def is_import_only_shell(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("--")]
    if len(lines) < 4:
        return False
    import_lines = sum(1 for line in lines if _IMPORT_LINE.match(line) or _MODULE_LINE.match(line))
    body_lines = [line for line in lines if not (_IMPORT_LINE.match(line) or _MODULE_LINE.match(line))]
    if import_lines >= 5 and len(body_lines) <= 2:
        return True
    if import_lines / len(lines) > 0.75:
        body_text = "\n".join(body_lines)
        has_definition = bool(
            re.search(
                r"\b(data|record|theorem|lemma|def|postulate|axiom|class|instance|structure)\b",
                body_text,
                re.IGNORECASE,
            )
            or ":=" in body_text
        )
        return not has_definition
    return False


_NATIVE_MATH_KEYS = (
    "mathjax_display_tex",
    "mathjax_inline_tex",
    "katex",
    "align",
    "equation",
)
_LEGACY_MATH_KEYS = (
    "codecogs_latex",
    "wp_latex",
    "img_math",
)


def _signal_sum(info: Mapping[str, Any], keys: tuple[str, ...]) -> int:
    return sum(int(info.get(key, 0) or 0) for key in keys)


def keep_openwebmath_record(record: Mapping[str, Any], text: str) -> bool:
    if is_file_path_noise(text):
        return False
    extraction = _extraction_info(record)
    math_score = float(extraction.get("math_score", 1.0) or 1.0)
    tex_signals = count_tex_signals(text, extraction)
    img_math = int(extraction.get("img_math", 0) or 0)
    if tex_signals == 0 and img_math > 0:
        return False
    if tex_signals == 0 and math_score < 0.5:
        return False
    if math_score < 0.25:
        return False
    return True


OPENWEBMATH_HQ_MIN_MATH_SCORE = 0.8
OPENWEBMATH_HQ_MIN_NATIVE_SIGNALS = 3
OPENWEBMATH_HQ_MAX_LEGACY_SHARE = 0.5


def keep_openwebmath_hq_record(record: Mapping[str, Any], text: str) -> bool:
    """HQ cut for reference-model OpenWebMath sampling."""
    if is_file_path_noise(text):
        return False
    extraction = _extraction_info(record)
    math_score = float(extraction.get("math_score", 0.0) or 0.0)
    if math_score < OPENWEBMATH_HQ_MIN_MATH_SCORE:
        return False
    native = _signal_sum(extraction, _NATIVE_MATH_KEYS)
    if native < OPENWEBMATH_HQ_MIN_NATIVE_SIGNALS:
        return False
    legacy = _signal_sum(extraction, _LEGACY_MATH_KEYS)
    total = native + legacy
    if total > 0 and legacy / total > OPENWEBMATH_HQ_MAX_LEGACY_SHARE:
        return False
    tex_signals = count_tex_signals(text, extraction)
    img_math = int(extraction.get("img_math", 0) or 0)
    if tex_signals == 0 and img_math > 0:
        return False
    return True


def keep_algebraic_stack_text(text: str) -> bool:
    return not is_import_only_shell(text)
