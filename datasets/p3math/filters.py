"""P3Math keepers for Proof-Pile-2 subsets.

Settled cuts:
- OpenWebMath: math_score >= 0.72, native TeX signals >= 2, legacy share <= 50%,
  plus path-noise and image-only rejects.
- AlgebraicStack: RefHQ import-shell drop, then keep only formal langs + TeX.
- arXiv: primary category math.*, strip bibliography tails; then proof-env
  gate, length clamps, extra LaTeX noise strip, English-language keep;
  then cross-list purity (all cats math.*) + pure-math subcategory allowlist.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

# --- arXiv secondary cuts (proof gate + length + cleanup + English) ---

ARXIV_MIN_CHARS = 10_000
ARXIV_MAX_CHARS = 200_000
ARXIV_MIN_PROOF_ENVS = 2

_PROOF_ENV = re.compile(
    r"\\begin\{(?:theorem|lemma|proposition|corollary|proof)\*?\}",
    re.IGNORECASE,
)

# Environments / sections that bloat tokens without helping proof LM CPT.
_STRIP_ENVS = (
    "figure",
    "figure*",
    "table",
    "table*",
    "tikzpicture",
    "tabular",
    "tabular*",
)
_STRIP_ENV_RE = re.compile(
    r"\\begin\{(" + "|".join(re.escape(e) for e in _STRIP_ENVS) + r")\}"
    r".*?"
    r"\\end\{\1\}",
    re.IGNORECASE | re.DOTALL,
)
_INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{[^}]*\}", re.IGNORECASE)
_ACK_SECTION = re.compile(
    r"(?:"
    r"\\(?:section|subsection|chapter|paragraph)\*?\{[^}]*(?:acknowledg|funding|disclosure)[^}]*\}"
    r"|\\begin\{acknowledg(?:e)?ments?\}"
    r").*?"
    r"(?=\\(?:section|subsection|chapter|bibliography|begin\{thebibliography\}|end\{document\})\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_ENGLISH_LANG_CODES = frozenset({"en", "eng", "english"})

# --- shared path / import heuristics (same spirit as refhq.math_quality) ---

_UNC_PATH = re.compile(r"\\\\[a-zA-Z0-9_.$-]+\\")
_WINDOWS_PATH = re.compile(r"[a-zA-Z]:\\")
_FILE_LABEL = re.compile(r"(?:^|\s)file\s+\d+:", re.IGNORECASE)
_IMPORT_LINE = re.compile(r"^(?:open\s+)?import\s+")
_MODULE_LINE = re.compile(r"^module\s+")

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

# Slightly relaxed HQ score / native, strict legacy share.
OPENWEBMATH_P3_MIN_MATH_SCORE = 0.72
OPENWEBMATH_P3_MIN_NATIVE_SIGNALS = 2
OPENWEBMATH_P3_MAX_LEGACY_SHARE = 0.5

# Formal proof assistants + TeX from AlgebraicStack language table.
FORMAL_AND_TEX_LANGUAGES: frozenset[str] = frozenset(
    {
        "agda",
        "coq",
        "idris",
        "isabelle",
        "lean",
        "tex",
        "latex",  # alias if present in meta
    }
)

_BIB_START = re.compile(
    r"(?:"
    r"\\begin\{thebibliography\}"
    r"|\\bibliography\s*\{"
    r"|\\printbibliography\b"
    r"|\\addcontentsline\{toc\}\{chapter\}\{References\}"
    r"|\\addcontentsline\{toc\}\{section\}\{References\}"
    r")",
    re.IGNORECASE,
)
_BIB_END = re.compile(r"\\end\{thebibliography\}", re.IGNORECASE)


def _signal_sum(info: Mapping[str, Any], keys: tuple[str, ...]) -> int:
    return sum(int(info.get(key, 0) or 0) for key in keys)


def parse_meta(record: Mapping[str, Any]) -> dict[str, Any]:
    """Parse PP2 `meta` or OWM-style `metadata` into a dict."""
    raw = record.get("meta", record.get("metadata"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def extraction_info(record: Mapping[str, Any]) -> dict[str, Any]:
    meta = parse_meta(record)
    extraction = meta.get("extraction_info")
    if isinstance(extraction, dict):
        return extraction
    # Some dumps store extraction fields at the top level of meta.
    if any(k in meta for k in ("math_score", "mathjax_inline_tex", "img_math")):
        return meta
    return {}


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


def keep_openwebmath_p3_record(record: Mapping[str, Any], text: str) -> bool:
    """Relaxed math_score/native + strict legacy image share."""
    if is_file_path_noise(text):
        return False
    extraction = extraction_info(record)
    math_score = float(extraction.get("math_score", 0.0) or 0.0)
    if math_score < OPENWEBMATH_P3_MIN_MATH_SCORE:
        return False
    native = _signal_sum(extraction, _NATIVE_MATH_KEYS)
    if native < OPENWEBMATH_P3_MIN_NATIVE_SIGNALS:
        return False
    legacy = _signal_sum(extraction, _LEGACY_MATH_KEYS)
    total = native + legacy
    if total > 0 and legacy / total > OPENWEBMATH_P3_MAX_LEGACY_SHARE:
        return False
    # Drop image-only pages with no extractable TeX signals in metadata/text path.
    img_math = int(extraction.get("img_math", 0) or 0)
    tex_meta = native + _signal_sum(
        extraction,
        ("script_math_tex", "mathjax_tag", "codecogs_latex", "wp_latex"),
    )
    if tex_meta == 0 and img_math > 0:
        return False
    return True


_EXT_TO_LANG = {
    "agda": "agda",
    "v": "coq",
    "lean": "lean",
    "thy": "isabelle",
    "idr": "idris",
    "tex": "tex",
    "latex": "tex",
    "ltx": "tex",
}


def _lang_from_pathish(path: str) -> str | None:
    if not path or "." not in path:
        return None
    ext = path.rsplit(".", 1)[-1].lower()
    return _EXT_TO_LANG.get(ext)


def _lang_from_shard_name(source_name: str | None) -> str | None:
    """Infer language from AlgebraicStack shard filenames when meta lacks lang/ext."""
    if not source_name:
        return None
    name = source_name.lower()
    # proofstep dumps
    if "lean_proofsteps" in name or "github-lean" in name:
        return "lean"
    if "isa_proofsteps" in name or "github-isabelle" in name:
        return "isabelle"
    if "github-coq" in name or name.startswith("coq"):
        return "coq"
    for lang in ("agda", "idris", "isabelle", "lean", "tex", "coq"):
        if name.startswith(lang) or f"-{lang}-" in name or f"_{lang}_" in name:
            return "tex" if lang == "tex" else lang
    return None


def algebraic_stack_language(
    record: Mapping[str, Any],
    text: str = "",
    source_name: str | None = None,
) -> str | None:
    """Best-effort language label from PP2 / AlgebraicStack metadata or shard name."""
    meta = parse_meta(record)
    for key in ("language", "lang", "repo_language", "file_language"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            # "Jupyter Notebook" -> "jupyter notebook"
            return val.strip().lower()
    for nest_key in ("metadata", "source", "info"):
        nest = meta.get(nest_key)
        if isinstance(nest, dict):
            for key in ("language", "lang"):
                val = nest.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip().lower()
    for key in ("path", "filename", "file_name", "mathlib_filename", "save_path"):
        hit = _lang_from_pathish(str(meta.get(key) or ""))
        if hit:
            return hit
    hit = _lang_from_shard_name(source_name)
    if hit:
        return hit
    head = (text or "")[:200].lower()
    for lang in FORMAL_AND_TEX_LANGUAGES:
        if f"language: {lang}" in head or f"lang: {lang}" in head:
            return lang
    return None


def keep_algebraic_stack_p3(
    record: Mapping[str, Any],
    text: str,
    source_name: str | None = None,
) -> bool:
    """RefHQ import-shell filter + formal/TeX language allowlist."""
    if is_import_only_shell(text):
        return False
    lang = algebraic_stack_language(record, text, source_name=source_name)
    if lang is None:
        return False
    return lang in FORMAL_AND_TEX_LANGUAGES

# Pure-math arXiv subcategories (math.XX). Applied papers in ST/OC/NA/AP/… are dropped.
ARXIV_PURE_MATH_SUBCATS: frozenset[str] = frozenset(
    {
        "AG",  # Algebraic Geometry
        "AT",  # Algebraic Topology
        "CT",  # Category Theory
        "AC",  # Commutative Algebra
        "NT",  # Number Theory
        "GT",  # Geometric Topology
        "DG",  # Differential Geometry
        "CO",  # Combinatorics
        "LO",  # Logic
        "RT",  # Representation Theory
        "RA",  # Rings and Algebras
        "KT",  # K-Theory and Homology
        "GR",  # Group Theory
        "OA",  # Operator Algebras
        "QA",  # Quantum Algebra
        "SG",  # Symplectic Geometry
        "MG",  # Metric Geometry
        "GN",  # General Topology
        "FA",  # Functional Analysis
        "CV",  # Complex Variables
        "CA",  # Classical Analysis and ODEs
        "SP",  # Spectral Theory
        "PR",  # Probability
    }
)

# Explicitly out of the pure-math allowlist (documentation / drop targets).
ARXIV_APPLIED_MATH_SUBCATS: frozenset[str] = frozenset(
    {
        "ST",  # Statistics Theory
        "OC",  # Optimization and Control
        "NA",  # Numerical Analysis
        "AP",  # Analysis of PDEs
        "MP",  # Mathematical Physics
        "GM",  # General Mathematics
        "HO",  # History and Overview
        "DS",  # Dynamical Systems
    }
)


def parse_categories(categories: str | list[str] | None) -> list[str]:
    if categories is None:
        return []
    if isinstance(categories, list):
        return [str(c).strip() for c in categories if str(c).strip()]
    return [p for p in str(categories).replace(",", " ").split() if p]


def primary_category(categories: str | list[str] | None) -> str | None:
    parts = parse_categories(categories)
    return parts[0] if parts else None


def primary_is_math(categories: str | list[str] | None) -> bool:
    primary = primary_category(categories)
    return bool(primary and primary.startswith("math."))


def math_subcategory(cat: str) -> str | None:
    """Return XX for ``math.XX`` (case-insensitive); else None."""
    c = cat.strip()
    if not c.lower().startswith("math."):
        return None
    sub = c.split(".", 1)[1].strip().upper()
    return sub or None


def categories_all_math(categories: str | list[str] | None) -> bool:
    """Cross-list purity: every listed category must be ``math.*`` (no cs.*/physics.*/stat.*)."""
    parts = parse_categories(categories)
    if not parts:
        return False
    return all(p.lower().startswith("math.") for p in parts)


def keep_arxiv_pure_math_categories(categories: str | list[str] | None) -> bool:
    """All cats ``math.*`` and every subcategory in the pure-math allowlist."""
    if not categories_all_math(categories):
        return False
    for cat in parse_categories(categories):
        sub = math_subcategory(cat)
        if sub is None or sub not in ARXIV_PURE_MATH_SUBCATS:
            return False
    return True


def strip_bibliography(text: str) -> str:
    """Remove bibliography / references tails from LaTeX-ish arXiv text."""
    if not text:
        return text
    match = _BIB_START.search(text)
    if not match:
        return text
    start = match.start()
    end_match = _BIB_END.search(text, match.end())
    if end_match:
        # Remove only the thebibliography block when both ends exist.
        return (text[:start] + text[end_match.end() :]).rstrip() + "\n"
    # Otherwise drop from first bib marker to EOF (common for \\bibliography{...}).
    return text[:start].rstrip() + "\n"


def normalize_arxiv_id(raw: str | None) -> str | None:
    if not raw:
        return None
    s = str(raw).strip()
    s = s.replace("arXiv:", "").replace("arxiv:", "").strip()
    s = s.split("v")[0].strip() if re.match(r".*v\d+$", s) else s
    # Old-style IDs sometimes include category prefix: math/0101001
    return s or None


def count_proof_envs(text: str) -> int:
    """Count theorem/lemma/proposition/corollary/proof begin-envs."""
    if not text:
        return 0
    return len(_PROOF_ENV.findall(text))


def strip_arxiv_noise(text: str) -> str:
    """Drop figures/tables/tikz, includegraphics, and acknowledgment sections."""
    if not text:
        return text
    out = _STRIP_ENV_RE.sub("\n", text)
    out = _INCLUDEGRAPHICS.sub("", out)
    out = _ACK_SECTION.sub("\n", out)
    # Collapse runs of blank lines created by removals.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + ("\n" if out.strip() else "")


def is_english_arxiv_record(record: Mapping[str, Any]) -> bool:
    """Keep only records tagged English via PP2 `meta.language` (no abstract heuristic)."""
    meta = parse_meta(record)
    lang = meta.get("language") or record.get("language")
    if not isinstance(lang, str) or not lang.strip():
        return False
    return lang.strip().lower() in _ENGLISH_LANG_CODES


def keep_arxiv_p3_text(
    text: str,
    *,
    min_chars: int = ARXIV_MIN_CHARS,
    max_chars: int = ARXIV_MAX_CHARS,
    min_proof_envs: int = ARXIV_MIN_PROOF_ENVS,
) -> bool:
    """Length + proof-environment gate on cleaned arXiv body text."""
    if not text or not text.strip():
        return False
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    if count_proof_envs(text) < min_proof_envs:
        return False
    return True


def prepare_arxiv_text(text: str) -> str:
    """Bibliography strip then figure/ack noise strip."""
    return strip_arxiv_noise(strip_bibliography(text or ""))
