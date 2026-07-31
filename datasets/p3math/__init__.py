"""P3Math: filtered Proof-Pile-2 math corpus + Lean4 Mathlib."""

from .filters import (
    ARXIV_PURE_MATH_SUBCATS,
    FORMAL_AND_TEX_LANGUAGES,
    categories_all_math,
    is_english_arxiv_record,
    keep_algebraic_stack_p3,
    keep_arxiv_p3_text,
    keep_arxiv_pure_math_categories,
    keep_openwebmath_p3_record,
    prepare_arxiv_text,
    primary_is_math,
    strip_bibliography,
)

__all__ = [
    "ARXIV_PURE_MATH_SUBCATS",
    "FORMAL_AND_TEX_LANGUAGES",
    "categories_all_math",
    "is_english_arxiv_record",
    "keep_algebraic_stack_p3",
    "keep_arxiv_p3_text",
    "keep_arxiv_pure_math_categories",
    "keep_openwebmath_p3_record",
    "prepare_arxiv_text",
    "primary_is_math",
    "strip_bibliography",
]
