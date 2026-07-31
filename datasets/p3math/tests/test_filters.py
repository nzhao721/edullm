"""Unit tests for P3Math filters."""

from __future__ import annotations

import json

from p3math.filters import (
    count_proof_envs,
    is_english_arxiv_record,
    keep_algebraic_stack_p3,
    keep_arxiv_p3_text,
    keep_openwebmath_p3_record,
    prepare_arxiv_text,
    primary_is_math,
    strip_bibliography,
)


def _owm_rec(math_score: float, native: int = 3, legacy: int = 0, img_math: int = 0) -> dict:
    extraction = {
        "math_score": math_score,
        "mathjax_inline_tex": native,
        "codecogs_latex": legacy,
        "img_math": img_math,
    }
    return {"meta": json.dumps({"extraction_info": extraction})}


def test_owm_relaxed_thresholds() -> None:
    text = "Some math page with content about integrals."
    assert keep_openwebmath_p3_record(_owm_rec(0.72, native=2), text)
    assert not keep_openwebmath_p3_record(_owm_rec(0.71, native=2), text)
    assert not keep_openwebmath_p3_record(_owm_rec(0.9, native=1), text)


def test_owm_legacy_share_strict() -> None:
    text = "legacy heavy page"
    # native=2, legacy=3 -> legacy share 0.6 > 0.5
    assert not keep_openwebmath_p3_record(_owm_rec(0.9, native=2, legacy=3), text)
    assert keep_openwebmath_p3_record(_owm_rec(0.9, native=3, legacy=3), text)


def test_alg_formal_language_and_import_shell() -> None:
    lean_ok = {
        "meta": json.dumps({"language": "lean"}),
        "text": "theorem foo : True := by trivial\n" * 5,
    }
    assert keep_algebraic_stack_p3(lean_ok, lean_ok["text"])

    py_bad = {
        "meta": json.dumps({"language": "python"}),
        "text": "import numpy as np\n" + "x = 1\n" * 10,
    }
    assert not keep_algebraic_stack_p3(py_bad, py_bad["text"])

    shell = {
        "meta": json.dumps({"language": "lean"}),
        "text": "\n".join([f"import Foo{i}" for i in range(8)]),
    }
    assert not keep_algebraic_stack_p3(shell, shell["text"])

    proofsteps = {"meta": json.dumps({"llama_tokens": 10}), "text": "[GOAL] ⊢ True\n" * 5}
    assert keep_algebraic_stack_p3(proofsteps, proofsteps["text"], source_name="lean_proofsteps.jsonl.zst")
    assert keep_algebraic_stack_p3(proofsteps, proofsteps["text"], source_name="isa_proofsteps.jsonl.zst")


def test_arxiv_primary_math_and_biblio_strip() -> None:
    assert primary_is_math("math.AG physics.hep-th")
    assert not primary_is_math("cs.LG math.NA")
    assert primary_is_math(["math.NT", "cs.LO"])

    body = "Main text.\\begin{thebibliography}{99}\\bibitem{a} A.\\end{thebibliography}\nTail"
    stripped = strip_bibliography(body)
    assert "thebibliography" not in stripped
    assert "Main text." in stripped
    assert "Tail" in stripped

    body2 = "Paper.\\bibliography{refs}\n"
    assert "bibliography" not in strip_bibliography(body2)


def test_arxiv_pure_math_crosslist_and_allowlist() -> None:
    from p3math.filters import (
        ARXIV_PURE_MATH_SUBCATS,
        categories_all_math,
        keep_arxiv_pure_math_categories,
    )

    assert "PR" in ARXIV_PURE_MATH_SUBCATS
    assert categories_all_math("math.AG math.NT")
    assert not categories_all_math("math.AG cs.LG")
    assert not categories_all_math("math.AG physics.hep-th")
    assert not categories_all_math("math.AG stat.ML")

    assert keep_arxiv_pure_math_categories("math.AG math.NT")
    assert keep_arxiv_pure_math_categories("math.PR")
    assert not keep_arxiv_pure_math_categories("math.AG math.AP")  # AP not allowlisted
    assert not keep_arxiv_pure_math_categories("math.ST")
    assert not keep_arxiv_pure_math_categories("math.AG cs.CG")  # cross-list
    # primary math but secondary non-math already fails all_math
    assert not keep_arxiv_pure_math_categories(["math.CO", "cs.CG"])


def test_arxiv_english_language_field_only() -> None:
    assert is_english_arxiv_record({"meta": {"language": "en"}})
    assert is_english_arxiv_record({"meta": json.dumps({"language": "eng"})})
    assert not is_english_arxiv_record({"meta": {"language": "fr"}})
    assert not is_english_arxiv_record({"meta": {"language": "ru"}})
    assert not is_english_arxiv_record({"meta": {}})  # missing → drop


def test_arxiv_proof_gate_length_and_noise_strip() -> None:
    proof = (
        "Intro. " * 800
        + "\\begin{theorem}T\\end{theorem}\n"
        + "\\begin{proof}P\\end{proof}\n"
        + "Body. " * 800
    )
    cleaned = prepare_arxiv_text(proof)
    assert len(cleaned) >= 10_000
    assert keep_arxiv_p3_text(cleaned)
    assert count_proof_envs(cleaned) >= 2

    short = "\\begin{theorem}T\\end{theorem}\\begin{proof}P\\end{proof}\n"
    assert not keep_arxiv_p3_text(prepare_arxiv_text(short))

    no_proof = "Body text without proof structures. " * 400
    assert not keep_arxiv_p3_text(prepare_arxiv_text(no_proof))

    noisy = (
        "Keep me. " * 1200
        + "\\begin{theorem}T\\end{theorem}\\begin{proof}P\\end{proof}\n"
        + "\\begin{figure}big figure blob\\end{figure}\n"
        + "\\includegraphics[width=1]{x.png}\n"
        + "\\section{Acknowledgments} Thanks everyone forever.\n"
        + "\\section{More} still here\n"
    )
    cleaned2 = prepare_arxiv_text(noisy)
    assert "figure blob" not in cleaned2
    assert "includegraphics" not in cleaned2
    assert "Acknowledgments" not in cleaned2
    assert "still here" in cleaned2
    assert keep_arxiv_p3_text(cleaned2)
