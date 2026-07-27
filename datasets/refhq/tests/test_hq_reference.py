"""Tests for HQ reference corpus planning and acceptance helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOLMA_HQ_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DOLMA_HQ_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "farmshare"))
sys.path.insert(0, str(DOLMA_HQ_ROOT / "scripts"))

from hq_reference_sources import (  # noqa: E402
    HQ_BUDGETS,
    HQ_DOMAINS,
    HQ_SOURCES,
    UNFILTERED_POOL_TOKENS,
    within_budget,
)
from smoke_code_copyright_strip import check_python_docs  # noqa: E402


def test_hq_budgets_sum_to_four_billion() -> None:
    assert abs(sum(HQ_BUDGETS.values()) - 4.0e9) < 1.0
    assert HQ_BUDGETS["dclm"] == 1.0e9
    for domain in HQ_DOMAINS:
        if domain != "dclm":
            assert HQ_BUDGETS[domain] == 0.5e9


def test_unfiltered_pools_and_sources_cover_all_domains() -> None:
    assert set(UNFILTERED_POOL_TOKENS) == set(HQ_DOMAINS)
    assert set(HQ_SOURCES) == set(HQ_DOMAINS)
    assert HQ_SOURCES["dclm"]["kind"] == "datadecide_npy"
    assert "ft7percentile_fw2" in HQ_SOURCES["dclm"]["prefix"]
    assert HQ_SOURCES["dclm"]["source_tokenizer_id"] == "allenai/gpt-neox-olmo-dolma-v1_5"
    assert HQ_SOURCES["dclm"]["max_files"] == 24
    assert HQ_SOURCES["starcoder"]["repo_id"] == "bigcode/starcoderdata"
    assert HQ_SOURCES["open-web-math"]["filter"] == "openwebmath-hq"
    assert UNFILTERED_POOL_TOKENS["dclm"] == 100e9


def test_within_budget_tolerance() -> None:
    assert within_budget(1.0e9, 1.0e9)
    assert within_budget(1.01e9, 1.0e9)
    assert within_budget(0.98e9, 1.0e9)
    assert not within_budget(0.95e9, 1.0e9)
    assert not within_budget(1.05e9, 1.0e9)


def test_plan_hq_reference_writes_manifest(tmp_path: Path) -> None:
    from plan_hq_reference import main as plan_main

    scratch = tmp_path / "hq-reference-v1"
    sys.argv = ["plan_hq_reference.py", "--scratch-root", str(scratch), "--seed", "7"]
    assert plan_main() == 0
    plan = json.loads((scratch / "manifests" / "plan.json").read_text(encoding="utf-8"))
    assert set(plan["domains"]) == set(HQ_DOMAINS)
    assert plan["seed"] == 7
    assert plan["domains"]["dclm"]["budget_tokens"] == 1.0e9
    assert plan["domains"]["starcoder"]["source"]["filter"] == "dolma-code-hq"


def test_finalize_accepts_in_budget_stats(tmp_path: Path) -> None:
    from finalize_hq_reference_upload import main as fin_main
    from plan_hq_reference import main as plan_main

    scratch = tmp_path / "hq-reference-v1"
    sys.argv = ["plan_hq_reference.py", "--scratch-root", str(scratch), "--seed", "1"]
    assert plan_main() == 0
    plan_path = scratch / "manifests" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for domain, domain_plan in plan["domains"].items():
        stats = {
            "domain": domain,
            "realized_tokens": domain_plan["budget_tokens"],
            "doc_count": 10,
        }
        Path(domain_plan["paths"]["stats"]).write_text(json.dumps(stats) + "\n", encoding="utf-8")
        Path(domain_plan["paths"]["out"]).mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "finalize_hq_reference_upload.py",
        "--plan",
        str(plan_path),
        "--skip-upload",
    ]
    assert fin_main() == 0


def test_finalize_rejects_out_of_budget_domain(tmp_path: Path) -> None:
    from finalize_hq_reference_upload import main as fin_main
    from plan_hq_reference import main as plan_main

    scratch = tmp_path / "hq-reference-v1"
    sys.argv = ["plan_hq_reference.py", "--scratch-root", str(scratch), "--seed", "1"]
    assert plan_main() == 0
    plan_path = scratch / "manifests" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for domain, domain_plan in plan["domains"].items():
        realized = domain_plan["budget_tokens"] * (0.5 if domain == "dclm" else 1.0)
        stats = {"domain": domain, "realized_tokens": realized, "doc_count": 1}
        Path(domain_plan["paths"]["stats"]).write_text(json.dumps(stats) + "\n", encoding="utf-8")
        Path(domain_plan["paths"]["out"]).mkdir(parents=True, exist_ok=True)

    sys.argv = ["finalize_hq_reference_upload.py", "--plan", str(plan_path), "--skip-upload"]
    assert fin_main() == 1


def test_copyright_strip_fixture() -> None:
    fixture = '''# Copyright 2024
"""Module doc."""

def add(a, b):
    return a + b
'''
    result = check_python_docs([fixture], limit=1)
    assert result["checked"] == 1
    assert result["ok"] == 1


def test_fill_until_budget_stops_at_tokens(monkeypatch) -> None:
    from build_hq_reference_domain import fill_until_budget

    monkeypatch.setattr(
        "build_hq_reference_domain.worker_init",
        lambda _tokenizer: None,
    )
    monkeypatch.setattr(
        "build_hq_reference_domain._count_tokens",
        lambda texts: [len(text.split()) for text in texts],
    )

    docs = ({"id": str(i), "text": "one two three four five"} for i in range(100))
    selected, stats = fill_until_budget(docs, budget=50, seed=1, keep=None)
    assert stats["realized_tokens"] >= 50
    assert stats["doc_count"] == len(selected)


def test_fill_until_budget_applies_keep_predicate(monkeypatch) -> None:
    from build_hq_reference_domain import fill_until_budget

    monkeypatch.setattr(
        "build_hq_reference_domain.worker_init",
        lambda _tokenizer: None,
    )
    monkeypatch.setattr(
        "build_hq_reference_domain._count_tokens",
        lambda texts: [10 for _ in texts],
    )

    docs = ({"id": str(i), "text": f"doc-{i}"} for i in range(20))
    selected, stats = fill_until_budget(
        docs,
        budget=1000,
        seed=2,
        keep=lambda doc: int(doc["id"]) % 2 == 0,
    )
    assert stats["rejected_docs"] > 0
    assert all(int(doc["id"]) % 2 == 0 for doc in selected)
