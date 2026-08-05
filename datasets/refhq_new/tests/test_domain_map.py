from __future__ import annotations

import pytest

from refhq_new.domain_map import DOMAINS, SOURCES, map_domain


def test_sources_and_domains_match_design() -> None:
    assert SOURCES == (
        "tulu-v2",
        "openhermes-25",
        "tulu-3",
        "hermes-3",
        "smoltalk",
        "dolci",
    )
    assert DOMAINS == ("general", "math", "code", "science", "chat")


def test_smoltalk_config_domain() -> None:
    assert map_domain("smoltalk", smoltalk_config="numina-cot-100k") == "math"
    assert map_domain("smoltalk", smoltalk_config="self-oss-instruct") == "code"
    assert map_domain("smoltalk", smoltalk_config="everyday-conversations") == "chat"
    assert map_domain("smoltalk", smoltalk_config="openhermes-100k") == "general"
    assert map_domain("smoltalk", {"source": "metamathqa-50k"}) == "math"


def test_tulu3_source_domain() -> None:
    assert map_domain("tulu-3", {"source": "allenai/tulu-3-personas-math"}) == "math"
    assert map_domain("tulu-3", {"source": "allenai/tulu-3-sft-personas-code"}) == "code"
    assert map_domain("tulu-3", {"source": "allenai/SciRIFF"}) == "science"
    assert map_domain("tulu-3", {"source": "allenai/WildChat-1M"}) == "chat"
    assert map_domain("tulu-3", {"source": "ai2-adapt-dev/flan_v2_converted"}) == "general"


def test_tulu_v2_dataset_domain() -> None:
    assert map_domain("tulu-v2", {"dataset": "code_alpaca"}) == "code"
    assert map_domain("tulu-v2", {"dataset": "sharegpt"}) == "chat"
    assert map_domain("tulu-v2", {"dataset": "cot"}) == "general"
    assert map_domain("tulu-v2", {"dataset": "unknown_slice"}) == "general"


def test_dolci_domain_column() -> None:
    assert map_domain("dolci", {"domain": "Math"}) == "math"
    assert map_domain("dolci", {"domain": "Science"}) == "science"
    assert map_domain("dolci", {"domain": "Coding"}) == "code"
    assert map_domain("dolci", {"domain": "Other"}) == "general"
    assert (
        map_domain(
            "dolci",
            {"domain": "", "source_dataset": "Dolci Instruct Python Algorithms"},
        )
        == "code"
    )


def test_openhermes_and_hermes_defaults() -> None:
    assert map_domain("openhermes-25", {"category": "coding"}) == "code"
    assert map_domain("openhermes-25", {}) == "general"
    assert map_domain("hermes-3", {}) == "general"


def test_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        map_domain("not-a-source", {})
