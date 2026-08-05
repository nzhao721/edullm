from __future__ import annotations

from pathlib import Path

from refhq_new.exclusion import keep_row, load_exclusion_rules, skip_smoltalk_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_exclusion_rules_covers_all_sources() -> None:
    rules = load_exclusion_rules(ROOT / "exclusion_rules.yaml")
    assert set(rules["sources"]) == {
        "tulu-v2",
        "openhermes-25",
        "tulu-3",
        "hermes-3",
        "smoltalk",
        "dolci",
    }


def test_tulu3_drops_safety_if_aya() -> None:
    rules = load_exclusion_rules()
    assert not keep_row(
        "tulu-3",
        {"source": "allenai/wildguardmix", "messages": []},
        rules,
    )
    assert not keep_row(
        "tulu-3",
        {"source": "allenai/wildjailbreak", "messages": []},
        rules,
    )
    assert not keep_row(
        "tulu-3",
        {"source": "allenai/coconot", "messages": []},
        rules,
    )
    assert not keep_row(
        "tulu-3",
        {"source": "allenai/tulu-3-sft-personas-instruction-following", "messages": []},
        rules,
    )
    assert not keep_row(
        "tulu-3",
        {"source": "CohereForAI/aya_dataset", "messages": []},
        rules,
    )
    assert keep_row(
        "tulu-3",
        {"source": "ai2-adapt-dev/flan_v2_converted", "messages": []},
        rules,
    )
    assert keep_row(
        "tulu-3",
        {"source": "allenai/tulu-3-personas-math", "messages": []},
        rules,
    )


def test_smoltalk_skip_configs() -> None:
    rules = load_exclusion_rules()
    assert skip_smoltalk_config("apigen-80k", rules)
    assert skip_smoltalk_config("smol-constraints", rules)
    assert not skip_smoltalk_config("numina-cot-100k", rules)
    assert not keep_row("smoltalk", {"messages": []}, rules, smoltalk_config="apigen-80k")
    assert keep_row("smoltalk", {"messages": []}, rules, smoltalk_config="openhermes-100k")


def test_dolci_drops_safety_precise_if_tools() -> None:
    rules = load_exclusion_rules()
    assert not keep_row("dolci", {"domain": "Safety", "source_dataset": "WildGuardMix"}, rules)
    assert not keep_row(
        "dolci",
        {"domain": "Precise IF", "source_dataset": "Dolci Instruct Precise IF"},
        rules,
    )
    assert not keep_row(
        "dolci",
        {"domain": "Other", "source_dataset": "Dolci Instruct Tool Use"},
        rules,
    )
    assert not keep_row(
        "dolci",
        {"domain": "Other", "source_dataset": "Aya"},
        rules,
    )
    assert not keep_row(
        "dolci",
        {
            "domain": "Coding",
            "source_dataset": "Evol CodeAlpaca",
            "messages": [{"role": "assistant", "content": "x", "function_calls": "{[]}"}],
        },
        rules,
    )
    assert keep_row(
        "dolci",
        {
            "domain": "Math",
            "source_dataset": "Tulu 3 Persona MATH",
            "messages": [
                {"role": "user", "content": "1+1?", "function_calls": None, "functions": None}
            ],
        },
        rules,
    )


def test_openhermes_optional_language_filter() -> None:
    rules = load_exclusion_rules()
    assert keep_row("openhermes-25", {"language": None}, rules)
    assert keep_row("openhermes-25", {"language": "en"}, rules)
    assert keep_row("openhermes-25", {}, rules)
    assert not keep_row("openhermes-25", {"language": "fr"}, rules)


def test_tulu_v2_and_hermes_keep_all() -> None:
    rules = load_exclusion_rules()
    assert keep_row("tulu-v2", {"dataset": "sharegpt", "messages": []}, rules)
    assert keep_row("hermes-3", {"conversations": []}, rules)
