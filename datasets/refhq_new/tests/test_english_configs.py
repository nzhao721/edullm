from __future__ import annotations

from pathlib import Path

import yaml

from refhq_new.domain_configs import dolma_config_path
from refhq_new.process import render_pre_mix_config, render_taggers_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def test_english_domain_uses_english_configs() -> None:
    assert dolma_config_path("english", "taggers", CONFIG_ROOT) == CONFIG_ROOT / "taggers-english.yaml"
    assert (
        dolma_config_path("english", "pre-mix", CONFIG_ROOT)
        == CONFIG_ROOT / "pre-mix-english.template.yaml"
    )
    assert dolma_config_path("en", "taggers", CONFIG_ROOT) == CONFIG_ROOT / "taggers-english.yaml"


def test_english_taggers_are_lang_id_only() -> None:
    text = (CONFIG_ROOT / "taggers-english.yaml").read_text(encoding="utf-8")
    assert "ft_lang_id_en_paragraph_with_doc_score_v2" in text
    assert "ft_lang_id_en_doc_v2" in text
    for forbidden in ("gopher_v1", "c4_v2", "code_secrets", "hate", "nsfw"):
        assert forbidden not in text


def test_render_taggers_config_lists_english_taggers() -> None:
    config = render_taggers_config(
        domain="english",
        documents_glob="/tmp/sample/documents.jsonl.gz",
        work_dir=Path("/tmp/sample/work/tag"),
        processes=2,
        config_root=CONFIG_ROOT,
    )
    assert config["experiment"] == "baseline-v1"
    assert config["documents"] == ["/tmp/sample/documents.jsonl.gz"]
    assert config["processes"] == 2
    assert "ft_lang_id_en_paragraph_with_doc_score_v2" in config["taggers"]
    assert "ft_lang_id_en_doc_v2" in config["taggers"]
    yaml.safe_dump(config)


def test_render_pre_mix_includes_english_score_threshold() -> None:
    config = render_pre_mix_config(
        domain="english",
        documents_glob="/tmp/sample/documents.jsonl.gz",
        output_path=Path("/tmp/sample/work/mix-output/documents"),
        work_dir=Path("/tmp/sample/work/pre-mix"),
        processes=1,
        english_score_threshold=0.5,
        config_root=CONFIG_ROOT,
    )
    stream = config["streams"][0]
    assert stream["name"] == "english"
    assert stream["attributes"] == ["baseline-v1"]
    assert stream["documents"] == ["/tmp/sample/documents.jsonl.gz"]
    include = stream["filter"]["include"]
    assert len(include) == 1
    assert "baseline_v1__ft_lang_id_en_paragraph_with_doc_score_v2__doc_en" in include[0]
    assert ">= 0.5" in include[0]
    assert "exclude" not in stream.get("filter", {}) or not stream["filter"].get("exclude")
    for forbidden in ("gopher", "c4", "hate", "nsfw", "code_secrets"):
        assert forbidden not in str(stream)
