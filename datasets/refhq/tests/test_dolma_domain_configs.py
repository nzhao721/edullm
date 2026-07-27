from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "datasets"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from refhq.domain_configs import dolma_config_path
from refhq.process import render_pre_mix_config, render_taggers_config


def test_code_hq_domain_uses_hq_configs() -> None:
    config_root = ROOT_DIR / "configs"
    assert dolma_config_path("code-hq", "taggers", config_root) == config_root / "taggers-code-hq.yaml"
    assert (
        dolma_config_path("code-hq", "pre-mix", config_root)
        == config_root / "pre-mix-code-hq.template.yaml"
    )
    assert (
        dolma_config_path("code-hq", "mix", config_root)
        == config_root / "mix-code-hq-domain.template.yaml"
    )


def test_code_hq_taggers_are_code_only() -> None:
    text = (ROOT_DIR / "configs" / "taggers-code-hq.yaml").read_text(encoding="utf-8")
    for tagger in (
        "code_copyright_comments_v1",
        "code_redpajama_taggers_v1",
        "code_secrets_v1",
    ):
        assert tagger in text
    for forbidden in ("gopher_v1", "c4_v2", "ft_lang_id"):
        assert forbidden not in text


def test_code_hq_pre_mix_strips_copyright_without_lang_filter() -> None:
    config = render_pre_mix_config(
        domain="code-hq",
        documents_glob="/tmp/sample/code/documents.jsonl.gz",
        output_path=Path("/tmp/sample/work/mix-output/documents"),
        work_dir=Path("/tmp/sample/work/pre-mix"),
        processes=1,
        config_root=ROOT_DIR / "configs",
    )
    stream = config["streams"][0]
    assert "include" not in stream.get("filter", {}) or stream["filter"].get("include") in (None, [])
    assert stream["attributes"] == ["baseline-v1"]
    assert "baseline_v1__code_secrets_v1__doc" in str(stream["filter"]["exclude"])
    replacements = stream["span_replacement"]
    assert any(
        item["span"].endswith("baseline_v1__code_copyright_comments_v1__copyright_notice")
        for item in replacements
    )
    assert any(
        item["span"].endswith("baseline_v1__code_copyright_comments_v1__comment_block")
        for item in replacements
    )


def test_code_hq_taggers_config_lists_code_taggers() -> None:
    import yaml

    config = render_taggers_config(
        domain="code-hq",
        documents_glob="/tmp/sample/code/documents.jsonl.gz",
        work_dir=Path("/tmp/sample/work/tag"),
        processes=1,
        config_root=ROOT_DIR / "configs",
    )
    assert config["experiment"] == "baseline-v1"
    assert config["documents"] == ["/tmp/sample/code/documents.jsonl.gz"]
    for tagger in (
        "code_copyright_comments_v1",
        "code_redpajama_taggers_v1",
        "code_secrets_v1",
    ):
        assert tagger in config["taggers"]
    yaml.safe_dump(config)
