"""Run Dolma tag + pre-mix locally for HQ reference code filtering."""

from __future__ import annotations

import copy
import gzip
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from .domain_configs import dolma_config_path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DOLMA_CONFIG_ROOT = REPO_ROOT / "configs"
TAG_EXPERIMENT = "baseline-v1"


def substitute_env(text: str, env: Mapping[str, str]) -> str:
    missing = sorted({name for name in re.findall(r"\$\{([A-Z0-9_]+)\}", text) if name not in env})
    if missing:
        raise ValueError(f"template variables are unset: {', '.join(missing)}")
    return re.sub(r"\$\{([A-Z0-9_]+)\}", lambda match: env[match.group(1)], text)


def render_taggers_config(
    *,
    domain: str,
    documents_glob: str,
    work_dir: Path,
    processes: int,
    config_root: Path = DOLMA_CONFIG_ROOT,
) -> dict[str, Any]:
    template_path = dolma_config_path(domain, "taggers", config_root)
    env = {
        "DOCUMENTS_GLOB": documents_glob,
        "TAG_EXPERIMENT": TAG_EXPERIMENT,
        "DOLMA_PROCESSES": str(processes),
        "WORK_DIR": work_dir.as_posix(),
    }
    return yaml.safe_load(substitute_env(template_path.read_text(encoding="utf-8"), env))


def render_pre_mix_config(
    *,
    domain: str,
    documents_glob: str,
    output_path: Path,
    work_dir: Path,
    processes: int,
    hate_threshold: float = 0.8,
    nsfw_threshold: float = 0.8,
    config_root: Path = DOLMA_CONFIG_ROOT,
    max_shard_bytes: int = 1_073_741_824,
) -> dict[str, Any]:
    template_path = dolma_config_path(domain, "pre-mix", config_root)
    env = {
        "WORK_DIR": work_dir.as_posix(),
        "HATE_THRESHOLD": str(hate_threshold),
        "NSFW_THRESHOLD": str(nsfw_threshold),
        "DOLMA_PROCESSES": str(processes),
    }
    config = yaml.safe_load(substitute_env(template_path.read_text(encoding="utf-8"), env))
    stream_template = config.pop("stream_template")
    stream = copy.deepcopy(stream_template)
    stream["name"] = domain
    stream["documents"] = [documents_glob]
    stream["output"]["path"] = output_path.as_posix()
    stream["output"]["max_size_in_bytes"] = max_shard_bytes
    config["streams"] = [stream]
    return config


def _require_dolma() -> str:
    executable = shutil.which("dolma")
    if executable is None:
        raise RuntimeError(
            "dolma is not installed or not on PATH; install requirements-dolma-hq.txt "
            "(pip install 'dolma[code]==1.1.2')"
        )
    return executable


def _run_dolma(config_path: Path, command: str) -> None:
    executable = _require_dolma()
    subprocess.run([executable, "--config", str(config_path), command], check=True)


def _glob_path(path: Path) -> str:
    return path.resolve().as_posix()


def _collect_mix_shards(output_prefix: Path) -> list[Path]:
    parent = output_prefix.parent
    stem = output_prefix.name
    patterns = (f"{stem}*.jsonl.gz", f"{stem}*.json.gz", "*.jsonl.gz", "*.json.gz")
    for base in (output_prefix, parent):
        if not base.is_dir():
            continue
        for pattern in patterns:
            shards = sorted(base.glob(pattern))
            if shards:
                return shards
    direct = Path(f"{output_prefix.as_posix()}.jsonl.gz")
    if direct.is_file():
        return [direct]
    direct_json = Path(f"{output_prefix.as_posix()}.json.gz")
    if direct_json.is_file():
        return [direct_json]
    raise FileNotFoundError(f"no Dolma mix output found for prefix {output_prefix}")


def _write_single_shard(source_shards: list[Path], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        for shard in source_shards:
            with gzip.open(shard, "rt", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    json.loads(line)
                    handle.write(line if line.endswith("\n") else f"{line}\n")
                    count += 1
    return count


def apply_dolma_pre_mix_domain(
    *,
    domain: str,
    shard_path: Path,
    work_root: Path,
    processes: int = 1,
    config_root: Path = DOLMA_CONFIG_ROOT,
) -> int:
    """Tag and pre-mix one local domain shard in place using Dolma HQ configs."""

    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)

    processes = max(1, int(processes))

    work_dir = work_root / domain
    work_dir.mkdir(parents=True, exist_ok=True)
    documents_glob = _glob_path(shard_path)
    tag_config_path = work_dir / "tag.yaml"
    tag_config_path.write_text(
        yaml.safe_dump(
            render_taggers_config(
                domain=domain,
                documents_glob=documents_glob,
                work_dir=work_dir / "tag",
                processes=processes,
                config_root=config_root,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    logger.info("%s: dolma tag start", domain)
    _run_dolma(tag_config_path, "tag")

    mix_output_prefix = work_dir / "mix-output" / "documents"
    mix_output_prefix.parent.mkdir(parents=True, exist_ok=True)
    mix_config_path = work_dir / "pre-mix.yaml"
    mix_config_path.write_text(
        yaml.safe_dump(
            render_pre_mix_config(
                domain=domain,
                documents_glob=documents_glob,
                output_path=mix_output_prefix,
                work_dir=work_dir / "pre-mix",
                processes=processes,
                config_root=config_root,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    logger.info("%s: dolma pre-mix start", domain)
    _run_dolma(mix_config_path, "mix")

    shards = _collect_mix_shards(mix_output_prefix)
    document_count = _write_single_shard(shards, shard_path)
    logger.info("%s: dolma pre-mix wrote %d documents", domain, document_count)
    return document_count
