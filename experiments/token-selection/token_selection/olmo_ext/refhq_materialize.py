"""Materialize RefHQ DistCP checkpoints from S3 into local flat ``.pt`` files.

Canonical weights live on S3 as olmo-core DistCP. FrozenReference / EMA seeding
need a local state dict. Call :func:`ensure_reference_paths` before launch when
``reference.load_path`` (or early/late) is null but ``s3_uri`` / ``s3_uris`` is set.

Safe under multi-process torchrun: mkdir-based lock + reuse of existing outputs.
Does not start training. Downloads use ``aws s3 sync`` on the train host.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union

log = logging.getLogger("token_selection.refhq_materialize")

DEFAULT_REFHQ_BASE = (
    "s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints"
)
DEFAULT_STEP1315 = f"{DEFAULT_REFHQ_BASE}/step1315/"
_BAD = frozenset({"", "null", "None", "REPLACE_ME"})


def default_ref_cache() -> Path:
    env = (os.environ.get("TOKEN_SELECTION_REF_CACHE") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Shared across arms so rho-1 and rel-ema-refhq reuse step1315.
    root = Path(__file__).resolve().parents[2]  # experiments/token-selection
    return (root / ".cache" / "refhq").resolve()


def _is_bad_path(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() in _BAD


def _run(cmd: Sequence[str]) -> None:
    log.info("+ %s", " ".join(cmd))
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(list(cmd))


def _step_slug(s3_uri: str) -> str:
    m = re.search(r"step(\d+)", s3_uri.rstrip("/"))
    if m:
        return f"step{m.group(1)}"
    # Stable fallback for non-standard URIs.
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", s3_uri.strip().rstrip("/"))
    return safe[-80:] or "ref"


def export_distcp_to_pt(
    s3_uri: str,
    *,
    work_dir: Path,
    output: Path,
    skip_download: bool = False,
    force: bool = False,
) -> Path:
    """Download DistCP from ``s3_uri``, unshard, copy flat ``model.pt`` to ``output``."""
    output = Path(output)
    if output.exists() and output.stat().st_size > 0 and not force:
        log.info("reuse existing reference %s", output)
        return output.resolve()

    work = Path(work_dir)
    ckpt_dir = work / "step_ckpt"
    unshard_dir = work / "unsharded"
    work.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("aws") is None:
            raise RuntimeError(
                "aws CLI not on PATH; cannot sync RefHQ DistCP from "
                f"{s3_uri}. Export on a host with credentials, or set "
                "reference.load_path to an existing local .pt."
            )
        _run(["aws", "s3", "sync", s3_uri.rstrip("/") + "/", str(ckpt_dir)])

    model_and_optim = ckpt_dir / "model_and_optim"
    if not (model_and_optim / ".metadata").exists():
        raise FileNotFoundError(
            f"Missing distcp metadata under {model_and_optim}; "
            "expected an olmo-core model_and_optim/ checkpoint"
        )

    try:
        from olmo_core.distributed.checkpoint import unshard_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "olmo_core is required to unshard RefHQ DistCP. Install the pinned "
            "edu-llm/OLMo-core checkout on the train host."
        ) from exc

    if unshard_dir.exists():
        shutil.rmtree(unshard_dir)
    unshard_dir.mkdir(parents=True, exist_ok=True)

    result = unshard_checkpoint(
        dir=str(model_and_optim),
        target_dir=str(unshard_dir),
        optim=False,
        save_overwrite=True,
    )
    log.info("unshard_result=%s", result)

    candidates = [
        unshard_dir / "model.pt",
        unshard_dir / "model.pth",
        unshard_dir / "model_and_optim" / "model.pt",
        ckpt_dir / "model.pt",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        pts = sorted(unshard_dir.rglob("*.pt"))
        if not pts:
            raise FileNotFoundError(
                f"unshard_checkpoint finished but no model.pt under {unshard_dir}"
            )
        src = pts[0]
        log.warning("using fallback unsharded file %s", src)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    tmp.replace(output)
    meta = {
        "source_s3": s3_uri,
        "local_distcp": str(ckpt_dir),
        "unsharded_src": str(src),
        "output": str(output),
        "bytes": output.stat().st_size,
    }
    output.with_suffix(".json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    log.info("READY reference.load_path=%s", output)
    return output.resolve()


def _with_mkdir_lock(lock_dir: Path, *, timeout_s: float = 7200.0) -> Any:
    """Context manager: exclusive mkdir lock with stale reclaim."""

    class _Lock:
        def __enter__(self) -> Path:
            deadline = time.time() + timeout_s
            while True:
                try:
                    lock_dir.mkdir(parents=False)
                    return lock_dir
                except FileExistsError:
                    # Stale lock older than 3h → reclaim.
                    try:
                        age = time.time() - lock_dir.stat().st_mtime
                    except OSError:
                        age = 0.0
                    if age > 3 * 3600:
                        log.warning("reclaiming stale lock %s (age=%.0fs)", lock_dir, age)
                        shutil.rmtree(lock_dir, ignore_errors=True)
                        continue
                    if time.time() > deadline:
                        raise TimeoutError(
                            f"timed out waiting for RefHQ materialize lock {lock_dir}"
                        )
                    time.sleep(2.0)

        def __exit__(self, *exc: Any) -> None:
            shutil.rmtree(lock_dir, ignore_errors=True)

    return _Lock()


def ensure_distcp_pt(
    s3_uri: str,
    *,
    cache_dir: Optional[Path] = None,
    output_name: Optional[str] = None,
    skip_download: bool = False,
    force: bool = False,
) -> Path:
    """Idempotent DistCP→``.pt`` under the shared RefHQ cache."""
    cache = Path(cache_dir) if cache_dir is not None else default_ref_cache()
    cache.mkdir(parents=True, exist_ok=True)
    slug = _step_slug(s3_uri)
    out_name = output_name or f"refhq_{slug}_model.pt"
    output = cache / out_name
    work = cache / "work" / slug
    lock = cache / f".lock_{slug}"

    if output.exists() and output.stat().st_size > 0 and not force:
        return output.resolve()

    with _with_mkdir_lock(lock):
        if output.exists() and output.stat().st_size > 0 and not force:
            return output.resolve()
        return export_distcp_to_pt(
            s3_uri,
            work_dir=work,
            output=output,
            skip_download=skip_download,
            force=force,
        )


def ensure_late_average_pt(
    s3_uris: Sequence[str],
    *,
    steps: Sequence[int],
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Export each step then write mean state dict for learnability late ref."""
    import torch

    from token_selection.olmo_ext.train_module import (
        average_reference_state_dicts,
        load_reference_state_dict,
    )

    cache = Path(cache_dir) if cache_dir is not None else default_ref_cache()
    cache.mkdir(parents=True, exist_ok=True)
    step_tag = "_".join(str(s) for s in steps)
    late_pt = cache / f"refhq_late_avg_{step_tag}.pt"
    lock = cache / f".lock_late_avg_{step_tag}"

    if late_pt.exists() and late_pt.stat().st_size > 0 and not force:
        return late_pt.resolve()

    singles: list[Path] = []
    for uri, step in zip(s3_uris, steps):
        singles.append(
            ensure_distcp_pt(
                str(uri),
                cache_dir=cache,
                output_name=f"refhq_step{int(step)}_model.pt",
                force=force,
            )
        )

    with _with_mkdir_lock(lock):
        if late_pt.exists() and late_pt.stat().st_size > 0 and not force:
            return late_pt.resolve()
        states = [load_reference_state_dict(p) for p in singles]
        averaged = average_reference_state_dicts(states)
        tmp = late_pt.with_suffix(late_pt.suffix + ".tmp")
        torch.save(
            {
                "model": averaged,
                "averaged_checkpoints": [str(p) for p in singles],
                "steps": [int(s) for s in steps],
                "note": (
                    "Late learnability reference = mean of RefHQ steps "
                    f"{list(steps)}. Use as reference.late.load_path."
                ),
            },
            tmp,
        )
        tmp.replace(late_pt)
        meta = {
            "output": str(late_pt),
            "steps": [int(s) for s in steps],
            "sources": [str(p) for p in singles],
            "n_tensors": len(averaged),
            "bytes": late_pt.stat().st_size,
        }
        late_pt.with_suffix(".json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        return late_pt.resolve()


def _declared_local_path(path: Any) -> bool:
    """True when load_path is a non-empty non-s3 declaration (file need not exist yet)."""
    if _is_bad_path(path):
        return False
    return not str(path).strip().startswith("s3://")


def _local_path_ok(path: Any) -> bool:
    if not _declared_local_path(path):
        return False
    p = Path(str(path).strip())
    return p.exists() and (p.is_file() or p.is_dir())


def ensure_reference_paths(
    cfg: MutableMapping[str, Any],
    *,
    method: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> Dict[str, str]:
    """Fill null ``reference*.load_path`` fields from YAML ``s3_uri`` / ``s3_uris``.

    Mutates ``cfg`` in place. Returns a small summary of paths that were set.
    No-op when local paths already exist. Raises if a required ref cannot be
    resolved (no local path and no S3 provenance).
    """
    cache = Path(cache_dir) if cache_dir is not None else default_ref_cache()
    resolved = method
    if resolved is None:
        methods = cfg.get("methods") or []
        if len(methods) == 1:
            resolved = str(methods[0])

    out: Dict[str, str] = {}
    ref = cfg.setdefault("reference", {})
    if not isinstance(ref, MutableMapping):
        raise TypeError("cfg['reference'] must be a mapping")

    needs_single = resolved == "rho_excess"
    if resolved == "rel_ema":
        ema = cfg.get("ema") or {}
        seed = str(ema.get("seed_mode") or cfg.get("ema_seed_mode") or "zero").lower()
        needs_single = seed == "refhq"

    if needs_single:
        if _local_path_ok(ref.get("load_path")):
            out["reference.load_path"] = str(Path(str(ref["load_path"])).resolve())
        else:
            s3_uri = str(ref.get("s3_uri") or DEFAULT_STEP1315).strip()
            if not s3_uri.startswith("s3://"):
                raise ValueError(
                    f"{resolved} needs reference.load_path or reference.s3_uri "
                    f"(got s3_uri={s3_uri!r})"
                )
            path = ensure_distcp_pt(s3_uri, cache_dir=cache, force=force)
            ref["load_path"] = str(path)
            out["reference.load_path"] = str(path)

    if resolved == "learnability":
        early = ref.setdefault("early", {})
        late = ref.setdefault("late", {})
        if not isinstance(early, MutableMapping) or not isinstance(late, MutableMapping):
            raise TypeError("reference.early / reference.late must be mappings")

        if _local_path_ok(early.get("load_path")):
            out["reference.early.load_path"] = str(
                Path(str(early["load_path"])).resolve()
            )
        else:
            early_uri = str(
                early.get("s3_uri") or f"{DEFAULT_REFHQ_BASE}/step250/"
            ).strip()
            path = ensure_distcp_pt(
                early_uri,
                cache_dir=cache,
                output_name="refhq_step250_early.pt",
                force=force,
            )
            early["load_path"] = str(path)
            out["reference.early.load_path"] = str(path)

        if _local_path_ok(late.get("load_path")):
            out["reference.late.load_path"] = str(Path(str(late["load_path"])).resolve())
        else:
            steps = late.get("steps") or [1000, 1125, 1315]
            steps_i = [int(s) for s in steps]
            uris = late.get("s3_uris")
            if not uris:
                uris = [f"{DEFAULT_REFHQ_BASE}/step{s}/" for s in steps_i]
            if len(list(uris)) != len(steps_i):
                raise ValueError(
                    "reference.late.s3_uris length must match reference.late.steps"
                )
            path = ensure_late_average_pt(
                [str(u) for u in uris],
                steps=steps_i,
                cache_dir=cache,
                force=force,
            )
            late["load_path"] = str(path)
            out["reference.late.load_path"] = str(path)

    return out


def reference_source_ok(cfg: Mapping[str, Any], *, method: Optional[str] = None) -> bool:
    """True if local load_path exists or S3 provenance is present for ``method``."""
    resolved = method
    if resolved is None:
        methods = cfg.get("methods") or []
        if len(methods) == 1:
            resolved = str(methods[0])
    ref = cfg.get("reference") or {}

    def _single_ok(block: Mapping[str, Any]) -> bool:
        if _declared_local_path(block.get("load_path")):
            return True
        uri = str(block.get("s3_uri") or "").strip()
        return uri.startswith("s3://")

    if resolved == "rho_excess":
        return _single_ok(ref)
    if resolved == "rel_ema":
        ema = cfg.get("ema") or {}
        seed = str(ema.get("seed_mode") or cfg.get("ema_seed_mode") or "zero").lower()
        if seed != "refhq":
            return True
        return _single_ok(ref)
    if resolved == "learnability":
        early = ref.get("early") or {}
        late = ref.get("late") or {}
        early_ok = _declared_local_path(early.get("load_path")) or str(
            early.get("s3_uri") or ""
        ).startswith("s3://")
        late_ok = (
            _declared_local_path(late.get("load_path"))
            or bool(late.get("s3_uris"))
            or bool(late.get("steps"))
        )
        return early_ok and late_ok
    return True
