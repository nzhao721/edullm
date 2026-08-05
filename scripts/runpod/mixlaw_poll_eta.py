#!/usr/bin/env python3
"""Compute MixLaw mix01 training ETA from log lines or explicit step/tok/s."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# MixLaw 370M production run constants (all arms).
TOTAL_STEPS = 2384
TOKENS_PER_STEP = 4_194_304
SAVE_INTERVAL = 125
GPU_COUNT = 8
# Conservative pause per remaining ladder milestone (eval + checkpoint + wandb).
EVAL_MINUTES_PER_MILESTONE = 12.0

# Legacy smoke / custom launcher format.
STEP_RE = re.compile(
    r"step=(\d+)/(\d+)\s+.*?\btok/s=(\d+)\s+\(avg=(\d+)\)",
)
STEP_TS_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*?step=(\d+)/(\d+)",
)

# OLMo-core console_logger format (launch.sh / mixlaw-train.log).
OLMO_STEP_RE = re.compile(r"\[step=(\d+)/(\d+)(?:,[^\]]*)?\]")
OLMO_STEP_TS_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*?\[step=(\d+)/(\d+)",
)
OLMO_DEVICE_TPS_RE = re.compile(r"throughput/device/TPS=([\d,]+)")
OLMO_DEVICE_AVG_TPS_RE = re.compile(
    r"throughput/device/TPS \(actual avg\)=([\d,]+)"
)
OLMO_LOG_ETA_RE = re.compile(r"\[step=\d+/\d+,[^\]]*eta=([^\],]+)")


def _import_ladder():
    repo = Path(__file__).resolve().parents[2]
    ladder_path = (
        repo
        / "experiments"
        / "token-selection"
        / "token_selection"
        / "olmo_ext"
        / "checkpoint_ladder.py"
    )
    if not ladder_path.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("checkpoint_ladder", ladder_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.permanent_checkpoint_steps(TOTAL_STEPS, SAVE_INTERVAL)


def remaining_milestones(current_step: int) -> list[int]:
    ladder = _import_ladder()
    if ladder is None:
        # Fallback: grid + final, omit 2250 for 2384-step budget.
        ladder = list(range(0, TOTAL_STEPS + 1, SAVE_INTERVAL))
        if 2250 in ladder and TOTAL_STEPS - 2250 < SAVE_INTERVAL:
            ladder.remove(2250)
        if TOTAL_STEPS not in ladder:
            ladder.append(TOTAL_STEPS)
        ladder = sorted(set(ladder))
    return [s for s in ladder if s > current_step]


def sec_per_step(tok_s: float) -> float:
    if tok_s <= 0:
        return 0.0
    return TOKENS_PER_STEP / tok_s


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=int(seconds))
    hours, rem = divmod(int(td.total_seconds()), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def estimate(
    current_step: int,
    *,
    tok_s: float,
    avg_tok_s: float | None = None,
    log_samples: list[tuple[datetime, int]] | None = None,
) -> dict:
    steps_left = max(0, TOTAL_STEPS - current_step)
    milestones = remaining_milestones(current_step)

    rate_tok_s = tok_s
    measured_sec_per_step: float | None = None
    if log_samples and len(log_samples) >= 2:
        (t0, s0), (t1, s1) = log_samples[-2], log_samples[-1]
        dstep = s1 - s0
        dsec = (t1 - t0).total_seconds()
        if dstep > 0 and dsec > 0:
            measured_sec_per_step = dsec / dstep
    if measured_sec_per_step is None and avg_tok_s and avg_tok_s > 0:
        measured_sec_per_step = sec_per_step(avg_tok_s)
    if measured_sec_per_step is None and rate_tok_s > 0:
        measured_sec_per_step = sec_per_step(rate_tok_s)

    train_sec = steps_left * (measured_sec_per_step or 0.0)
    eval_sec = len(milestones) * EVAL_MINUTES_PER_MILESTONE * 60.0
    total_sec = train_sec + eval_sec

    next_milestone = milestones[0] if milestones else TOTAL_STEPS
    steps_to_milestone = max(0, next_milestone - current_step)
    milestone_train_sec = steps_to_milestone * (measured_sec_per_step or 0.0)

    now = datetime.now(timezone.utc)
    finish_at = now + timedelta(seconds=total_sec)
    milestone_at = now + timedelta(seconds=milestone_train_sec)

    pct = 100.0 * current_step / TOTAL_STEPS if TOTAL_STEPS else 0.0

    return {
        "current_step": current_step,
        "total_steps": TOTAL_STEPS,
        "steps_remaining": steps_left,
        "progress_pct": round(pct, 1),
        "tok_s": int(tok_s),
        "avg_tok_s": int(avg_tok_s) if avg_tok_s else None,
        "sec_per_step": round(measured_sec_per_step or 0.0, 2),
        "milestones_remaining": milestones,
        "eta_train_only": format_duration(train_sec),
        "eta_train_only_sec": int(train_sec),
        "eta_eval_buffer": format_duration(eval_sec),
        "eta_total": format_duration(total_sec),
        "eta_total_sec": int(total_sec),
        "eta_finish_utc": finish_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "next_milestone_step": next_milestone,
        "eta_next_milestone": format_duration(milestone_train_sec),
        "eta_next_milestone_utc": milestone_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _parse_tps(value: str) -> float:
    return float(value.replace(",", ""))


def parse_logs(text: str) -> dict | None:
    samples: list[tuple[datetime, int]] = []
    last_match: re.Match[str] | None = None
    olmo_step: int | None = None
    olmo_total: int | None = None
    device_tps: float | None = None
    device_avg_tps: float | None = None

    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            last_match = m
        ts_m = STEP_TS_RE.search(line)
        if ts_m:
            ts = datetime.strptime(ts_m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(
                tzinfo=timezone.utc
            )
            samples.append((ts, int(ts_m.group(2))))

        olmo_m = OLMO_STEP_RE.search(line)
        if olmo_m:
            olmo_step = int(olmo_m.group(1))
            olmo_total = int(olmo_m.group(2))
        olmo_ts_m = OLMO_STEP_TS_RE.search(line)
        if olmo_ts_m:
            ts = datetime.strptime(olmo_ts_m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(
                tzinfo=timezone.utc
            )
            samples.append((ts, int(olmo_ts_m.group(2)))
        tps_m = OLMO_DEVICE_TPS_RE.search(line)
        if tps_m:
            device_tps = _parse_tps(tps_m.group(1))
        avg_m = OLMO_DEVICE_AVG_TPS_RE.search(line)
        if avg_m:
            device_avg_tps = _parse_tps(avg_m.group(1))

    if last_match is not None:
        step = int(last_match.group(1))
        tok_s = float(last_match.group(3))
        avg_tok_s = float(last_match.group(4))
        return estimate(
            step,
            tok_s=tok_s,
            avg_tok_s=avg_tok_s,
            log_samples=samples,
        )

    if olmo_step is None:
        return None

    aggregate_tok_s = 0.0
    aggregate_avg_tok_s: float | None = None
    if device_tps is not None and device_tps > 0:
        aggregate_tok_s = device_tps * GPU_COUNT
    if device_avg_tps is not None and device_avg_tps > 0:
        aggregate_avg_tok_s = device_avg_tps * GPU_COUNT
    if aggregate_tok_s <= 0 and samples:
        # Step-only lines: estimate from recent step timestamps.
        if len(samples) >= 2:
            (t0, s0), (t1, s1) = samples[-2], samples[-1]
            dstep = s1 - s0
            dsec = (t1 - t0).total_seconds()
            if dstep > 0 and dsec > 0:
                aggregate_tok_s = (dstep * TOKENS_PER_STEP) / dsec

    if aggregate_tok_s <= 0:
        return None

    return estimate(
        olmo_step,
        tok_s=aggregate_tok_s,
        avg_tok_s=aggregate_avg_tok_s,
        log_samples=samples,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", type=int, help="Current training step")
    ap.add_argument("--tok-s", type=float, help="Recent tok/s from log line")
    ap.add_argument("--avg-tok-s", type=float, help="Average tok/s from log line")
    ap.add_argument(
        "--log-file",
        type=Path,
        help="Parse latest step= line from a log file",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args()

    result: dict | None = None
    if args.log_file:
        result = parse_logs(args.log_file.read_text(encoding="utf-8", errors="replace"))
    elif args.step is not None and args.tok_s is not None:
        result = estimate(
            args.step,
            tok_s=args.tok_s,
            avg_tok_s=args.avg_tok_s,
        )
    else:
        result = parse_logs(sys.stdin.read())

    if result is None:
        print("no step= log line found", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(
        f"step {result['current_step']}/{result['total_steps']} "
        f"({result['progress_pct']}%) | "
        f"tok/s={result['tok_s']} sec/step~{result['sec_per_step']}"
    )
    print(
        f"ETA train-only: {result['eta_train_only']} | "
        f"eval buffer ({len(result['milestones_remaining'])} milestones): "
        f"{result['eta_eval_buffer']}"
    )
    print(
        f"ETA total: {result['eta_total']} (finish ~{result['eta_finish_utc']}) | "
        f"next milestone step {result['next_milestone_step']} in "
        f"{result['eta_next_milestone']} (~{result['eta_next_milestone_utc']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
