#!/usr/bin/env bash
set +e
echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
pidfile=/workspace/bootstrap/full-run.pid
log=/workspace/bootstrap/full-run.log
if [[ ! -f "$pidfile" ]] || ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  echo "STATUS: not_running"
  tail -n 15 "$log" 2>/dev/null
  exit 0
fi

phase=unknown
prepare_pid=""
train_pid=""
prepare_pid="$(pgrep -f prepare_annotated_corpus.py | head -n 1)"
train_pid="$(pgrep -f train_smollm2_135m_colmlm_ddp.py | head -n 1)"
if [[ -n "$prepare_pid" ]]; then phase=prepare; fi
if [[ -n "$train_pid" ]]; then phase=train; fi

run_dir="$(ls -dt /workspace/smollm2-colmlm-* 2>/dev/null | head -n 1)"
if [[ -z "$run_dir" ]]; then
  run_name="$(grep '\[train\] full run' "$log" 2>/dev/null | tail -n 1 | sed -n 's/.*full run \([^;]*\).*/\1/p')"
  [[ -n "$run_name" ]] && run_dir="/workspace/${run_name}"
fi

echo "STATUS: running phase=${phase} pid=$(cat "$pidfile") run_dir=${run_dir:-unknown}"

grep -E '^prepared |^\[prepare\]|^\[train\]|^step=' "$log" 2>/dev/null | tail -n 6
tail -n 2 "$log" 2>/dev/null

python3 - "$phase" "$run_dir" "$prepare_pid" <<'PY'
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

phase, run_dir, prepare_pid = sys.argv[1:4]
run_dir = Path(run_dir) if run_dir and run_dir != "unknown" else None
MAX_TOKENS = 20_000_000_000
EVAL_INTERVAL = 250_000_000
EVAL_MINUTES_PER_MILESTONE = 5.0
TRAIN_ETA_FALLBACK_SEC = 7 * 3600
TOTAL_SHARDS = 19


def fmt_eta(seconds: float) -> str:
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def finish_utc(seconds: float) -> str:
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "unknown"
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def remaining_eval_count(seen: int) -> int:
    return sum(1 for b in range(EVAL_INTERVAL, MAX_TOKENS + 1, EVAL_INTERVAL) if b > seen)


def next_eval_tokens(seen: int) -> int | None:
    nxt = ((seen // EVAL_INTERVAL) + 1) * EVAL_INTERVAL
    return nxt if nxt <= MAX_TOKENS else None


if phase == "prepare" and run_dir and prepare_pid:
    prepare_log = run_dir / "logs/prepare.log"
    text = prepare_log.read_text(encoding="utf-8", errors="replace") if prepare_log.is_file() else ""
    matches = re.findall(r"prepared (\d+)/(\d+)", text)
    if matches:
        done, total = map(int, matches[-1])
        try:
            etimes = int(Path(f"/proc/{prepare_pid}/stat").read_text().split()[21]) // 100
        except OSError:
            etimes = 0
        # jiffies fallback: read ps etimes
        if etimes <= 0:
            import subprocess

            out = subprocess.check_output(["ps", "-o", "etimes=", "-p", prepare_pid], text=True).strip()
            etimes = int(out or 0)
        if done > 0 and etimes > 0:
            per_shard = etimes / done
            remaining = max(total - done, 0)
            prepare_sec = remaining * per_shard
            total_sec = prepare_sec + TRAIN_ETA_FALLBACK_SEC
            print(f"ETA_prepare: {fmt_eta(prepare_sec)} ({done}/{total} shards, ~{per_shard:.0f}s/shard)")
            print(
                f"ETA_total: {fmt_eta(total_sec)} "
                f"(prepare + ~{TRAIN_ETA_FALLBACK_SEC // 3600}h train estimate, "
                f"finish ~{finish_utc(total_sec)})"
            )
        else:
            print(f"ETA_prepare: warming up ({done}/{total} shards)")
            print(
                f"ETA_total: unknown (prepare warming up; train estimate ~{TRAIN_ETA_FALLBACK_SEC // 3600}h after prepare)"
            )
    else:
        print("ETA_prepare: unknown (no shard progress yet)")
        print("ETA_total: unknown (no prepare progress yet)")

elif phase == "train" and run_dir:
    progress = run_dir / "output/progress/train.jsonl"
    metrics = None
    if progress.is_file():
        for line in progress.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    metrics = json.loads(line)
                except json.JSONDecodeError:
                    pass
    if not metrics:
        log = Path("/workspace/bootstrap/full-run.log")
        if log.is_file():
            for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
                if line.startswith("step="):
                    m = re.search(r"tokens_seen=([\d,]+)", line)
                    tps_m = re.search(r"recent_tps=([\d,]+)", line)
                    avg_m = re.search(r"avg_tps=([\d,]+)", line)
                    if m:
                        metrics = {
                            "tokens_seen": int(m.group(1).replace(",", "")),
                            "recent_tps": float(tps_m.group(1).replace(",", "")) if tps_m else 0.0,
                            "avg_tps": float(avg_m.group(1).replace(",", "")) if avg_m else 0.0,
                            "step": 0,
                        }
                        step_m = re.search(r"step=(\d+)/", line)
                        if step_m:
                            metrics["step"] = int(step_m.group(1))
                    break
    if metrics:
        seen = int(metrics.get("tokens_seen", 0))
        recent = float(metrics.get("recent_tps", 0) or 0)
        avg = float(metrics.get("avg_tps", 0) or 0)
        tps = recent if recent > 0 else avg
        remaining = max(MAX_TOKENS - seen, 0)
        pct = 100.0 * seen / MAX_TOKENS
        print(
            f"train_progress: step={metrics.get('step', '?')} "
            f"tokens={seen:,}/{MAX_TOKENS:,} ({pct:.2f}%) "
            f"recent_tps={recent:,.0f} avg_tps={avg:,.0f}"
        )
        log = Path("/workspace/bootstrap/full-run.log")
        log_lines = (
            log.read_text(encoding="utf-8", errors="replace").splitlines()
            if log.is_file()
            else []
        )
        log_tail = "\n".join(log_lines[-30:])
        in_eval = "running sharded task eval" in log_tail and not any(
            line.startswith("step=") for line in log_lines[-5:]
        )
        eval_count = remaining_eval_count(seen)
        if in_eval and eval_count > 0:
            eval_count -= 1
        if tps > 0:
            train_sec = remaining / tps
            eval_sec = eval_count * EVAL_MINUTES_PER_MILESTONE * 60.0
            total_sec = train_sec + eval_sec
            print(f"ETA_train: {fmt_eta(train_sec)} at {tps:,.0f} tok/s")
            print(
                f"ETA_total: {fmt_eta(total_sec)} "
                f"({eval_count} evals x {EVAL_MINUTES_PER_MILESTONE:.0f}m buffer, "
                f"finish ~{finish_utc(total_sec)})"
            )
            nxt = next_eval_tokens(seen)
            if nxt is not None:
                print(
                    f"ETA_next_eval: {fmt_eta((nxt - seen) / tps)} "
                    f"at {nxt:,} tokens"
                    + (" (eval running now)" if in_eval else "")
                )
        else:
            print("ETA_train: warming up (compile/autotune; no stable throughput yet)")
            print("ETA_total: unknown (waiting for stable throughput)")
    else:
        print("ETA_train: warming up (no metrics yet)")
        print("ETA_total: unknown (no metrics yet)")
else:
    print("ETA_train: n/a")
    print("ETA_total: n/a")
PY

if [[ "$phase" == train ]]; then
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
fi
