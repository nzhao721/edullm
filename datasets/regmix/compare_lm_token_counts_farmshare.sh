#!/usr/bin/env bash
set -Eeuo pipefail
SOCKET="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
RUN=/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810
ssh -S "${SOCKET}" -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash <<REMOTE
set -Eeuo pipefail
RUN=${RUN}
python3 <<'PY'
import gzip, json
from pathlib import Path

run = Path("${RUN}")
trim = json.loads((run / "plan/trim_results.json").read_text())
trim_content = sum(int(x.get("tokens_after") or 0) for x in trim)
trim_with_eos = sum(int(x.get("tokens_with_eos") or 0) for x in trim)
trim_docs = sum(int(x.get("docs_after") or 0) for x in trim)

ready = json.loads((run / "lm_labels/labels/READY").read_text())
scored = int(ready["scored_tokens"])
n_docs = int(ready["n_docs"])

n_tokens = 0
n_loss = 0
for path in (run / "lm_labels/labels/docs").rglob("*.done"):
    row = json.loads(path.read_text())
    n_tokens += int(row.get("n_tokens") or 0)
    n_loss += int(row.get("scored_tokens") or 0)

manifest = [json.loads(l) for l in (run / "lm_labels/lm_work_manifest.jsonl").read_text().splitlines() if l.strip()]
est = sum(int(x.get("est_tokens") or 0) for x in manifest)

tok_meta_total = 0
tok_root = run / "tokenized"
for meta in tok_root.glob("*/*.json"):
    if meta.name.endswith(".json") and not meta.name.endswith("-trimmed.json"):
        try:
            m = json.loads(meta.read_text())
            tok_meta_total += int(m.get("tokens_content") or m.get("tokens_with_eos") or 0)
        except Exception:
            pass

print(json.dumps({
    "trim_tokens_after_content": trim_content,
    "trim_tokens_with_eos": trim_with_eos,
    "trim_docs_after": trim_docs,
    "ready_scored_tokens": scored,
    "ready_n_docs": n_docs,
    "sum_done_n_tokens": n_tokens,
    "sum_done_scored_tokens": n_loss,
    "manifest_est_tokens": est,
    "tokenized_meta_sum": tok_meta_total,
    "scored_plus_first_token_per_doc": scored + n_docs,
    "gap_vs_trim_content": trim_content - (scored + n_docs),
    "gap_vs_10b_published": 10_000_058_051 - scored,
}, indent=2))
PY
REMOTE
