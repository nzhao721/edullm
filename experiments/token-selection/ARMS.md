# Token-selection experiments

Shared package: [`token_selection/`](token_selection/) (`PYTHONPATH=experiments/token-selection`).
Reference architecture source of truth: [`reference/`](reference/) (RefHQ CE, leave as-is).

| Arm | Directory | Selection | Status |
|-----|-----------|-----------|--------|
| Control (full CE) | [`control/`](control/) | none | Standalone trainer; retrain under ladder contract |
| BLADE | [`blade/`](blade/) | top-60% `L_ref − L_proxy` | Syncs 500/875/1250/1625/2000; K=75, τ=375, γ=0.6, λ=1.0; blade_start=500; ckpts save proxy+ref |
| RHO-1 | [`rho-1/`](rho-1/) | top-60% `L_curr − L_ref` | Frozen RefHQ step1315; `t0=0`; YAML spine |
| REL exp-α | [`rel-ema-exp/`](rel-ema-exp/) | top-60% `L_hist − L_curr` | Bias-corrected EMA from zero; `α(t)=1−e^(−t/300)`; `t0=0` |
| REL RefHQ-init | [`rel-ema-refhq/`](rel-ema-refhq/) | top-60% REL | Seed EMA from RefHQ; constant α=0.9985; `t0=0` |
| Middle PPL (token) | [`middle-ppl-token/`](middle-ppl-token/) | middle-60% by `L_curr` | Online scorer; `t0=0` |
| Middle PPL (doc) | [`middle-ppl-doc/`](middle-ppl-doc/) | middle-60% docs by RefHQ PPL | Offline filter + CE; needs LM labels |
| Attention | [`attention/`](attention/) | top-60% attn-received | `attention_topk`; FA-safe hook+recompute; `t0=0` |
| Learnability (token) | [`learnability-token/`](learnability-token/) | top-60% `L_early − L_late` | Dual frozen RefHQ (250 vs avg 1000/1125/1315); `t0=0` |
| Learnability (doc) | [`learnability-doc/`](learnability-doc/) | top-60% docs by early→late | Offline filter + CE; needs LM labels |
| Reference (RefHQ) | [`reference/`](reference/) | — | Frozen HQ 5.5B; arch / refs only |

### Shared contracts

- **Architecture:** `TransformerConfig.olmo2_370M` (full attn, no SWA), GBS `4_194_304`, seq 2048, peak LR `4e-4`, warmup 24, `alpha_f=0.1` (match RefHQ).
- **Permanent checkpoints:** step 0, every 125 (skip last grid point if within 125 of final), plus final — e.g. `{0,125,…,2250,2384}` (omit 2375). Helper: `token_selection.olmo_ext.checkpoint_ladder`.
- **Online selection warmup:** `t0_steps=0` / `t0_frac=0` for all online scorers. BLADE keeps its separate 500-step proxy warmup.
- **Task loss:** full 20-label OLMo-ladder `task_loss_bpb` (RC 5-shot) via `task_loss_hook` / `TaskLossEvalCallback` on each permanent save. Evaluator: `scripts/farmshare/task_loss/eval_task_loss_olmo_core.py`.
- **Hardware:** discover world size from `torchrun` / env; no hardcoded GPU count, device pins, or host paths as required defaults.

See each arm’s `README.md` for launch commands. Do not submit AWS workloads unless explicitly authorized.

---

Full experiment plan: [README.md](README.md).