# Skill-DAG experiments

Domain-mixture levers under a shared Mixing Laws Dataset / OLMoHQ × OLMo-2 370M one-epoch contract.

| Experiment | Question | Outcome |
|------------|----------|---------|
| MixLaw | Can a mixing law from short runs beat fixed mixtures? | **Yes** — fitted mixtures beat the baseline (\(p < 10^{-4}\)) |
| Skill-It | Can mid-run Skill-It reweighting beat the best fixed mixture? | **No** — both arms beat the Olmo-mix-1124 control (probe \(p < 10^{-4}\), derivative \(p = 0.004\)) but neither beat the LightGBM static mixture they start from |

Shared evaluation: macro task-loss CE bits-per-byte over 20 OLMES-style labels;
power-law **alpha-free** residual-bootstrap CIs on fitted finals (steps ≥ 1000),
one independent resampling stream per arm, reproducible offline via
[`mixlaw/fit_and_bootstrap_370m.py`](mixlaw/fit_and_bootstrap_370m.py).
A100-hours: MixLaw 188.74, Skill-It 101.06 (two arms; the earlier 148.2 counted a
third arm that was never run). FLOPs: \(2.63\times10^{19}\) per 370M
arm (W&B); MixLaw 60M pilot grid \(\approx 4.17\times10^{18}\); Skill-It 60M
probes \(\approx 1.22\times10^{18}\).

**Seed noise floor.** Two Olmo-mix-1124 runs differing only in dataloader seed
land 0.0044 bpb apart (\(p = 0.34\)); treat differences below ~0.004 bpb as
indistinguishable.
