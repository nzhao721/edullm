# Skill-DAG experiments

Domain-mixture levers under a shared Mixing Laws Dataset / OLMoHQ × OLMo-2 370M one-epoch contract.

| Experiment | Question | Outcome |
|------------|----------|---------|
| MixLaw | Can a mixing law from short runs beat fixed mixtures? | **Yes** — fitted mixtures beat the baseline (\(p < 10^{-4}\)) |
| Skill-It | Can mid-run Skill-It reweighting beat the fixed baseline? | **No** — all arms ≤ control |

Shared evaluation: macro task-loss CE bits-per-byte over 20 OLMES-style labels; power-law residual-bootstrap CIs on fitted finals (steps ≥ 1000). A100-hours: MixLaw 188.74, Skill-It 148.2. FLOPs: \(2.63\times10^{19}\) per 370M arm (W&B); MixLaw 60M pilot grid \(\approx 2.3\times10^{18}\); Skill-It 60M probes \(\approx 6.8\times10^{17}\).
