#!/usr/bin/env python3
plan = {
    "dclm": 28.6,
    "arxiv": 20.8,
    "starcoder": 20.3,
    "pes2o": 26.3,
    "open-web-math": 12.2,
    "algebraic-stack": 11.8,
    "wiki": 3.66,
}
meas = {
    "dclm": 29.691,
    "arxiv": 22.148,
    "starcoder": 6.139,
    "pes2o": 12.309,
    "open-web-math": 13.238,
    "algebraic-stack": 12.902,
    "wiki": 3.752,
}
print(f"{'domain':<18}{'plan':>8}{'meas':>8}{'pct_err':>10}{'ok':>6}")
for d in plan:
    err = abs(plan[d] - meas[d]) / meas[d]
    ok = err <= 0.10
    print(f"{d:<18}{plan[d]:8.3f}{meas[d]:8.3f}{100*err:9.1f}%{str(ok):>6}")
    if not ok:
        # How much to upsample measured to reach plan (within 10%)
        target_lo = plan[d] / 1.10
        target_hi = plan[d] * 1.10
        need = max(0.0, target_lo - meas[d])
        print(f"  -> measured short of plan/1.10={target_lo:.3f}B; need +{need:.3f}B to enter 10% band")
