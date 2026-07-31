import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mixlaw"), str(ROOT / "skillit")]
from mixlaw_common import CURVE_FAMILIES, DOMAINS
from skillit_math import default_mixlaw_fit_path, load_fit_json, online_A_from_fit, regmix_weight_vector

A_off = np.load(Path(__file__).parent / "artifacts/probes_full/A_offline.npy")
A_on = online_A_from_fit(load_fit_json(default_mixlaw_fit_path()), regmix_weight_vector(DOMAINS))
for name, A in [("OFFLINE", A_off), ("ONLINE", A_on)]:
    print(name)
    for i, d in enumerate(DOMAINS):
        print(d, *[round(A[i, j], 4) for j in range(6)])
