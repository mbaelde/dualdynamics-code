"""Sub-harmonic power of lambda (band around omega/2) per (dual_signal, eps), and the go/no-go ratios."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch

runs = defaultdict(list)
for p in sorted(Path(sys.argv[1]).glob("*.npz")):
    d = np.load(p)
    lam = d["lam"][d["lam"].size // 2 :]
    f, pxx = welch(lam - lam.mean(), fs=1.0, nperseg=4096)
    f_sub = float(d["omega"]) / 2 / (2 * np.pi)
    band = (f > 0.7 * f_sub) & (f < 1.3 * f_sub)
    fwd = (f > 0.85 * 2 * f_sub) & (f < 1.15 * 2 * f_sub)
    runs[(str(d["dual_signal"]), float(d["eps"]))].append(
        (np.trapezoid(pxx[band], f[band]), np.trapezoid(pxx[fwd], f[fwd]), lam.std(), float(d["c_hat"][-5000:].std()))
    )

med = {}
for k in sorted(runs):
    a = np.array(runs[k])
    med[k] = np.median(a[:, 0])
    print(f"{k[0]:6s} eps={k[1]:.1f} n={len(a)} P_sub med={med[k]:.2e} per-seed={np.round(a[:, 0] / 1e-6, 2).tolist()}e-6 "
          f"P_fwd={np.median(a[:, 1]):.2e} std_lam={np.median(a[:, 2]):.4f} std_chat={np.median(a[:, 3]):.4f}")
for e in (0.3, 0.5):
    print(f"eps={e}: fresh/critic={med[('fresh', e)] / med[('critic', e)]:.2f} fresh/eps0={med[('fresh', e)] / med[('fresh', 0.0)]:.2f}")
