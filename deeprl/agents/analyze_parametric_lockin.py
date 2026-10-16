"""Phase-locked amplitude of lambda at omega/2 per run group, and the pre-registered go/no-go.

Usage: python agents/analyze_parametric_lockin.py RUN_DIR [--omega-ref 0.026]
Run files are named <GROUP>_s<seed>.npz. Unmodulated runs (omega = 0) are
evaluated at omega_ref / 2.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import hilbert


def lockin(x: np.ndarray, t: np.ndarray, w: float) -> float:
    return float(2 * abs(np.mean(x * np.exp(-1j * w * t))))


def envelope_slope(lam: np.ndarray, w: float, start: int, length: int = 4000) -> float:
    """Log-slope of the omega/2 envelope (band-passed by demodulation) over the early window."""
    t = np.arange(start, start + length)
    x = lam[t] - lam[t].mean()
    z = x * np.exp(-1j * w * t)
    k = int(4 * np.pi / w)  # two sub-harmonic periods
    env = np.abs(np.convolve(z, np.ones(k) / k, "valid")) + 1e-9
    return float(np.polyfit(np.arange(env.size), np.log(env), 1)[0])


def main(run_dir: str, omega_ref: float) -> None:
    groups: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    for p in sorted(Path(run_dir).glob("*_s*.npz")):
        d = np.load(p)
        lam, n = d["lam"], d["lam"].size
        w = float(d["omega"]) or omega_ref
        t = np.arange(n // 2, n)
        x = lam[t] - lam[t].mean()
        groups[p.stem.rsplit("_s", 1)[0]].append(
            (
                lockin(x, t, w / 2),
                lockin(x, t, w),
                envelope_slope(lam, w / 2, int(d["learning_starts"])),
                float(np.mean(lam[t] == 0)),
                float(x.std()),
            )
        )
    for g in sorted(groups):
        a = np.array(groups[g])
        print(
            f"{g:12s} n={len(a)} A_sub med={np.median(a[:, 0]):.3f} seeds={np.round(a[:, 0], 3).tolist()} "
            f"A_w={np.median(a[:, 1]):.3f} slope={np.median(a[:, 2]):+.1e} frac0={np.median(a[:, 3]):.2f} std={np.median(a[:, 4]):.3f}"
        )
    if not {"F_e0.6", "A_e0", "OFF_e0.6", "COH_e0.6"} <= groups.keys():
        return  # go/no-go only defined on the pre-registered groups
    fresh = np.array(groups["F_e0.6"])[:, 0]
    controls = {k: np.array(groups[k])[:, 0] for k in ("A_e0", "OFF_e0.6", "COH_e0.6")}
    ratios = {k: np.median(fresh) / np.median(v) for k, v in controls.items()}
    ceiling = max(v.max() for v in controls.values())
    n_above = int(np.sum(fresh > ceiling))
    go = all(r >= 3 for r in ratios.values()) and n_above >= 4
    print("ratios:", {k: round(float(r), 1) for k, r in ratios.items()}, f"seeds above control max ({ceiling:.3f}): {n_above}/5")
    print("GO" if go else "NO-GO")


if __name__ == "__main__":
    ref = float(sys.argv[sys.argv.index("--omega-ref") + 1]) if "--omega-ref" in sys.argv else 0.026
    main(sys.argv[1], ref)
