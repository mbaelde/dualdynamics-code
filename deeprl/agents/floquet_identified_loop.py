"""Floquet threshold of the identified linearised primal-dual loop.

Primal (identified by OLS on eps=0 runs): dx_t = (1 - k_a) dx_{t-1} - k_l dlam_t.
Dual (exact, PID-Lagrangian with integral I and proportional gain kp):
    G_t = (1/W) sum_{s=t-W+1..t} c_s dx_{s-1},  c_s = 1 + eps sin(w s),
    I_{t+1} = I_t + eta G_t,  lam_{t+1} = I_{t+1} + kp G_t   (kp = 0: plain Lagrangian).
State z_t = (dI_t, dx_{t-1}, ..., dx_{t-W-1}).

Usage:
    python agents/floquet_identified_loop.py <dir with A_e0_s*.npz>            # identify + thresholds
    python agents/floquet_identified_loop.py <dir> --design                     # predictions for r grid
"""

import sys
from pathlib import Path

import numpy as np

ETA0, POLICY_LR0 = 0.1, 3e-3  # step sizes of the eps=0 identification runs
W_MOD = 2 * np.pi / 242  # ~0.026, commensurate so the burn-in is phase-agnostic


def identify(path: Path) -> tuple[float, float]:
    d = np.load(path)
    lam, x = d["lam"], d["a_mean"]
    n = len(x)
    lam, x = lam[n // 2 :], x[n // 2 :]
    dx = x[1:] - x[:-1]
    reg = np.column_stack([x[:-1] - x.mean(), lam[1:] - lam.mean(), np.ones(len(dx))])
    coef, *_ = np.linalg.lstsq(reg, dx, rcond=None)
    return -coef[0], -coef[1]


def step_matrix(k_a: float, k_l: float, eta: float, kp: float, c_new: np.ndarray, c_old: np.ndarray) -> np.ndarray:
    """c_new[j] = c_s for s = t-W+1+j (window of G_t), c_old the same for G_{t-1}."""
    w = len(c_new)
    m = np.zeros((w + 2, w + 2))
    g_new = np.zeros(w + 2)  # G_t over x_{t-W}..x_{t-1}
    g_old = np.zeros(w + 2)  # G_{t-1} over x_{t-W-1}..x_{t-2}
    for j in range(w):
        g_new[w - j] += c_new[j] / w
        g_old[w + 1 - j] += c_old[j] / w
    m[0] = g_new * eta
    m[0, 0] += 1.0
    m[1] = -k_l * kp * g_old
    m[1, 0] -= k_l
    m[1, 1] += 1 - k_a
    for i in range(2, w + 2):
        m[i, i - 1] = 1.0
    return m


def max_exponent(k_a: float, k_l: float, eta: float, kp: float, eps: float, w: float, win: int, n: int = 30000) -> float:
    """Top Lyapunov exponent per step by renormalised power iteration."""
    z = np.ones(win + 2)
    acc, burn = 0.0, 3000
    for t in range(n):
        s = np.arange(t - win + 1, t + 1)
        m = step_matrix(k_a, k_l, eta, kp, 1 + eps * np.sin(w * s), 1 + eps * np.sin(w * (s - 1)))
        z = m @ z
        nz = np.linalg.norm(z)
        z /= nz
        if t >= burn:
            acc += np.log(nz)
    return acc / (n - burn)


def threshold(k_a: float, k_l: float, eta: float, kp: float, w: float, win: int, hi: float = 2.0) -> float:
    if max_exponent(k_a, k_l, eta, kp, hi, w, win) <= 0:
        return np.inf
    lo = 0.0
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if max_exponent(k_a, k_l, eta, kp, mid, w, win) <= 0 else (lo, mid)
    return hi


def scaled(k_a0: float, k_l0: float, r: float) -> tuple[float, float, float, float]:
    """Step sizes at timescale ratio r with eta_theta * eta_lambda (hence omega_0) held fixed.

    Assumes the effective primal gain scales linearly with the SGD policy lr.
    Returns (k_a, k_l, lambda_lr, policy_lr).
    """
    eta_th = np.sqrt(k_a0 * ETA0 / r)
    return eta_th, k_l0 / k_a0 * eta_th, r * eta_th, POLICY_LR0 * eta_th / k_a0


if __name__ == "__main__":
    files = sorted(Path(sys.argv[1]).glob("A_e0_s*.npz"))
    ks = np.array([identify(f) for f in files])
    for f, (ka, kl) in zip(files, ks):
        print(f"{f.stem}: k_a={ka:.3e} k_l={kl:.3e}")
    k_a, k_l = ks.mean(axis=0)
    print(f"pooled: k_a={k_a:.3e} k_l={k_l:.3e} r={ETA0 / k_a:.1f}")
    assert max_exponent(k_a, k_l, ETA0, 0.0, 0.0, 0.026, 10) < 0  # unforced loop is stable

    if "--design" in sys.argv:
        for r in (1, 10, 100):
            ka, kl, eta, plr = scaled(k_a, k_l, r)
            print(f"r={r}: policy_lr={plr:.3e} lambda_lr={eta:.3e} k_a={ka:.2e}")
            for kp in (0.0, 0.5, 1.0):
                print(f"  kp={kp}: eps_c={threshold(ka, kl, eta, kp, W_MOD, 10):.3f}")
                print("    eps=0.4 mu(w):", " ".join(f"{w}:{max_exponent(ka, kl, eta, kp, 0.4, w, 10):+.1e}" for w in (0.018, 0.022, 0.026, 0.030, 0.036)))
        sys.exit()

    for win in (10, 1):
        print(f"W={win}: eps_c(w=0.026, pooled)={threshold(k_a, k_l, ETA0, 0.0, W_MOD, win):.3f}")
    for ka, kl in ks:
        print(f"  per-seed eps_c W=10: {threshold(ka, kl, ETA0, 0.0, W_MOD, 10):.3f}  W=1: {threshold(ka, kl, ETA0, 0.0, W_MOD, 1):.3f}")
    for w in (0.018, 0.022, 0.026, 0.030, 0.036):
        print(f"tongue eps=0.4 w={w}: mu W=10 {max_exponent(k_a, k_l, ETA0, 0.0, 0.4, w, 10):+.2e}  W=1 {max_exponent(k_a, k_l, ETA0, 0.0, 0.4, w, 1):+.2e}")
