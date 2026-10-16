"""Where does the parametric resonance sit relative to Borkar's timescale ordering?

Complement to `toy_k1_parametric.py`, motivated by Borkar-style two-timescale
separation. The parametric strand measured the
mismatch threshold for r = eta_lambda / eta_theta in {10, 1e2, 1e3, 1e4}, i.e.
only on the FAST-DUAL side. But the CMDP literature following Borkar (2005)
puts the multiplier on the SLOWER timescale, i.e. r << 1. We need to know what
happens to the phenomenon on that side before writing anything about the
paper's relation to timescale separation.

Two methodological corrections relative to `toy_k1_parametric.py`:

1. `mismatch_threshold` only evaluates Floquet at omega = 2 omega_nat, the
   center of the tongue of the UNMODULATED system. At strong mismatch the
   tongue tilts and its maximum shifts (measured here toward 0.9 x 2 omega_nat),
   so a threshold read at the center is overestimated. Here we sweep the
   period on an absolute grid and take the max over omega, which gives the
   true threshold.
2. In the overdamped regime (small r) the spectrum of the unmodulated system
   is real: `natural_mode` returns omega_nat = 0 and any grid relative to
   2 omega_nat is undefined. The absolute grid is the only one that covers
   both regimes.

Experiment design: the product eta_theta * eta_lambda is held at 1e-4, so
omega_0 = c sqrt(eta_theta eta_lambda) = 1e-2 is constant, and only
zeta = (a / 2c) sqrt(eta_theta / eta_lambda) = 0.5 / sqrt(r) varies. The
first-order predicted threshold, 4 zeta = 2 / sqrt(r), crosses the physically
admissible maximal mismatch (|eps| <= 1 on each side, so mismatch <= 2) at
r = (a/c)^2 = 1.

This value of 1 is the FIRST-ORDER boundary, not the true one. The tongue of
the discrete map tilts, so the true threshold (max over omega) is lower than
the threshold read at the center: measured at r = 1, it is 1.9484 for
eta_theta = eta_lambda = 1e-2 and 1.8507 for eta_theta = eta_lambda = 1e-1, in
both cases below the bound of 2. The measured boundary is r* in (0.96, 0.98]
at product 1e-4 and in (0.85, 0.90] at product 1e-2: the gap widens as the
step sizes grow. Only the overdamped half is exact, r <= (a/c)^2 / 4 = 0.25,
where the unmodulated spectrum is real (discriminant
eta_theta^2 a^2 - 4 eta_theta eta_lambda c^2 <= 0) and there is no oscillating
mode to excite, at any mismatch. The rates just beyond the boundary are tiny:
+3.4e-5 per step at r = 0.98 and +8.4e-5 at r = 1 (product 1e-4), i.e.
doubling times of 2e4 and 8e3 steps.

Run: python borkar_regime_check.py
"""

from __future__ import annotations

import numpy as np

from toy_k1_parametric import floquet_exponent, natural_mode, step_matrix

# Coarse absolute grid, in steps. Upper bound at 3000 steps: beyond that we are
# three decades below omega_0 = 1e-2, outside any tongue. It alone covers the
# overdamped regime, where there is no omega_nat to refine around.
COARSE_PERIODS = np.unique(np.geomspace(8, 3000, 60).astype(int))

# r = eta_lambda / eta_theta, at fixed product eta_theta * eta_lambda = 1e-4.
CONFIGS = [(r, np.sqrt(1e-4 / r), np.sqrt(1e-4 * r)) for r in (1e-2, 1e-1, 1.0, 1e1, 1e2)]

EPS_MAX = 0.99  # |eps| <= 1: beyond that the sensitivity changes sign


def period_grid(eta_t: float, eta_l: float) -> np.ndarray:
    """Coarse absolute grid, refined around the principal tongue if it exists.

    Without this refinement the sweep misses narrow tongues (width proportional
    to the mismatch), and the measured threshold is then an upper bound set by
    the grid step rather than by the physics: measured at 0.2499 instead of
    0.18 at r = 1e2.
    """
    w_nat, _ = natural_mode(eta_t, eta_l)
    if w_nat <= 0.0:
        return COARSE_PERIODS
    n_star = np.pi / w_nat  # period of the principal tongue, omega = 2 omega_nat
    fine = np.round(np.linspace(0.55 * n_star, 1.45 * n_star, 121)).astype(int)
    return np.unique(np.concatenate([COARSE_PERIODS, fine]))


def max_exponent(eta_t: float, eta_l: float, **kw) -> tuple[float, int]:
    """Maximal Floquet exponent over the period grid, and the period reaching it."""
    best = (-np.inf, 0)
    for n in period_grid(eta_t, eta_l):
        e = floquet_exponent(int(n), eta_t, eta_l, **kw)
        if e > best[0]:
            best = (e, int(n))
    return best


def decay_rate(eta_t: float, eta_l: float) -> float:
    """Per-step decay rate of the unmodulated system (> 0 means stable)."""
    rho = max(abs(np.linalg.eigvals(step_matrix(eta_t, eta_l))))
    return float(-np.log(rho))


def threshold_any_omega(eta_t: float, eta_l: float, hi: float = 2.0 * EPS_MAX) -> float:
    """Smallest mismatch |eps_lam - eps_theta| unstable for AT LEAST one omega.

    Mismatch split symmetrically (eps_theta = +d/2, eps_lam = -d/2) to stay
    within the admissible domain as long as possible.
    """

    def unstable(d: float) -> bool:
        return max_exponent(eta_t, eta_l, eps_theta=0.5 * d, eps_lam=-0.5 * d)[0] > 0.0

    if not unstable(hi):
        return float("nan")
    lo = 0.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if unstable(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main() -> None:
    print("=== 1. Regime of the unmodulated system as a function of r = eta_lambda / eta_theta ===")
    print(f"{'r':>8} {'eta_theta':>10} {'eta_lambda':>11} {'omega_nat':>11} {'sigma':>10} {'zeta=0.5/sqrt(r)':>17} {'4 zeta':>8}")
    modes = {}
    for r, eta_t, eta_l in CONFIGS:
        w_nat, _ = natural_mode(eta_t, eta_l)
        sigma = decay_rate(eta_t, eta_l)
        zeta = 0.5 / np.sqrt(r)
        modes[r] = w_nat
        print(f"{r:8.0e} {eta_t:10.4e} {eta_l:11.4e} {w_nat:11.6f} {sigma:10.3e} {zeta:17.4g} {4 * zeta:8.4g}")

    # Overdamped (real spectrum) exactly when zeta >= 1, i.e. r <= (a/2c)^2 * 4 = 1...
    # in discrete time the boundary shifts a bit, we just check the two extremes.
    assert modes[1e-2] == 0.0, "r = 1e-2 should be overdamped (no oscillating mode)"
    assert modes[1e2] > 0.0, "r = 1e2 should be underdamped"

    print("\n=== 2. MAXIMAL mismatch (eps_theta = +0.99, eps_lam = -0.99) ===")
    print(f"{'r':>8} {'max exponent':>14} {'period (steps)':>14} {'sigma':>10} {'verdict':>10}")
    exps = {}
    for r, eta_t, eta_l in CONFIGS:
        e, n = max_exponent(eta_t, eta_l, eps_theta=EPS_MAX, eps_lam=-EPS_MAX)
        exps[r] = e
        print(f"{r:8.0e} {e:+14.4e} {n:14d} {decay_rate(eta_t, eta_l):10.3e} {'UNSTABLE' if e > 0 else 'stable':>10}")

    assert exps[1e2] > 0.0, "r = 1e2 must already be unstable at maximal mismatch"
    assert exps[1e1] > 0.0, "r = 1e1 must already be unstable at maximal mismatch"
    assert exps[1e-1] < 0.0, "r = 1e-1 (slow dual) must resist maximal mismatch"
    assert exps[1e-2] < 0.0, "r = 1e-2 (Borkar CMDP ordering) must resist maximal mismatch"

    print("\n=== 3. COHERENT modulation (eps_theta = eps_lam = 0.99) ===")
    print(f"{'r':>8} {'max exponent':>14} {'period (steps)':>14} {'verdict':>10}")
    for r, eta_t, eta_l in CONFIGS:
        e, n = max_exponent(eta_t, eta_l, eps_theta=EPS_MAX, eps_lam=EPS_MAX)
        print(f"{r:8.0e} {e:+14.4e} {n:14d} {'UNSTABLE' if e > 0 else 'stable':>10}")
        assert e < 0.0, f"(R3) must hold at both ends of the scale, breaks at r = {r}"

    print("\n=== 4. Mismatch threshold, max over omega vs. reading at the tongue center ===")
    print(f"{'r':>8} {'threshold (max over omega)':>22} {'4 zeta predicted':>14} {'ratio':>8}")
    for r, eta_t, eta_l in CONFIGS:
        thr = threshold_any_omega(eta_t, eta_l)
        pred = 2.0 / np.sqrt(r)
        ratio = thr / pred if np.isfinite(thr) else np.nan
        print(f"{r:8.0e} {thr:22.4f} {pred:14.4g} {ratio:8.3f}")
        if r >= 10.0:
            assert 0.7 < ratio < 1.15, f"threshold outside tolerance of the 4 zeta prediction at r = {r}"

    # At r = 1 the prediction 4 zeta = 2 coincides with the maximal admissible
    # mismatch: the true threshold (max over omega) must fall just below 2, not
    # be absent, which reading only at the tongue center missed.
    thr_r1 = threshold_any_omega(1e-2, 1e-2)
    print(f"\nr = 1: true threshold = {thr_r1:.4f} (prediction 4 zeta = 2.000)")
    assert np.isfinite(thr_r1) and thr_r1 > 1.7, "at r = 1 the threshold must exist but stay near-maximal"

    print("\nAll asserts pass.")


if __name__ == "__main__":
    main()
