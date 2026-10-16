"""Operational metric of the projected regime: coherent vs. mismatched.

Complement of `toy_k1_parametric.py` section 6b (fraction of clipped steps),
whose raw figure is weak evidence on its own: 28.0% clipped steps under
mismatch vs. 26.7% under coherent modulation, a gap that does not visibly
separate the two regimes.

Two operational metrics, integrated over the horizon rather than pointwise:

    integrated violation: time average of max(0, c*theta_t + tau_bar), the
        uncompensated constraint signal when the multiplier is stuck at zero
        (violation that is not bought back by the price).
    regret on J: time average of (a/2)(theta_t - theta*)^2, the excess
        quadratic cost relative to the unmodulated saddle.

Measured at several depths and at two operating points (timescale ratio
r = eta_lambda / eta_theta ~ 1e2, the one of Table `tab:opoint` of the paper, and
r ~ 1e1, closer to the overdamped boundary) to check that the separation is
not an artifact of a single operating point.

Run: python projected_regime_metrics.py
"""

from __future__ import annotations

import numpy as np

from toy_k1_parametric import A_CURV, C_SENS, TAU_BAR, n_for_omega, natural_mode


def simulate_metrics(
    omega: float,
    eta_t: float,
    eta_l: float,
    eps_theta: float = 0.0,
    eps_lam: float = 0.0,
    n_periods: int = 400,
    a: float = A_CURV,
    c: float = C_SENS,
    tau_bar: float = TAU_BAR,
) -> tuple[float, float, float]:
    """Projected recursion. Returns (integrated violation, regret on J, % clipped),
    time averages over the whole horizon (the transient is swallowed by 400
    periods, cf. `toy_k1_parametric.simulate` which uses 200)."""
    n_per = n_for_omega(omega)
    n_steps = n_periods * n_per
    theta_star = -tau_bar / c
    theta, lam = theta_star, a * tau_bar / c**2
    viol_sum = 0.0
    regret_sum = 0.0
    n_clip = 0
    for t in range(n_steps):
        s = np.sin(2.0 * np.pi * (t % n_per) / n_per)
        c_th = c * (1.0 + eps_theta * s)
        c_la = c * (1.0 + eps_lam * s)
        g_t = c_th * theta + tau_bar
        viol_sum += max(0.0, g_t)
        regret_sum += 0.5 * a * (theta - theta_star) ** 2
        theta_next = theta - eta_t * (a * theta + c_th * lam)
        lam_next = lam + eta_l * (c_la * theta + tau_bar)
        if lam_next < 0.0:
            n_clip += 1
            lam_next = 0.0
        theta, lam = theta_next, lam_next
        if not np.isfinite(theta):
            return float("nan"), float("nan"), n_clip / (t + 1)
    return viol_sum / n_steps, regret_sum / n_steps, n_clip / n_steps


def report(eta_t: float, eta_l: float, depths: tuple[float, ...]) -> None:
    w_nat, _ = natural_mode(eta_t, eta_l)
    w_star = 2.0 * w_nat
    r = eta_l / eta_t
    print(f"\n--- eta_theta={eta_t:g} eta_lambda={eta_l:g} (r={r:.3g}) ---")
    print(f"  {'depth':>6} {'viol coh':>11} {'viol mis':>11} {'ratio':>7} "
          f"{'regret coh':>11} {'regret mis':>11} {'ratio':>7} {'%clip coh':>9} {'%clip mis':>9}")
    for eps in depths:
        v_c, r_c, f_c = simulate_metrics(w_star, eta_t, eta_l, eps_theta=eps, eps_lam=eps)
        v_m, r_m, f_m = simulate_metrics(w_star, eta_t, eta_l, eps_lam=eps)
        rv = v_m / v_c if v_c > 0 else float("inf")
        rr = r_m / r_c if r_c > 0 else float("inf")
        print(
            f"  {eps:>6g} {v_c:>11.4g} {v_m:>11.4g} {rv:>7.2f} "
            f"{r_c:>11.4g} {r_m:>11.4g} {rr:>7.2f} {100*f_c:>8.1f}% {100*f_m:>8.1f}%"
        )


def main() -> None:
    depths = (0.3, 0.5, 0.7, 0.9)
    print("=== Integrated metrics, projected system, coherent vs. mismatched ===")
    report(eta_t=1e-3, eta_l=1e-1, depths=depths)   # r=1e2, operating point of Table `tab:opoint`
    report(eta_t=1e-2, eta_l=1e-1, depths=depths)   # r=1e1, closer to the overdamped boundary
    print("\nOK")


if __name__ == "__main__":
    main()
