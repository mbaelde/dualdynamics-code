"""PARAMETRIC forcing of the coupled policy-dual system (toy K=1).

Complement of `toy_k1_linear.py`, which handled ADDITIVE forcing
(g = c theta + tau_bar + A sin(omega t)) and concluded a bounded plateau
2 A a / c^2 at low frequency: no 1/omega divergence, refuting the initial
conjecture. What remains open in this file: can modulating a COEFFICIENT
(rather than an additive term) produce a genuine parametric resonance, i.e. a
true instability, in the slow regime?

The system becomes linear with periodic coefficients, the LTI frequency
response no longer applies, the correct tool is Floquet theory. We allow
three independent modulation channels, all in sin(omega t):

    eps_theta : c as seen by the PRIMAL update   (term -eta_theta c lambda)
    eps_lam   : c as seen by the DUAL update     (term +eta_lambda c theta)
    eps_a     : reward curvature a

    M_t = [[1 - eta_theta a_t, -eta_theta c_theta_t],
           [ eta_lambda c_lam_t,        1          ]]

Separating eps_theta from eps_lam is not gratuitous: in practice the primal
and the dual do not use the same constraint estimate (distinct critics,
shifted targets, asynchronous updates), so the sensitivity seen on the two
sides of the loop differs.

THEORY. In the continuous-time limit, with c_theta and c_lam variable:

    theta'' + (eta_theta a - c_theta'/c_theta) theta'
            + (eta_theta eta_lambda c_theta c_lam - eta_theta a c_theta'/c_theta) theta = 0

Modulating c therefore acts both on the stiffness AND on the damping. After
eliminating the theta' term (a standard change of variable), the effective
stiffness receives, to first order in eps, the forcing

    sin(omega t) * [ (eps_theta + eps_lam) omega_0^2 - (omega^2 / 2) eps_theta ]

with omega_0 = c sqrt(eta_theta eta_lambda) the natural frequency. At the
principal instability tongue omega = 2 omega_0, this coefficient equals

    (eps_lam - eps_theta) omega_0^2.

Hence the central prediction to test: the resonant parametric forcing is
proportional to the MISMATCH between the sensitivity seen by the primal and
that seen by the dual. A perfectly coherent modulation (eps_theta = eps_lam),
which is the case of a physical non-stationarity of the constraint, cancels
this forcing to first order in eps, at any depth, but not the second-order
term: that one is measured here, and stays negative up to eps = 0.99. The
instability threshold then reads on |eps_lam - eps_theta|:

    |eps_lam - eps_theta| > 4 zeta,   zeta = (a / 2c) sqrt(eta_theta / eta_lambda)

and this tolerance collapses as sqrt(eta_theta / eta_lambda) in the Borkar
regime eta_theta << eta_lambda.

Comparison point: with a frozen policy (eta_theta = 0), M_t is triangular
with diagonal (1,1), so rho(Phi) = 1 for any eps. No parametric resonance is
possible. The failure mode is CREATED by the coupling.

Run: python toy_k1_parametric.py
"""

from __future__ import annotations

import numpy as np

A_CURV = 1.0
C_SENS = 1.0
TAU_BAR = 1.0


def step_matrix(
    eta_t: float, eta_l: float, a: float = A_CURV, c_theta: float = C_SENS, c_lam: float = C_SENS
) -> np.ndarray:
    return np.array([[1.0 - eta_t * a, -eta_t * c_theta], [eta_l * c_lam, 1.0]])


def natural_mode(eta_t: float, eta_l: float, a: float = A_CURV, c: float = C_SENS) -> tuple[float, float]:
    """Natural frequency and per-step decay rate, read off the spectrum of the
    UNMODULATED system. More reliable than the continuous-time formulas for
    predicting the tongue.

    Returns (omega_nat, sigma), in that order. Swapping the two gives an
    inverse zeta that still looks plausible: check against
    zeta ~= (a/2c) r^-1/2. In the overdamped regime (r <= (a/c)^2 / 4) the
    spectrum is real and omega_nat = 0.
    """
    ev = np.linalg.eigvals(step_matrix(eta_t, eta_l, a=a, c_theta=c, c_lam=c))
    k = int(np.argmax(abs(ev)))
    return float(abs(np.angle(ev[k]))), float(-np.log(abs(ev[k])))


def monodromy(
    n_steps: int,
    eta_t: float,
    eta_l: float,
    eps_theta: float = 0.0,
    eps_lam: float = 0.0,
    eps_a: float = 0.0,
    a: float = A_CURV,
    c: float = C_SENS,
) -> np.ndarray:
    """Monodromy over one period of n_steps steps (omega = 2 pi / n_steps, an
    integer period so Floquet is exact)."""
    phi = np.eye(2)
    for t in range(n_steps):
        s = np.sin(2.0 * np.pi * t / n_steps)
        m = step_matrix(
            eta_t,
            eta_l,
            a=a * (1.0 + eps_a * s),
            c_theta=c * (1.0 + eps_theta * s),
            c_lam=c * (1.0 + eps_lam * s),
        )
        phi = m @ phi
    return phi


def floquet_exponent(n_steps: int, eta_t: float, eta_l: float, **kw) -> float:
    """Floquet exponent per step: > 0 means parametric instability."""
    rho = max(abs(np.linalg.eigvals(monodromy(n_steps, eta_t, eta_l, **kw))))
    return float(np.log(rho) / n_steps)


def n_for_omega(omega: float) -> int:
    return max(8, int(round(2.0 * np.pi / omega)))


def scan_omega(
    eta_t: float, eta_l: float, ratios: np.ndarray | None = None, **kw
) -> tuple[float, float]:
    """Sweeps omega around 2*omega_nat and returns (max exponent, ratio omega/2omega_nat)."""
    w_nat, _ = natural_mode(eta_t, eta_l)
    if ratios is None:
        ratios = np.concatenate([np.linspace(0.4, 1.6, 121), np.linspace(1.6, 5.0, 35)])
    best = (-np.inf, np.nan)
    for r in ratios:
        n = n_for_omega(r * 2.0 * w_nat)
        e = floquet_exponent(n, eta_t, eta_l, **kw)
        if e > best[0]:
            best = (e, float(r))
    return best


def mismatch_threshold(eta_t: float, eta_l: float, eps_theta: float = 0.0, hi: float = 1.5) -> float:
    """Smallest |eps_lam - eps_theta| that makes the system unstable, at omega = 2*omega_nat."""
    w_nat, _ = natural_mode(eta_t, eta_l)
    n = n_for_omega(2.0 * w_nat)

    def unstable(d: float) -> bool:
        return floquet_exponent(n, eta_t, eta_l, eps_theta=eps_theta, eps_lam=eps_theta + d) > 0.0

    if not unstable(hi):
        return float("nan")
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if unstable(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def simulate(
    omega: float,
    eta_t: float,
    eta_l: float,
    eps_theta: float = 0.0,
    eps_lam: float = 0.0,
    n_periods: int = 200,
    a: float = A_CURV,
    c: float = C_SENS,
    tau_bar: float = TAU_BAR,
    project: bool = True,
) -> tuple[float, float, float]:
    """Full recursion (affine term tau_bar, optional [.]_+ projection).

    Returns (max lambda reached, p2p over the last period, fraction of steps
    where the projection bites). The clipped fraction is the quantity of
    interest: lambda stuck at 0 means the constraint is not enforced over that
    portion of the cycle.
    """
    n_per = n_for_omega(omega)
    n_steps = n_periods * n_per
    theta = -tau_bar / c
    lam = a * tau_bar / c**2
    hist = np.full(n_steps, np.nan)
    n_clip = 0
    for t in range(n_steps):
        s = np.sin(2.0 * np.pi * (t % n_per) / n_per)
        c_th = c * (1.0 + eps_theta * s)
        c_la = c * (1.0 + eps_lam * s)
        theta_next = theta - eta_t * (a * theta + c_th * lam)
        lam_next = lam + eta_l * (c_la * theta + tau_bar)
        if lam_next < 0.0:
            n_clip += 1
            if project:
                lam_next = 0.0
        theta, lam = theta_next, lam_next
        hist[t] = lam
        if not np.isfinite(lam) or abs(lam) > 1e13:
            hist[t:] = lam
            n_clip = n_clip if project else n_clip
            break
    last = hist[-n_per:]
    return (
        float(np.nanmax(np.abs(hist))),
        float(np.nanmax(last) - np.nanmin(last)),
        n_clip / n_steps,
    )


def main() -> None:
    print("=== 0. Sanity check of the Floquet implementation (eps = 0: Phi = M^N) ===")
    for eta_t, eta_l in ((1e-3, 1e-1), (1e-2, 1e-2)):
        n = 128
        rho_m = float(max(abs(np.linalg.eigvals(step_matrix(eta_t, eta_l)))))
        e = floquet_exponent(n, eta_t, eta_l)
        print(f"  eta_t={eta_t:<7g} eta_l={eta_l:<7g} ln rho(M)={np.log(rho_m):+.3e}  Floquet={e:+.3e}")
        assert abs(e - np.log(rho_m)) < 1e-12, (eta_t, eta_l)
        assert rho_m < 1.0, "unmodulated system unstable, outside the hypothesis (eta_l c^2 < a)"

    print("\n=== 1. Frozen policy: parametric resonance is impossible ===")
    for eps in (0.0, 0.3, 0.9):
        e = floquet_exponent(256, 0.0, 1e-2, eps_theta=eps, eps_lam=eps)
        print(f"  eta_theta=0  eps={eps:<5g} Floquet exponent={e:+.3e}")
        assert abs(e) < 1e-15, eps
    print("  -> the failure mode tested here only exists in the COUPLED system.")

    eta_t, eta_l = 1e-3, 1e-1
    w_nat, dec = natural_mode(eta_t, eta_l)
    h_crit = 4.0 * dec / w_nat
    print(f"\n  Reference: eta_t={eta_t:g} eta_l={eta_l:g}")
    print(f"  omega_nat={w_nat:.6g} (continuous formula c*sqrt(eta_t*eta_l)={np.sqrt(eta_t*eta_l):.6g})")
    print(f"  decay/step={dec:.4g}  predicted stiffness threshold h_crit=4*dec/omega_nat={h_crit:.4g}")
    print(f"  expected principal tongue at omega=2*omega_nat={2*w_nat:.6g} ({n_for_omega(2*w_nat)} steps/period)")

    print("\n=== 2. Diagnostic: modulating the DUAL channel only (pure stiffness) ===")
    print("  Mathieu reference case: the tongue must appear above h_crit.")
    print(f"  {'eps_lam':>9} {'max exponent':>14} {'omega/2w_nat':>13} {'verdict':>10}")
    for eps in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        e, r = scan_omega(eta_t, eta_l, eps_lam=eps)
        print(f"  {eps:>9g} {e:>+14.4e} {r:>13.4g} {'UNSTABLE' if e > 0 else 'stable':>10}")
    e_lo, _ = scan_omega(eta_t, eta_l, eps_lam=0.5 * h_crit)
    e_hi, r_hi = scan_omega(eta_t, eta_l, eps_lam=2.0 * h_crit)
    assert e_lo < 0.0 < e_hi, (e_lo, e_hi)
    assert 0.9 <= r_hi <= 1.1, f"principal tongue misplaced: {r_hi}"
    print(f"  -> Floquet machinery and Mathieu prediction validated (tongue at omega/2w_nat={r_hi:.3g}).")

    print("\n=== 3. COHERENT modulation of c (physical case): the tongue disappears ===")
    print("  eps_theta = eps_lam: the resonant forcing cancels to first order.")
    print(f"  {'eps':>9} {'max exponent':>14} {'omega/2w_nat':>13} {'verdict':>10}")
    for eps in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        e, r = scan_omega(eta_t, eta_l, eps_theta=eps, eps_lam=eps)
        print(f"  {eps:>9g} {e:>+14.4e} {r:>13.4g} {'UNSTABLE' if e > 0 else 'stable':>10}")
    e_coh, _ = scan_omega(eta_t, eta_l, eps_theta=0.9, eps_lam=0.9)
    e_dual, _ = scan_omega(eta_t, eta_l, eps_lam=0.9)
    print(f"  -> at eps=0.9: coherent {e_coh:+.3e} vs. dual channel alone {e_dual:+.3e}")
    assert e_coh < e_dual, (e_coh, e_dual)

    print("\n=== 4. The resonant forcing is proportional to the MISMATCH eps_lam - eps_theta ===")
    print("  eps_theta fixed at 0.2, eps_lam swept around it.")
    print(f"  predicted threshold on |eps_lam - eps_theta|: h_crit={h_crit:.4g}")
    print(f"  {'eps_lam':>9} {'mismatch':>11} {'max exponent':>14} {'verdict':>10}")
    for eps_l_ in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        e, _ = scan_omega(eta_t, eta_l, eps_theta=0.2, eps_lam=eps_l_)
        print(f"  {eps_l_:>9g} {eps_l_-0.2:>+11.3g} {e:>+14.4e} {'UNSTABLE' if e > 0 else 'stable':>10}")
    e_match, _ = scan_omega(eta_t, eta_l, eps_theta=0.2, eps_lam=0.2)
    assert e_match < 0.0, e_match

    print("\n=== 5. Mismatch threshold vs. timescale ratio (Borkar regime) ===")
    print(f"  {'eta_t':>8} {'eta_l':>8} {'r=el/et':>9} {'meas. threshold':>15} {'h_crit':>9} {'ratio':>7} {'sqrt(et/el)':>12}")
    prev = None
    for eta_t_, eta_l_ in ((1e-1, 1e-1), (1e-2, 1e-1), (1e-3, 1e-1), (1e-4, 1e-1), (1e-5, 1e-1)):
        w_, d_ = natural_mode(eta_t_, eta_l_)
        hc = 4.0 * d_ / w_
        thr = mismatch_threshold(eta_t_, eta_l_, eps_theta=0.0)
        ratio = thr / hc if np.isfinite(thr) else float("nan")
        print(
            f"  {eta_t_:>8g} {eta_l_:>8g} {eta_l_/eta_t_:>9g} {thr:>15.4g} {hc:>9.4g}"
            f" {ratio:>7.2f} {np.sqrt(eta_t_/eta_l_):>12.4g}"
        )
        if np.isfinite(thr):
            assert 0.5 <= ratio <= 2.0, (eta_t_, eta_l_, ratio)
            if prev is not None:
                assert thr < prev, "the threshold must decrease as the timescale ratio increases"
            prev = thr
    print("  -> tolerance to mismatch collapses as sqrt(eta_theta/eta_lambda): the regime")
    print("     that makes the primal-dual analysis tractable is the one that tolerates no mismatch.")

    print("\n=== 6a. Full recursion WITHOUT projection: the divergence is real ===")
    w_star = 2.0 * w_nat
    n_per = n_for_omega(w_star)
    print(f"  omega*={w_star:.5g} ({n_per} steps/period), 200 periods = {200*n_per} steps")
    print(f"  {'eps_th':>7} {'eps_lam':>8} {'Floquet':>12} {'exp(Floquet*T)':>15} {'max lambda':>12}")
    for eps_th, eps_l_ in ((0.0, 0.0), (0.3, 0.3), (0.9, 0.9), (0.0, 0.3), (0.3, 0.0), (0.2, 0.5)):
        e = floquet_exponent(n_per, eta_t, eta_l, eps_theta=eps_th, eps_lam=eps_l_)
        pm, _, _ = simulate(w_star, eta_t, eta_l, eps_theta=eps_th, eps_lam=eps_l_, project=False)
        print(f"  {eps_th:>7g} {eps_l_:>8g} {e:>+12.3e} {np.exp(e*200*n_per):>15.3e} {pm:>12.4g}")
    pm_coh, _, _ = simulate(w_star, eta_t, eta_l, eps_theta=0.3, eps_lam=0.3, project=False)
    pm_mis, _, _ = simulate(w_star, eta_t, eta_l, eps_lam=0.3, project=False)
    print(f"  -> coherent: max lambda={pm_coh:.4g} (bounded). Mismatch: {pm_mis:.4g}.")
    assert pm_mis > 1e5 * pm_coh, (pm_coh, pm_mis)

    print("\n=== 6b. With [.]_+ projection: saturation into a limit cycle, constraint relaxed ===")
    print("  The divergence cannot blow up downward (lambda >= 0), it saturates instead.")
    print(f"  {'eps_th':>7} {'eps_lam':>8} {'Floquet':>12} {'max lambda':>11} {'final p2p':>10} {'% clipped':>9}")
    for eps_th, eps_l_ in ((0.0, 0.0), (0.3, 0.3), (0.0, 0.3), (0.2, 0.5)):
        e = floquet_exponent(n_per, eta_t, eta_l, eps_theta=eps_th, eps_lam=eps_l_)
        pm, p2, frac = simulate(w_star, eta_t, eta_l, eps_theta=eps_th, eps_lam=eps_l_)
        print(f"  {eps_th:>7g} {eps_l_:>8g} {e:>+12.3e} {pm:>11.4g} {p2:>10.4g} {100*frac:>8.1f}%")
    _, _, f_coh = simulate(w_star, eta_t, eta_l, eps_theta=0.3, eps_lam=0.3)
    _, _, f_mis = simulate(w_star, eta_t, eta_l, eps_lam=0.3)
    print(f"  -> the failure mode is not an infinite amplitude but lambda stuck at 0")
    print(f"     over {100*f_mis:.1f}% of the time under mismatch, vs. {100*f_coh:.1f}% under coherent modulation.")
    assert f_mis > f_coh, (f_coh, f_mis)

    print("\n=== 7. Modulating the curvature a (damping) ===")
    print("  Modulating a acts on damping, not on stiffness: expected effect is weak.")
    for eps in (0.3, 0.9):
        e, r = scan_omega(eta_t, eta_l, eps_a=eps)
        print(f"  eps_a={eps:<5g} max exponent={e:+.4e} at omega/2w_nat={r:.4g}  {'UNSTABLE' if e > 0 else 'stable'}")

    print("\n=== 8. Sanity check: outside the tongue, same mismatch, no divergence ===")
    print(f"  {'omega/omega*':>13} {'Floquet':>12} {'max lambda (no proj.)':>24}")
    for ratio in (0.4, 0.7, 1.0, 1.5, 2.5):
        w = ratio * w_star
        e = floquet_exponent(n_for_omega(w), eta_t, eta_l, eps_lam=0.3)
        pm, _, _ = simulate(w, eta_t, eta_l, eps_lam=0.3, project=False)
        print(f"  {ratio:>13g} {e:>+12.3e} {pm:>24.4g}")
    print("  -> the instability is confined to the tongue: it is a resonance, not a generic drift.")

    print("\nOK")


if __name__ == "__main__":
    main()
