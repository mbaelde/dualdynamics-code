"""Toy K=1 linear analysis of the coupled policy-dual system under sinusoidal forcing.

Purpose: test the initial conjecture, namely that
the asymptotic peak-to-peak amplitude of the dual satisfies
Omega(eta_lambda / omega) in the COUPLED system, independently of eta_theta.

Model (scalar, affine constraint, concave quadratic reward):

    J(theta)         = -(a/2) theta^2
    g(theta ; xi_t)  = c theta + tau_bar + A sin(omega t)      [constraint g <= 0]

    theta_{t+1} = theta_t + eta_theta * d/dtheta [ J - lambda_t g ]
                = theta_t - eta_theta ( a theta_t + c lambda_t )
    lambda_{t+1} = [ lambda_t + eta_lambda g(theta_t ; xi_t) ]_+

Unforced stationary point: g = 0 gives theta* = -tau_bar/c, then
a theta* + c lambda* = 0 gives lambda* = a tau_bar / c^2 > 0 (active
constraint, so the deviation analysis stays in the interior lambda > 0 and
the projection [.]_+ is inactive as long as the amplitude stays below
lambda*).

In deviation (theta~, lambda~) the system is LTI with input
u_t = eta_lambda A sin(omega t):

    M = [[1 - eta_theta a, -eta_theta c],
         [ eta_lambda c  ,        1     ]]

    H_lambda(z) = (z - 1 + eta_theta a) / D(z)
    D(z)        = (z - 1 + eta_theta a)(z - 1) + eta_theta eta_lambda c^2

and the steady-state peak-to-peak amplitude is
2 eta_lambda A |H_lambda(e^{j omega})|.

Two limits we want to compare:
  - frozen policy (eta_theta = 0): H = 1/(z-1), amplitude = eta_lambda A / sin(omega/2),
    i.e. the O(alpha/omega) frozen-policy law.
  - coupled, omega -> 0: H -> a / (eta_lambda c^2), amplitude -> 2 A a / c^2,
    a constant independent of BOTH learning rates.

Run: python toy_k1_linear.py
"""

from __future__ import annotations

import numpy as np

# Reference parameters. Stability of the coupled system: eta_lambda c^2 < a
# (cf. the Jury criterion check in check_stability_condition).
A_CURV = 1.0  # a: reward curvature
C_SENS = 1.0  # c: constraint sensitivity to theta
TAU_BAR = 1.0
AMP = 0.1  # A: forcing amplitude
ETA_L = 0.01  # eta_lambda
ETA_T = 0.05  # eta_theta


def peak_to_peak_analytic(
    omega: float, eta_t: float, eta_l: float, a: float = A_CURV, c: float = C_SENS, amp: float = AMP
) -> float:
    """Steady-state peak-to-peak amplitude of lambda, via the frequency response."""
    z = np.exp(1j * omega)
    num = z - 1.0 + eta_t * a
    den = (z - 1.0 + eta_t * a) * (z - 1.0) + eta_t * eta_l * c**2
    return float(2.0 * eta_l * amp * abs(num / den))


def spectral_radius(eta_t: float, eta_l: float, a: float = A_CURV, c: float = C_SENS) -> float:
    m = np.array([[1.0 - eta_t * a, -eta_t * c], [eta_l * c, 1.0]])
    return float(max(abs(np.linalg.eigvals(m))))


def simulate(
    omega: float,
    eta_t: float,
    eta_l: float,
    n_periods: int = 40,
    a: float = A_CURV,
    c: float = C_SENS,
    amp: float = AMP,
    tau_bar: float = TAU_BAR,
    project: bool = True,
) -> float:
    """Simulates the non-linearized recursion (with projection) and measures the
    peak-to-peak amplitude over the last period, once the transient has died out."""
    period = 2.0 * np.pi / omega
    n_steps = int(n_periods * period)
    theta = -tau_bar / c
    lam = a * tau_bar / c**2
    lam_hist = np.empty(n_steps)
    for t in range(n_steps):
        g = c * theta + tau_bar + amp * np.sin(omega * t)
        theta_next = theta - eta_t * (a * theta + c * lam)
        lam_next = lam + eta_l * g
        if project:
            lam_next = max(0.0, lam_next)
        theta, lam = theta_next, lam_next
        lam_hist[t] = lam
    tail = lam_hist[-int(period) :]
    return float(tail.max() - tail.min())


def check_stability_condition() -> None:
    """The Jury criterion on z^2 - (2 - eta_t a) z + (1 - eta_t a + eta_t eta_l c^2)
    requires eta_lambda c^2 < a. Numerical check of the threshold."""
    for eta_l, expect_stable in ((0.5, True), (1.5, False)):
        rho = spectral_radius(ETA_T, eta_l, a=1.0, c=1.0)
        assert (rho < 1.0) == expect_stable, (eta_l, rho)


def main() -> None:
    check_stability_condition()

    print("=== 1. Simulation vs. analytic validation (coupled system) ===")
    for omega in (0.3, 0.03, 0.003):
        sim = simulate(omega, ETA_T, ETA_L)
        ana = peak_to_peak_analytic(omega, ETA_T, ETA_L)
        print(f"  omega={omega:<8g} sim={sim:.6f}  analytic={ana:.6f}  gap={abs(sim-ana)/ana:.2%}")
        assert abs(sim - ana) / ana < 0.05, (omega, sim, ana)

    print("\n=== 2. Frozen policy (eta_theta=0): should recover the Prop 1 law ===")
    for omega in (0.3, 0.03, 0.003):
        ana = peak_to_peak_analytic(omega, 0.0, ETA_L)
        prop1 = ETA_L * AMP / np.sin(omega / 2.0)
        print(f"  omega={omega:<8g} H-based={ana:.6f}  eta*A/sin(w/2)={prop1:.6f}")
        assert abs(ana - prop1) / prop1 < 1e-9, (omega, ana, prop1)

    print("\n=== 3. Sweep over omega: coupled vs. frozen ===")
    plateau = 2.0 * AMP * A_CURV / C_SENS**2
    omega_c = ETA_L * C_SENS**2 / A_CURV
    print(f"  predicted plateau 2*A*a/c^2 = {plateau:.6f}   omega_c = eta_l*c^2/a = {omega_c:g}")
    print(f"  {'omega':>10} {'coupled':>12} {'frozen':>12} {'frozen/coupled':>14}")
    for omega in (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
        cpl = peak_to_peak_analytic(omega, ETA_T, ETA_L)
        frz = peak_to_peak_analytic(omega, 0.0, ETA_L)
        print(f"  {omega:>10g} {cpl:>12.6f} {frz:>12.6f} {frz/cpl:>14.1f}")
    # The coupled system saturates, the frozen one diverges as 1/omega.
    assert abs(peak_to_peak_analytic(1e-6, ETA_T, ETA_L) - plateau) / plateau < 1e-3
    assert peak_to_peak_analytic(1e-6, 0.0, ETA_L) > 1e3 * plateau

    print("\n=== 4. Dependence on eta_theta at low frequency (test 'independently of eta_theta') ===")
    for eta_t in (1e-3, 1e-2, 1e-1, 0.5):
        vals = [peak_to_peak_analytic(w, eta_t, ETA_L) for w in (1e-5, 1e-3, 1e-1)]
        rho = spectral_radius(eta_t, ETA_L)
        print(f"  eta_theta={eta_t:<7g} rho={rho:.5f}  p2p(1e-5)={vals[0]:.6f} (1e-3)={vals[1]:.6f} (1e-1)={vals[2]:.6f}")

    print("\n=== 5. Dependence of the plateau on eta_lambda ===")
    for eta_l in (1e-4, 1e-3, 1e-2, 1e-1):
        p = peak_to_peak_analytic(1e-8, ETA_T, eta_l)
        print(f"  eta_lambda={eta_l:<7g} p2p(omega->0)={p:.6f}  (predicted plateau {plateau:.6f})")
        assert abs(p - plateau) / plateau < 1e-3, (eta_l, p)

    print("\n=== 6. Resonance peak: sup over omega of the amplitude ===")
    grid = np.logspace(-7, np.log10(np.pi), 4000)
    for eta_l in (1e-3, 1e-2, 1e-1):
        for eta_t in (1e-2, 1e-1):
            vals = np.array([peak_to_peak_analytic(w, eta_t, eta_l) for w in grid])
            i = int(vals.argmax())
            print(
                f"  eta_l={eta_l:<7g} eta_t={eta_t:<6g} sup={vals[i]:.6f} reached at omega={grid[i]:.3g}"
                f"  (plateau {plateau:.6f})"
            )

    print("\n=== 7. Law of the resonance peak as a function of (eta_theta, eta_lambda) ===")
    print("  Complex poles iff eta_theta a^2 < 4 eta_lambda c^2 (discriminant eps^2 - 4 delta).")
    print("  Observed law: sup_omega p2p ~ plateau * max(1, eta_lambda / eta_theta).")
    print(f"  {'eta_t':>8} {'eta_l':>8} {'ratio r':>8} {'sup_p2p':>10} {'sup/plateau':>12} {'/max(1,r)':>10}")
    for eta_t in (1e-3, 1e-2, 1e-1):
        for eta_l in (1e-3, 1e-2, 1e-1):
            if spectral_radius(eta_t, eta_l) >= 1.0:
                continue
            vals = np.array([peak_to_peak_analytic(w, eta_t, eta_l) for w in grid])
            sup = float(vals.max())
            r = eta_l / eta_t
            ratio = sup / plateau / max(1.0, r)
            print(f"  {eta_t:>8g} {eta_l:>8g} {r:>8g} {sup:>10.5f} {sup/plateau:>12.2f} {ratio:>10.2f}")
            # The peak is always at least the plateau, and grows at most linearly in r.
            # The plateau is the omega -> 0 limit, so a sup not reached on a finite grid.
            assert sup >= plateau * (1.0 - 1e-6), (eta_t, eta_l, sup)
            assert 0.9 <= ratio <= 1.6, (eta_t, eta_l, ratio)

    print("\n=== 8. Simulation validation at low frequency (the plateau is not an artifact) ===")
    for omega in (1e-3, 1e-4):
        sim = simulate(omega, ETA_T, ETA_L, n_periods=10)
        ana = peak_to_peak_analytic(omega, ETA_T, ETA_L)
        print(f"  omega={omega:<8g} sim={sim:.6f}  analytic={ana:.6f}  plateau={plateau:.6f}")
        assert abs(sim - ana) / ana < 0.02, (omega, sim, ana)

    print("\n=== 9. Active [.]_+ projection (forcing amplitude > lambda*) ===")
    lam_star = A_CURV * TAU_BAR / C_SENS**2
    for amp in (0.1, 1.0, 3.0):
        pred_lin = 2.0 * amp * A_CURV / C_SENS**2
        sim_proj = simulate(1e-3, ETA_T, ETA_L, n_periods=10, amp=amp, project=True)
        sim_free = simulate(1e-3, ETA_T, ETA_L, n_periods=10, amp=amp, project=False)
        print(
            f"  A={amp:<5g} linear plateau={pred_lin:.4f} (lambda*={lam_star:g})"
            f"  sim with proj={sim_proj:.4f}  without proj={sim_free:.4f}"
        )

    print("\nOK")


if __name__ == "__main__":
    main()
