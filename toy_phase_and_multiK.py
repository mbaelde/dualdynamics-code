"""Two extensions of the parametric threshold from `toy_k1_parametric.py`, to
check before writing them into the paper.

L2 -- PHASE MISMATCH. The paper states the threshold on |eps_lam - eps_theta|,
i.e. on an AMPLITUDE mismatch, assuming both modulations are in phase. If the
two channels carry different phases,

    c_theta(t) = c (1 + eps_theta sin(omega t + phi_theta))
    c_lam(t)   = c (1 + eps_lam   sin(omega t + phi_lam))

the first-order calculation is unchanged in structure: the stiffness forcing
coefficient equals

    eps_theta sin(omega t + phi_theta) (omega_0^2 - omega^2/2)
  + eps_lam   sin(omega t + phi_lam)    omega_0^2

and at the principal tongue omega = 2 omega_0 the first bracket equals
-omega_0^2, so the forcing is omega_0^2 Im{ (eps_lam e^{i phi_lam} - eps_theta e^{i phi_theta}) e^{i omega t} },
of amplitude omega_0^2 |eps_lam - eps_theta e^{i dphi}| with dphi = phi_theta - phi_lam.
The width of a Mathieu tongue depends only on the modulus of the forcing (the
phase is only a choice of time origin), hence the PREDICTION

    |eps_lam - eps_theta e^{i dphi}| > 4 zeta,
    |eps_lam - eps_theta e^{i dphi}|^2 = eps_lam^2 + eps_theta^2 - 2 eps_lam eps_theta cos(dphi)

Consequences to check: coherent cancellation requires equality of BOTH
amplitudes AND phases; in quadrature the mismatch equals sqrt(eps_lam^2 + eps_theta^2)
and never cancels; in phase opposition it equals eps_lam + eps_theta, the
worst possible case.

L3 -- K CONSTRAINTS. With theta scalar and lambda in R^K, under a modulation
waveform COMMON to the K constraints,

    theta_{t+1} = theta_t - eta (a theta_t + (1 + eps_theta s_t) <c, lambda_t>)
    lambda_{t+1} = lambda_t + alpha (1 + eps_lam s_t) c theta_t

the dual update is entirely carried by the direction c, so u_t :=
<c, lambda_t> closes on itself and the homogeneous system reduces EXACTLY to
the scalar case with c^2 -> ||c||^2. The K-1 directions of lambda orthogonal
to c are Floquet-neutral (multiplier 1). The prediction is therefore

    omega_0 = ||c|| sqrt(eta alpha),   zeta = (a / 2||c||) sqrt(eta / alpha),
    mismatch threshold = 4 zeta, which decreases as ||c||^{-1}, i.e. as
    K^{-1/2} for K constraints of comparable sensitivities.

The constant offsets b_k only enter the inhomogeneous part, i.e. the already-
bounded additive channel, and cannot change the sign of an exponent.

What is NOT covered here and must stay open: different depths eps_lam,k or
different frequencies from one constraint to another, which make the dual
direction time-varying and break the reduction. Test 4 measures this.

Run: python toy_phase_and_multiK.py
"""

from __future__ import annotations

import numpy as np

A_CURV = 1.0


def step_matrix_k(eta: float, alpha: float, a: float, c_hat: np.ndarray, c_meas: np.ndarray) -> np.ndarray:
    """Step matrix of the tangent system (theta, lambda_1..lambda_K).

    c_hat  : sensitivity seen by the PRIMAL (constraint estimate)
    c_meas : sensitivity seen by the DUAL   (constraint measurement)
    """
    k = c_hat.size
    m = np.eye(k + 1)
    m[0, 0] = 1.0 - eta * a
    m[0, 1:] = -eta * c_hat
    m[1:, 0] = alpha * c_meas
    return m


def monodromy_k(
    n_steps: int,
    eta: float,
    alpha: float,
    c: np.ndarray,
    eps_theta: np.ndarray,
    eps_lam: np.ndarray,
    phi_theta: float = 0.0,
    phi_lam: float = 0.0,
    a: float = A_CURV,
    freq_mult: np.ndarray | None = None,
) -> np.ndarray:
    """Monodromy over one period of n_steps steps. freq_mult lets each constraint
    carry an integer multiple of the base frequency (test 4)."""
    if freq_mult is None:
        freq_mult = np.ones_like(c)
    phi = np.eye(c.size + 1)
    for t in range(n_steps):
        ph = 2.0 * np.pi * t / n_steps
        s_th = np.sin(freq_mult * ph + phi_theta)
        s_la = np.sin(freq_mult * ph + phi_lam)
        phi = step_matrix_k(eta, alpha, a, c * (1.0 + eps_theta * s_th), c * (1.0 + eps_lam * s_la)) @ phi
    return phi


def floquet_k(n_steps: int, eta: float, alpha: float, c: np.ndarray, **kw) -> float:
    """Floquet exponent per step, ignoring the Floquet-neutral directions.

    Under a common waveform, the K-1 directions of lambda orthogonal to c are
    left strictly invariant by the recursion, so the monodromy carries K-1
    eigenvalues exactly equal to 1. Keeping them would cap the exponent at 0
    and turn the slightest rounding error into a false instability. They are
    marginal and non-growing: the rest of the spectrum decides stability.
    """
    ev = np.linalg.eigvals(monodromy_k(n_steps, eta, alpha, c, **kw))
    ev = ev[abs(ev - 1.0) > 1e-9]
    if ev.size == 0:
        return 0.0
    return float(np.log(max(abs(ev))) / n_steps)


def natural_mode_k(eta: float, alpha: float, c: np.ndarray, a: float = A_CURV) -> tuple[float, float]:
    """(omega_0, sigma) read off the spectrum of the unmodulated system, ignoring
    eigenvalues exactly equal to 1 (neutral directions orthogonal to c)."""
    ev = np.linalg.eigvals(step_matrix_k(eta, alpha, a, c, c))
    ev = ev[abs(ev - 1.0) > 1e-12]
    k = int(np.argmax(abs(ev)))
    return float(abs(np.angle(ev[k]))), float(-np.log(abs(ev[k])))


def n_for_omega(omega: float) -> int:
    return max(8, int(round(2.0 * np.pi / omega)))


def scan_floquet(eta: float, alpha: float, c: np.ndarray, ratios: np.ndarray | None = None, **kw) -> float:
    """Maximal Floquet exponent over a sweep of omega around 2 omega_0."""
    w0, _ = natural_mode_k(eta, alpha, c)
    if ratios is None:
        ratios = np.linspace(0.90, 1.10, 41)
    return max(floquet_k(n_for_omega(r * 2.0 * w0), eta, alpha, c, **kw) for r in ratios)


def threshold_on(fn, hi: float = 3.0, n_bisect: int = 50) -> float:
    """Smallest value of the scalar parameter making fn(x) > 0, by bisection."""
    if fn(hi) <= 0.0:
        return float("nan")
    lo = 0.0
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if fn(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main() -> None:
    eta, alpha = 1e-3, 1e-1

    print("=== 0. Control: K=1 without modulation, the machinery recovers the scalar case ===")
    c1 = np.array([1.0])
    w0, sig = natural_mode_k(eta, alpha, c1)
    print(f"  omega_0={w0:.6g}  (prediction ||c|| sqrt(eta alpha)={np.sqrt(eta*alpha):.6g})")
    print(f"  sigma={sig:.6g}  zeta={sig/w0:.6g}  (prediction (a/2||c||) sqrt(eta/alpha)={0.5*np.sqrt(eta/alpha):.6g})")
    e0 = floquet_k(128, eta, alpha, c1, eps_theta=np.zeros(1), eps_lam=np.zeros(1))
    assert abs(e0 + sig) < 1e-12, (e0, sig)
    print(f"  Floquet at eps=0: {e0:+.4e} = -sigma  OK")

    # ------------------------------------------------------------------ L2
    print("\n=== 1. L2: mismatch threshold vs. the predicted COMPLEX mismatch ===")
    print("  At fixed (eps_theta, eps_lam) shape and phase offset, we look for the critical")
    print("  scale factor s*, then compare s* |eps_lam - eps_theta e^{i dphi}| to the threshold 4 zeta.")
    zeta = sig / w0
    print(f"  predicted threshold on the modulus: 4 zeta = {4*zeta:.5g}")
    print(f"  {'eps_theta':>10} {'eps_lam':>8} {'dphi/pi':>8} {'s*':>8} {'|mismatch|':>12} {'/4zeta':>8}")
    ok = True
    for e_th, e_la in ((0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.0, 0.5)):
        for dphi_over_pi in (0.0, 0.25, 0.5, 0.75, 1.0):
            dphi = dphi_over_pi * np.pi
            unit = float(np.hypot(e_la - e_th * np.cos(dphi), e_th * np.sin(dphi)))
            if unit < 1e-12:  # mismatch identically zero: nothing to measure a threshold on
                continue

            def unstable(s: float, e_th=e_th, e_la=e_la, dphi=dphi) -> float:
                return scan_floquet(
                    eta,
                    alpha,
                    c1,
                    eps_theta=np.array([s * e_th]),
                    eps_lam=np.array([s * e_la]),
                    phi_theta=dphi,
                    phi_lam=0.0,
                )

            s = threshold_on(unstable)
            mod = s * unit
            ratio = mod / (4.0 * zeta)
            print(
                f"  {e_th:>10g} {e_la:>8g} {dphi_over_pi:>8g} {s:>8.4g} {mod:>12.5g} {ratio:>8.3f}"
            )
            ok &= 0.85 <= ratio <= 1.15
    print(f"  -> the modulus of the complex mismatch predicts the threshold: {'OK' if ok else 'FAILED'}")
    assert ok

    print("\n  Corollary tested separately: coherent cancellation requires phase AND amplitude.")
    for dphi_over_pi in (0.0, 0.5, 1.0):
        e = scan_floquet(
            eta,
            alpha,
            c1,
            eps_theta=np.array([0.5]),
            eps_lam=np.array([0.5]),
            phi_theta=dphi_over_pi * np.pi,
            phi_lam=0.0,
        )
        pred = 0.5 * np.hypot(1 - np.cos(dphi_over_pi * np.pi), np.sin(dphi_over_pi * np.pi))
        print(
            f"  eps_theta=eps_lam=0.5, dphi={dphi_over_pi:g}pi: |mismatch|={pred:.4g}"
            f" ({'>' if pred > 4*zeta else '<'} 4zeta={4*zeta:.4g})  Floquet={e:+.4e}"
            f"  {'UNSTABLE' if e > 0 else 'stable'}"
        )

    # ------------------------------------------------------------------ L3
    print("\n=== 2. L3: exact reduction to K constraints, common waveform ===")
    rng = np.random.default_rng(20260822)
    print(f"  {'K':>3} {'||c||':>8} {'Floquet (K)':>14} {'scalar Floquet ||c||':>23} {'gap':>10}")
    for k in (1, 2, 3, 5):
        c = np.abs(rng.normal(1.0, 0.3, size=k))
        nrm = float(np.linalg.norm(c))
        kw = dict(eps_theta=np.full(k, 0.1), eps_lam=np.full(k, 0.4))
        n = n_for_omega(2.0 * natural_mode_k(eta, alpha, c)[0])
        e_k = floquet_k(n, eta, alpha, c, **kw)
        e_s = floquet_k(
            n, eta, alpha, np.array([nrm]), eps_theta=np.full(1, 0.1), eps_lam=np.full(1, 0.4)
        )
        print(f"  {k:>3} {nrm:>8.5g} {e_k:>+14.6e} {e_s:>+23.6e} {abs(e_k-e_s):>10.2e}")
        assert abs(e_k - e_s) < 1e-12, (k, e_k, e_s)
    print("  -> equality to machine precision: the reduction is exact, not approximate.")

    print("\n  Neutral directions: the unmodulated spectrum must contain K-1 eigenvalues equal to 1.")
    for k in (2, 3, 5):
        c = np.abs(rng.normal(1.0, 0.3, size=k))
        ev = np.linalg.eigvals(step_matrix_k(eta, alpha, A_CURV, c, c))
        n_one = int(np.sum(abs(ev - 1.0) < 1e-12))
        print(f"  K={k}: {n_one} eigenvalues at 1 (expected {k-1})")
        assert n_one == k - 1, (k, ev)

    print("\n=== 3. L3: tolerance to mismatch decreases as K^{-1/2} ===")
    print("  K constraints of unit sensitivity, so ||c|| = sqrt(K).")
    print(f"  {'K':>3} {'||c||':>8} {'measured threshold':>19} {'predicted 4 zeta':>16} {'ratio':>7}")
    for k in (1, 2, 4, 8):
        c = np.ones(k)
        w_, s_ = natural_mode_k(eta, alpha, c)
        pred = 4.0 * s_ / w_

        def unstable(x: float, c=c, k=k) -> float:
            return scan_floquet(eta, alpha, c, eps_theta=np.zeros(k), eps_lam=np.full(k, x))

        thr = threshold_on(unstable)
        print(f"  {k:>3} {np.sqrt(k):>8.4g} {thr:>19.5g} {pred:>16.5g} {thr/pred:>7.3f}")
        assert 0.85 <= thr / pred <= 1.15, (k, thr, pred)
    print("  -> adding active constraints weakens the loop: tolerance follows 1/||c||.")

    print("\n=== 4. What is NOT covered: heterogeneous waveforms ===")
    print("  (a) different depths eps_lam,k from one constraint to another, K=2")
    c = np.ones(2)
    w_, s_ = natural_mode_k(eta, alpha, c)
    pred = 4.0 * s_ / w_
    print(f"  homogeneous threshold measured in test 3: {pred:.5g} (predicted)")
    print(f"  {'eps_lam':>16} {'mean':>9} {'Floquet':>13} {'verdict':>10}")
    for pair in ((0.2, 0.2), (0.1, 0.3), (0.0, 0.4), (0.3, 0.5), (0.0, 0.8)):
        e = scan_floquet(eta, alpha, c, eps_theta=np.zeros(2), eps_lam=np.array(pair))
        print(f"  {str(pair):>16} {np.mean(pair):>9.3g} {e:>+13.4e} {'UNSTABLE' if e > 0 else 'stable':>10}")
    print("  (b) different frequencies from one constraint to another (integer multiples), K=2")
    print(f"  {'freq_mult':>16} {'Floquet':>13} {'verdict':>10}")
    for fm in ((1.0, 1.0), (1.0, 2.0), (1.0, 3.0)):
        e = scan_floquet(
            eta,
            alpha,
            c,
            eps_theta=np.zeros(2),
            eps_lam=np.full(2, 0.3),
            freq_mult=np.array(fm),
        )
        print(f"  {str(fm):>16} {e:>+13.4e} {'UNSTABLE' if e > 0 else 'stable':>10}")

    print("\nOK")


if __name__ == "__main__":
    main()
