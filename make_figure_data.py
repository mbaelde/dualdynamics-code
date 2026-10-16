"""Data for the two figures of the paper, written as pgfplots-readable tables.

Figure `fig:tongue` (tongue.dat) is a fresh measurement: it reuses the instruments of
toy_k1_parametric (step_matrix / monodromy / floquet_exponent) at the reference
operating point of Table `tab:opoint`, so the value at omega / 2 omega_nat = 1 reproduces the
rows of Table `tab:floquet` by construction.

Figure `fig:bandwidth` (bandwidth_*.dat) is a re-presentation of measurements already reported in
the paper, not a new run: the absolute thresholds come from Table `tab:ec_sqrtB` (toy_k1_broadband,
section 4 of its main) and from Table `tab:ec_etath` (section 6), and this script only forms the
dimensionless ratio threshold / single-line threshold. Re-running the Lyapunov
estimator here would consume a different stream of the shared generator and land a
few per mille away from the published numbers, which is why the figure reads them
instead of recomputing them.

Run from the project root:  py -3.14 make_figure_data.py
"""

from __future__ import annotations

import math
from pathlib import Path

from toy_k1_parametric import floquet_exponent, natural_mode

OUT = Path(__file__).parent / "Rapport" / "figures"

# Reference operating point of Table `tab:opoint`, hardcoded in the main of toy_k1_parametric.
ETA_T, ETA_L = 1e-3, 1e-1

# Figure `fig:tongue`: mismatch depths, pure dual modulation (eps_theta = 0, eps_lam = h).
# 0.15, 0.20, 0.30 and 0.90 are rows of Table `tab:floquet`; 0.10 sits below the threshold 4 zeta,
# 0.50 well above it. The figure plots the first five; 0.90 is written out as well
# because its peak position quantifies the tilt of the tongue at large depth.
DEPTHS = (0.10, 0.15, 0.20, 0.30, 0.50, 0.90)

# Modulation periods in steps. Stepping over integer periods rather than over a grid
# of frequencies avoids the quantization artefact of rounding 2 pi / omega: every
# point is an exactly realizable forcing, and the ratio is recovered from the period.
# The upper end reaches the secondary tongue at omega = omega_nat, ratio 1/2.
N_MIN, N_MAX, N_STEP = 196, 785, 1


def tongue() -> None:
    w_nat, sigma = natural_mode(ETA_T, ETA_L)
    rows = []
    for n in range(N_MIN, N_MAX, N_STEP):
        ratio = (2.0 * math.pi / n) / (2.0 * w_nat)
        exps = [1e4 * floquet_exponent(n, ETA_T, ETA_L, eps_lam=h) for h in DEPTHS]
        rows.append((ratio, exps))
    rows.sort(key=lambda r: r[0])

    header = "ratio " + " ".join(f"h{h:g}".replace(".", "") for h in DEPTHS)
    lines = [header]
    lines += [f"{r:.6f} " + " ".join(f"{e:.5f}" for e in exps) for r, exps in rows]
    (OUT / "tongue.dat").write_text("\n".join(lines) + "\n", encoding="ascii")

    # Controls: the instrument must return the published operating point, and the peak
    # of each tongue must reproduce the corresponding row of Table `tab:floquet` to a relative 1%.
    assert abs(w_nat - 0.00999216) < 1e-7, w_nat
    assert abs(sigma - 4.502026e-4) < 1e-9, sigma
    for h, expected in ((0.20, 0.497), (0.30, 2.988), (0.90, 17.55)):
        got = max(r[1][DEPTHS.index(h)] for r in rows)
        assert abs(got - expected) < 0.01 * abs(expected), (h, got, expected)

    print(f"tongue.dat: {len(rows)} periods, omega_nat = {w_nat:.8f}, sigma = {sigma:.6e}")
    print(f"  unmodulated exponent -sigma = {-1e4 * sigma:.5f} in units of 1e-4")
    for k, h in enumerate(DEPTHS):
        best = max(rows, key=lambda r: r[1][k])
        low = max((r for r in rows if r[0] < 0.75), key=lambda r: r[1][k])
        print(
            f"  h = {h:.2f}: peak {best[1][k]:+.5f}e-4 at ratio {best[0]:.4f}"
            f" | secondary {low[1][k]:+.5f}e-4 at ratio {low[0]:.4f}"
        )


# Figure `fig:bandwidth`, from the tables. eps_0 is the measured single-line threshold at the same
# operating point, so each series is normalized by its own narrowband threshold.
SQRTB = {  # Table `tab:ec_sqrtB`: eta_th = 1e-3, sigma = 4.502e-4, single line 0.1277
    0.5: 0.1382,
    2.0: 0.1590,
    8.0: 0.2310,
    32.0: 0.4262,
}
SQRTB_EPS0 = 0.1277
ETATH = [  # Table `tab:ec_etath`: fixed physical bandwidth 4.5e-3 rad/step, two step sizes
    (10.0, 0.2570, 0.1277, "1e-3"),
    (100.0, 0.2393, 0.04222, "1e-4"),
]
KAPPA = 0.309


def bandwidth() -> None:
    sweep = ["boverS ratio"] + [f"{b:g} {thr / SQRTB_EPS0:.4f}" for b, thr in sorted(SQRTB.items())]
    fixed = ["boverS ratio"] + [f"{b:g} {thr / eps0:.4f}" for b, thr, eps0, _ in ETATH]
    (OUT / "bandwidth_sweep.dat").write_text("\n".join(sweep) + "\n", encoding="ascii")
    (OUT / "bandwidth_fixed.dat").write_text("\n".join(fixed) + "\n", encoding="ascii")

    print("bandwidth_sweep.dat / bandwidth_fixed.dat:")
    for b, thr in sorted(SQRTB.items()):
        r = thr / SQRTB_EPS0
        print(f"  B/sigma = {b:5g}: measured {r:.4f}, law {math.sqrt(1 + KAPPA * b):.4f}")
    for b, thr, eps0, tag in ETATH:
        r = thr / eps0
        print(f"  B/sigma = {b:5g}: measured {r:.4f}, law {math.sqrt(1 + KAPPA * b):.4f} ({tag})")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tongue()
    bandwidth()


if __name__ == "__main__":
    main()
