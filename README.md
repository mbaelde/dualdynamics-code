# dualdynamics-code

Code for the paper "Sensitivity mismatch and parametric resonance in primal-dual dynamics" and its extended version "Parametric Resonance in Primal-Dual Learning for Slowly Non-Stationary Constrained MDPs" (M. Baelde): the primal-dual policy/multiplier recursion of a constrained MDP under a slowly time-varying constraint, and the parametric-resonance instability that appears when the primal and dual step sizes see that drift with different sensitivity.

Two independent parts:

```
./        numpy scripts for the analysis of the reduced primal-dual recursion (Floquet exponents, thresholds, tongues)
deeprl/   Lagrangian SAC agents on a constrained bandit with modulated cost sensitivity
```

Section, table and equation labels below are the papers' own (`\label{...}` in the `.tex`). They refer to the extended version; the short version keeps `lem:cancellation`, `thm:coherent`, `thm:threshold`, `eq:threshold`, `tab:floquet`, `tab:borkar` and `fig:tongue` under the same names, and summarizes the deep RL experiment in one paragraph.

## Theory scripts

Pure numpy, Python 3.10+. The scripts import each other flat (run with the repo root as `sys.path[0]`), which is why they sit in one directory rather than a package. Each one runs standalone from the repo root and checks its own instruments with assertions, e.g.:

```
python -m pip install numpy
python borkar_regime_check.py
```

| Paper | Script | Question it answers |
| --- | --- | --- |
| Section `sec:additive`, `prop:floor` | `toy_k1_linear.py` | Additive drift on the constraint: does the dual amplitude diverge as the forcing frequency goes to zero, or does it plateau? |
| Thm `thm:coherent`, Lemma `lem:cancellation` | `toy_k1_parametric.py` | Parametric (multiplicative) drift on the constraint sensitivity: is the coupled primal-dual recursion Floquet-stable under coherent modulation, and does the forcing cancel exactly at the principal tongue? |
| Thm `thm:threshold`, `eq:threshold` | `toy_k1_parametric.py` | Under a primal/dual sensitivity *mismatch*, what is the narrowband instability threshold, and does it match the measured Floquet exponent (Table `tab:floquet`)? |
| `prop:spectral`, `eq:spectral`, `eq:broadband` | `toy_k1_broadband.py` | Does the coherent-cancellation result survive broadband (not single-tone) forcing, and what is the bandwidth-dependent correction (Table `tab:ec_sqrtB`)? |
| Table `tab:borkar` | `borkar_regime_check.py` | Where does the phenomenon sit relative to the standard CMDP two-timescale ordering ($r = \eta_\lambda/\eta_\theta \ll 1$)? Threshold maximized over the forcing frequency, for the full grid of $r$ values in the table. |
| Table `tab:ec_etath` | `toy_k1_broadband.py` | Cancellation of the primal step size at fixed physical bandwidth. |
| `rem:multiK` | `toy_phase_and_multiK.py` | Two extensions: phase mismatch between the two modulation channels (not just amplitude mismatch), and $K>1$ constraints with a shared waveform. |
| Section `sec:exp_prescription`, Table `tab:ec_recursion` (projected regime) | `projected_regime_metrics.py` | Under the $[\cdot]_+$ projection, is coherent vs. mismatched modulation distinguishable on operational metrics (integrated constraint violation, cost regret) rather than on the raw clipped-fraction figure alone? |
| Figures `fig:tongue`, `fig:bandwidth` | `make_figure_data.py` | Produces the two pgfplots data files: the measured tongue at the reference operating point (Table `tab:opoint`), and the bandwidth-normalized threshold curve against `eq:spectral`. |

## deeprl/

Section `sec:exp_deeprl`, Figure `fig:deeprl`, Table `tab:deeprl_cost`. Python 3.12+. Install with `pip` (or `uv`) from inside `deeprl/`:

```
cd deeprl
python -m venv .venv && . .venv/bin/activate
pip install torch numpy gymnasium tyro scipy     # same list as pyproject.toml
export PYTHONPATH=.                              # scripts import `agents` and `cmdp_env` from this folder
```

| File | Role |
| --- | --- |
| `cmdp_env/envs/parametric_cmdp.py` | One-step constrained bandit, action `a` in [0,1], reward `a - a^2/2`, cost `c_t a - b0`, `c_t = 1 + eps sin(omega t)`; the agent does not observe the phase |
| `agents/sac_lagrangian_parametric.py` | Lagrangian SAC with separate reward and cost critics, switchable dual signal (`fresh` or `critic`) |
| `agents/sac_lagrangian_parametric_coherent.py` | The agents of `sec:exp_deeprl`: Lagrangian (default), coherent (`--qc-phase`, phase fed to the cost critic only) and PI-Lagrangian (`--pid-kp 1.0`) |
| `agents/analyze_parametric_lockin.py` | Amplitude `A_{1/2}` of the multiplier at `omega/2`, demodulated over the second half of each run, per run group |
| `agents/floquet_identified_loop.py` | Identification of the effective primal gains on unmodulated runs and Floquet threshold of the identified loop (also prints the step sizes for a grid of timescale ratios `r` with `--design`) |
| `agents/fit_var_parametric.py`, `agents/analyze_parametric.py` | Auxiliary diagnostics (VAR(1) fit of `(a, lambda)`; spectral sub-harmonic power, which expects a grid of `fresh` and `critic` runs at several `eps`) |

Defaults of the coherent-agent script match the paper: actor SGD with learning rate `3e-3`, dual step `eta_lambda = 0.1`, dual window `W = 10`, replay buffer `5000`, `20000` steps, `b0 = 0.3`. Every run writes a compressed `.npz` (`lam`, `a_mean`, `c_hat`, `g`, and the arguments).

### Reproducing the deep RL experiment

Run files must be named `<GROUP>_s<seed>.npz`; `analyze_parametric_lockin.py` groups them by the part before `_s`. Each run takes a few minutes on one CPU thread; seeds are independent and can be launched in parallel. Example, one run (Lagrangian agent at `r = 100`, `eps = 0.4`, `omega = 0.026`, seed 1):

```
python agents/sac_lagrangian_parametric_coherent.py --policy-lr 0.002277 --lambda-lr 0.1317 \
    --eps 0.4 --omega 0.026 --seed 1 --dump-path runs/LAG_r100_e0.4_w0.026_s1.npz
```

1. Identification (unmodulated runs, base step sizes `3e-3` and `0.1`, seeds 1 to 5), then gains, effective `r`, predicted thresholds, and (with `--design`, pure-Python Floquet bisections, slow: more than an hour on one core) the design step sizes and thresholds per `r`:

   ```
   for s in 1 2 3 4 5; do python agents/sac_lagrangian_parametric_coherent.py --seed $s --dump-path runs/ident/A_e0_s$s.npz; done
   python agents/floquet_identified_loop.py runs/ident
   python agents/floquet_identified_loop.py runs/ident --design
   ```

2. Timescale grid at fixed `eta_theta * eta_lambda` (step sizes given by `--design`; the values used are `r = 100`: `--policy-lr 0.002277 --lambda-lr 0.1317`; `r = 10`: `--policy-lr 0.007201 --lambda-lr 0.04166`; `r = 1`: `--policy-lr 0.02277 --lambda-lr 0.01317`). Frequencies `omega` in `{0.018, 0.022, 0.026, 0.030, 0.036}`:
   - `r = 100`: `--eps 0.4` with the default `--b0 0.3`.
   - `r = 10` and `r = 1`: `--eps 0.8 --b0 0.15` (the instantaneous saddle stays inside the action interval only for `b0 < 1 - eps`).
   - Lagrangian agent: no extra flag, seeds 1 to 10 (group names such as `LAG_r100_e0.4_w0.026`).
   - Coherent agent: add `--qc-phase`, seeds 1 to 5.
   - PI-Lagrangian agent: add `--pid-kp 1.0`, seeds 1 to 10.
   - Unmodulated reference: omit `--eps` and `--omega`.

3. Lock-in amplitude (Figure `fig:deeprl`): `python agents/analyze_parametric_lockin.py runs/<dir> --omega-ref 0.026`. A run counts as locked when `A_{1/2} > 0.4`; unmodulated runs are evaluated at `omega_ref / 2`.

4. Table `tab:deeprl_cost`: computed from the logged arrays of the same runs. The windowed constraint value is the mean of `g` over the second half of a run, and the return of the deterministic action is `a - a^2/2` evaluated at the mean of `a_mean` over the same half.

The identification, Floquet and analysis scripts need numpy and scipy only; only the agent scripts need torch.
