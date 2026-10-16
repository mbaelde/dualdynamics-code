"""Lagrangian SAC on `ParametricCMDP` with a matched coherent control.

Same learner as `sac_lagrangian_parametric.py` with the fresh dual signal
(raw cost averaged over the last `dual_window` steps). The only switch is
what the cost critic sees:

* ``qc_phase=False``  Q_c(a) is fit on the replay buffer without the
                      modulation phase, so it averages c_t over the buffer:
                      the primal sees an unmodulated sensitivity while the
                      dual sees the modulated one (mismatched, h = eps).
* ``qc_phase=True``   Q_c(phase, a) receives (sin omega t, cos omega t) and is
                      queried at the current phase in the actor loss, so the
                      primal sees the same modulated sensitivity as the dual
                      (coherent, h ~ 0). The actor itself stays phase-blind
                      and the dual path is unchanged, so noise and delay are
                      matched between the two arms.

``dual_freeze_until`` holds lambda at ``lambda_init`` (the saddle value) while
the actor and critics train, so the dual starts with a small kick; comparing
lock-in with and without it separates a linear instability from a large-
amplitude state reached through the start-up transient.

``pid_kp > 0`` turns the dual into a PI-Lagrangian (Stooke et al. 2020): the
integral I follows the plain update and the actor sees lam = [I + pid_kp g]_+.
``pid_kp = 0`` is the plain Lagrangian.

Usage:
    uv run python agents/sac_lagrangian_parametric_coherent.py --eps 0.6 \\
        --omega 0.03 --qc-phase --dump-path runs/param/coh_s1.npz
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from agents.common import Actor, ReplayBuffer, SoftQNetwork
from cmdp_env.envs.parametric_cmdp import ParametricCMDP
from cmdp_env.utils import seed_everything


@dataclass
class Args:
    seed: int = 1
    total_timesteps: int = 20_000
    learning_starts: int = 1_000
    # environment
    b0: float = 0.3
    c: float = 1.0
    eps: float = 0.0
    omega: float = 0.0
    # learner
    buffer_size: int = 5_000
    batch_size: int = 256
    hidden: int = 64
    policy_lr: float = 3e-3
    q_lr: float = 1e-3
    actor_optim: Literal["adam", "sgd"] = "sgd"
    alpha: float = 0.01
    lambda_init: float = 0.7
    lambda_lr: float = 0.1
    dual_window: int = 10
    # Hold lambda at lambda_init until this step so the actor settles near the
    # saddle first: removes the start-up transient that kicks the loop.
    dual_freeze_until: int = 0
    # Proportional gain on the windowed cost (PI-Lagrangian); 0 = plain Lagrangian.
    pid_kp: float = 0.0
    qc_phase: bool = False
    dump_path: str = "runs/param/coherent.npz"


def train(args: Args) -> dict[str, np.ndarray]:
    seed_everything(args.seed, deterministic_torch=True)
    torch.set_num_threads(1)  # seeds run as parallel processes
    env = ParametricCMDP(b0=args.b0, c=args.c, eps=args.eps, omega=args.omega)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)  # warm-up actions; reset(seed) does not seed the space
    low, high = env.action_space.low, env.action_space.high

    def phase(t: int) -> np.ndarray:
        if not args.qc_phase:
            return np.zeros(2, dtype=np.float32)
        return np.array([np.sin(args.omega * t), np.cos(args.omega * t)], dtype=np.float32)

    # Buffer obs holds the phase features; the actor is fed a constant input.
    actor = Actor(2, 1, low, high, hidden=args.hidden)
    q_r, q_c = SoftQNetwork(2, 1, args.hidden), SoftQNetwork(2, 1, args.hidden)
    q_optim = torch.optim.Adam([*q_r.parameters(), *q_c.parameters()], lr=args.q_lr)
    opt_cls = torch.optim.Adam if args.actor_optim == "adam" else torch.optim.SGD
    actor_optim = opt_cls(actor.parameters(), lr=args.policy_lr)

    buffer = ReplayBuffer(args.buffer_size, 2, 1)
    rng = np.random.default_rng(args.seed)
    recent_costs: deque[float] = deque(maxlen=args.dual_window)
    lam = integral = args.lambda_init
    s0 = torch.zeros(1, 2)
    s0_batch = torch.zeros(args.batch_size, 2)

    n = args.total_timesteps
    log = {k: np.full(n, np.nan) for k in ("lam", "a_mean", "c_true", "c_hat", "g")}
    t0 = time.time()
    for t in range(n):
        ph = phase(env.t)
        if t < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = actor.sample(s0)[0].numpy().flatten()
        _, reward, _, _, info = env.step(action)
        buffer.add(ph, action, reward, info["cost"], ph, True)
        recent_costs.append(info["cost"])
        env.reset()
        log["c_true"][t] = info["sensitivity"]
        log["lam"][t] = lam
        if t < args.learning_starts:
            continue

        b = buffer.sample(args.batch_size, rng)
        q_loss = F.mse_loss(q_r(b["obs"], b["actions"]), b["rewards"]) + F.mse_loss(
            q_c(b["obs"], b["actions"]), b["costs"]
        )
        q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        q_optim.step()

        # The primal is evaluated at the phase of the step about to be played.
        ph_now = torch.from_numpy(phase(env.t)).expand(args.batch_size, 2)
        pi_a, log_prob, _ = actor.sample(s0_batch)
        actor_loss = (args.alpha * log_prob - q_r(ph_now, pi_a) + lam * q_c(ph_now, pi_a)).mean()
        actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optim.step()

        g = float(np.mean(recent_costs))
        if t >= args.dual_freeze_until:
            integral = max(0.0, integral + args.lambda_lr * g)
            lam = max(0.0, integral + args.pid_kp * g)

        # Diagnostics: deterministic action and the sensitivity the primal sees.
        a_mean = actor.sample(s0)[2].detach().requires_grad_(True)
        (c_hat,) = torch.autograd.grad(q_c(ph_now[:1], a_mean).sum(), a_mean)
        log["a_mean"][t], log["c_hat"][t], log["g"][t] = a_mean.item(), c_hat.item(), g

    out = Path(args.dump_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: np.array(v) for k, v in vars(args).items()}
    np.savez_compressed(out, **log, **meta, dual_signal=np.array("fresh"), sps=np.array(n / (time.time() - t0)))
    return log


if __name__ == "__main__":
    a = tyro.cli(Args)
    log = train(a)
    tail = slice(len(log["lam"]) // 2, None)
    print(
        f"lam={np.nanmean(log['lam'][tail]):.3f}+-{np.nanstd(log['lam'][tail]):.3f} "
        f"a={np.nanmean(log['a_mean'][tail]):.3f} c_hat={np.nanmean(log['c_hat'][tail]):.3f}"
    )
