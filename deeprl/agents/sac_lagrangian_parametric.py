"""Lagrangian SAC on `ParametricCMDP`, with a switchable dual signal.

Separate critics Q_r and Q_c are fit on the replay buffer, so lambda enters
the actor loss only:

    actor:  maximize  Q_r(s, pi(s)) - lambda * Q_c(s, pi(s)) + alpha * H(pi)
    dual:   lambda <- max(0, lambda + eta_lambda * g)

`dual_signal` selects g:

* ``fresh``    raw cost averaged over the last `dual_window` env steps, as in
               deployed Lagrangian codebases. The primal sees the constraint
               through Q_c, fit on stale buffer data, so the two paths see
               different sensitivities (mismatched case).
* ``critic``   E_{a ~ pi} Q_c(s, a) on the sampled batch: the dual consumes
               the same estimate as the primal (coherent case).

Episodes have length 1, so the critic targets are the immediate reward and
cost (no bootstrap, no target networks).

Usage:
    uv run python agents/sac_lagrangian_parametric.py --eps 0.5 --omega 0.02 \\
        --dual-signal fresh --dump-path runs/param/fresh_s1.npz
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
    total_timesteps: int = 60_000
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
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    actor_optim: Literal["adam", "sgd"] = "adam"
    alpha: float = 0.01
    lambda_init: float = 0.7
    lambda_lr: float = 3e-2
    dual_signal: Literal["fresh", "critic"] = "fresh"
    dual_window: int = 10
    dump_path: str = "runs/param/run.npz"


def train(args: Args) -> dict[str, np.ndarray]:
    seed_everything(args.seed, deterministic_torch=True)
    torch.set_num_threads(1)  # seeds run as parallel processes
    env = ParametricCMDP(b0=args.b0, c=args.c, eps=args.eps, omega=args.omega)
    obs, _ = env.reset(seed=args.seed)
    low, high = env.action_space.low, env.action_space.high

    actor = Actor(1, 1, low, high, hidden=args.hidden)
    q_r, q_c = SoftQNetwork(1, 1, args.hidden), SoftQNetwork(1, 1, args.hidden)
    q_optim = torch.optim.Adam([*q_r.parameters(), *q_c.parameters()], lr=args.q_lr)
    opt_cls = torch.optim.Adam if args.actor_optim == "adam" else torch.optim.SGD
    actor_optim = opt_cls(actor.parameters(), lr=args.policy_lr)

    buffer = ReplayBuffer(args.buffer_size, 1, 1)
    rng = np.random.default_rng(args.seed)
    recent_costs: deque[float] = deque(maxlen=args.dual_window)
    lam = args.lambda_init
    s0 = torch.zeros(1, 1)

    n = args.total_timesteps
    log = {k: np.full(n, np.nan) for k in ("lam", "a_mean", "c_true", "c_hat", "g")}
    t0 = time.time()
    for t in range(n):
        if t < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = actor.sample(torch.from_numpy(obs).unsqueeze(0))[0].numpy().flatten()
        next_obs, reward, _, _, info = env.step(action)
        buffer.add(obs, action, reward, info["cost"], next_obs, True)
        recent_costs.append(info["cost"])
        obs, _ = env.reset()
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

        pi_a, log_prob, _ = actor.sample(b["obs"])
        qc_pi = q_c(b["obs"], pi_a)
        actor_loss = (args.alpha * log_prob - q_r(b["obs"], pi_a) + lam * qc_pi).mean()
        actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optim.step()

        g = float(np.mean(recent_costs)) if args.dual_signal == "fresh" else qc_pi.mean().item()
        lam = max(0.0, lam + args.lambda_lr * g)

        # Diagnostics at the deterministic action: policy level and the
        # sensitivity the primal actually sees, dQ_c/da.
        a_mean = actor.sample(s0)[2].detach().requires_grad_(True)
        (c_hat,) = torch.autograd.grad(q_c(s0, a_mean).sum(), a_mean)
        log["a_mean"][t], log["c_hat"][t], log["g"][t] = a_mean.item(), c_hat.item(), g

    out = Path(args.dump_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: np.array(v) for k, v in vars(args).items()}
    np.savez_compressed(out, **log, **meta, sps=np.array(n / (time.time() - t0)))
    return log


if __name__ == "__main__":
    a = tyro.cli(Args)
    log = train(a)
    tail = slice(len(log["lam"]) // 2, None)
    print(
        f"lam={np.nanmean(log['lam'][tail]):.3f}+-{np.nanstd(log['lam'][tail]):.3f} "
        f"a={np.nanmean(log['a_mean'][tail]):.3f} c_hat={np.nanmean(log['c_hat'][tail]):.3f}"
    )
