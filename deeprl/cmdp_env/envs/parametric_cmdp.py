"""Single-constraint bandit CMDP with a sinusoidally modulated cost sensitivity.

Parametric counterpart of `SinusoidalCMDP`: the non-stationarity moves the
*sensitivity* of the constraint to the action, not its level,

    r(a)    = a - 0.5 a^2,
    cost(a) = c_t a - b0,   c_t = c (1 + eps sin(omega t)),

where t is a global step counter that persists across resets, so the
modulation runs on the learning timescale. Episodes have length 1 and the
observation is constant: the agent cannot see the phase and faces a slowly
drifting constrained problem. Frozen-time saddle: a* = b0 / c_t,
lambda* = (1 - a*) / c_t.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ParametricCMDP(gym.Env[np.ndarray, np.ndarray]):
    def __init__(
        self,
        b0: float = 0.3,
        c: float = 1.0,
        eps: float = 0.0,
        omega: float = 0.0,
        phase_offset: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < b0 < c * (1.0 - abs(eps)):
            raise ValueError(f"need 0 < b0 < c (1 - |eps|) for an interior saddle; got {b0}")
        self.b0, self.c, self.eps, self.omega = float(b0), float(c), float(eps), float(omega)
        self.phase_offset = float(phase_offset)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.t = 0  # global step, never reset

    def sensitivity(self, t: int) -> float:
        return self.c * (1.0 + self.eps * float(np.sin(self.omega * t + self.phase_offset)))

    def _obs(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return self._obs(), {}

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = float(np.clip(np.asarray(action, dtype=np.float64).item(), 0.0, 1.0))
        c_t = self.sensitivity(self.t)
        self.t += 1
        cost = c_t * a - self.b0
        return self._obs(), a - 0.5 * a * a, True, False, {"cost": cost, "sensitivity": c_t}


if __name__ == "__main__":
    env = ParametricCMDP(b0=0.3, eps=0.5, omega=0.1)
    env.reset(seed=0)
    cs = [env.step(np.array([0.3]))[4]["sensitivity"] for _ in range(1000)]
    assert abs(np.mean(cs) - 1.0) < 0.02 and abs(max(cs) - 1.5) < 1e-3, (np.mean(cs), max(cs))
    env.reset()
    assert env.t == 1000, "global clock must survive reset"
    _, r, term, _, info = env.step(np.array([0.5]))
    assert term and abs(r - 0.375) < 1e-12
    print("ok")
