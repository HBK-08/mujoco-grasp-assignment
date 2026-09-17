from __future__ import annotations

from typing import Protocol

import mujoco

from .types import Observation, PolicyDecision


class GraspPolicy(Protocol):
    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        """Reset episode state. `task` deliberately excludes the evaluator target label."""

    def act(self, observation: Observation) -> PolicyDecision:
        """Return one structured decision at the policy/control rate."""

