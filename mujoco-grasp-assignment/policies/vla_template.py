from __future__ import annotations

import mujoco
import numpy as np

from graspbench.adapters import CartesianDeltaAdapter
from graspbench.types import CartesianDeltaCommand, Observation, PolicyDecision


class VLATemplate:
    """VLA action-head contract before wrapping with CartesianDeltaAdapter."""

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        self.instruction = task["instruction"]
        # TODO: load a small/local policy or initialize a remote inference client.

    def act(self, observation: Observation) -> PolicyDecision:
        # TODO: preprocess observation.cameras and proprioception exactly as the
        # selected VLA expects; map its action head to meters/radians and [0,1].
        action = np.zeros(7, dtype=np.float64)
        return PolicyDecision(
            command=CartesianDeltaCommand(action[:3], action[3:6], float(action[6])),
            stage="vla_rollout",
            rationale="Zero-action VLA scaffold; replace with model inference.",
        )


class AdaptedVLATemplate(CartesianDeltaAdapter):
    """Runnable evaluator entry point: `policies.vla_template:AdaptedVLATemplate`."""

    def __init__(self) -> None:
        super().__init__(VLATemplate())

