from __future__ import annotations

import mujoco

from graspbench.config import HOME_Q
from graspbench.types import JointPositionCommand, Observation, PolicyDecision


class StudentPolicy:
    """Runnable scaffold. The framework runs slow ``act`` calls on one worker."""

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        self.task = task
        self.model = model

        # TODO 1: initialize a hosted VLM/SAM/VLA target-grounding module.
        # TODO 2: initialize a planner or numerical IK backend.
        # TODO 3: initialize explicit state, retry budget, and termination checks.
        # Do not create a per-step executor here. ``grasp-eval`` already calls
        # act() on one asynchronous worker and advances physics at 25 Hz.

    def act(self, observation: Observation) -> PolicyDecision:
        # The starter intentionally holds the safe home configuration so that
        # installation, logging, rendering, and evaluation can be tested before
        # any assignment logic is implemented. Real HTTP calls belong only at
        # state events; local IK/state-machine updates should remain fast.
        return PolicyDecision(
            command=JointPositionCommand(HOME_Q, gripper_opening=1.0),
            stage="todo",
            rationale="Starter policy: hold a safe pose. Implement the assignment pipeline here.",
            target_id=None,
            done=False,
        )
