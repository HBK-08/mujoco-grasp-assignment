from __future__ import annotations

import mujoco
import numpy as np

from .ik import DampedLeastSquaresIK
from .types import CartesianDeltaCommand, JointPositionCommand, Observation, PolicyDecision


class CartesianDeltaAdapter:
    """Adapt a VLA-style 7D delta action policy to the benchmark's joint controller."""

    def __init__(self, policy, *, max_translation: float = 0.035, max_rotation: float = 0.15):
        self.policy = policy
        self.max_translation = float(max_translation)
        self.max_rotation = float(max_rotation)
        self.ik: DampedLeastSquaresIK | None = None

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        self.policy.reset(task, model)
        self.ik = DampedLeastSquaresIK(model)

    def act(self, observation: Observation) -> PolicyDecision:
        if self.ik is None:
            raise RuntimeError("Call reset before act")
        decision = self.policy.act(observation)
        if isinstance(decision.command, JointPositionCommand):
            return decision
        command: CartesianDeltaCommand = decision.command
        translation = np.clip(command.translation, -self.max_translation, self.max_translation)
        rotation = np.clip(command.rotation_vector, -self.max_rotation, self.max_rotation)
        target_pos = observation.ee_position + translation

        delta_quat = np.empty(4)
        angle = float(np.linalg.norm(rotation))
        if angle < 1e-10:
            delta_quat[:] = (1.0, 0.0, 0.0, 0.0)
        else:
            mujoco.mju_axisAngle2Quat(delta_quat, rotation / angle, angle)
        target_quat = np.empty(4)
        mujoco.mju_mulQuat(target_quat, delta_quat, observation.ee_quaternion)
        result = self.ik.solve(
            observation.joint_position,
            target_pos,
            target_quat,
            max_iterations=80,
        )
        decision.command = JointPositionCommand(result.joint_position, command.gripper_opening)
        decision.debug = {**decision.debug, "adapter_ik_converged": result.converged}
        return decision
