from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class IKResult:
    joint_position: np.ndarray
    converged: bool
    position_error: float
    orientation_error: float
    iterations: int


class DampedLeastSquaresIK:
    """CPU-only 6D numerical IK using MuJoCo's analytic site Jacobian."""

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "grasp_site",
        damping: float = 1e-3,
        step_size: float = 0.65,
        orientation_weight: float = 0.35,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Unknown site: {site_name}")
        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        )
        self.qpos_ids = np.array([model.jnt_qposadr[jid] for jid in self.joint_ids])
        self.dof_ids = np.array([model.jnt_dofadr[jid] for jid in self.joint_ids])
        self.lower = model.jnt_range[self.joint_ids, 0].copy()
        self.upper = model.jnt_range[self.joint_ids, 1].copy()
        self.damping = float(damping)
        self.step_size = float(step_size)
        self.orientation_weight = float(orientation_weight)

    def solve(
        self,
        initial_qpos: np.ndarray,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        *,
        max_iterations: int = 180,
        position_tolerance: float = 1.5e-3,
        orientation_tolerance: float = 2.5e-2,
        rest_qpos: np.ndarray | None = None,
    ) -> IKResult:
        initial_qpos = np.asarray(initial_qpos, dtype=np.float64)
        target_position = np.asarray(target_position, dtype=np.float64)
        target_quaternion = np.asarray(target_quaternion, dtype=np.float64)
        if initial_qpos.shape == (7,):
            self.data.qpos[self.qpos_ids] = initial_qpos
        elif initial_qpos.shape == (self.model.nq,):
            self.data.qpos[:] = initial_qpos
        else:
            raise ValueError("initial_qpos must contain 7 arm joints or the complete model qpos")

        target_quaternion = target_quaternion / np.linalg.norm(target_quaternion)
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        target_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(target_rotation, target_quaternion)
        target_rotation = target_rotation.reshape(3, 3)
        last_pos_error = np.inf
        last_rot_error = np.inf

        for iteration in range(1, max_iterations + 1):
            mujoco.mj_forward(self.model, self.data)
            position_error = target_position - self.data.site_xpos[self.site_id]
            current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            rotation_error = _rotation_vector(target_rotation @ current_rotation.T)
            last_pos_error = float(np.linalg.norm(position_error))
            last_rot_error = float(np.linalg.norm(rotation_error))
            if last_pos_error < position_tolerance and last_rot_error < orientation_tolerance:
                return IKResult(
                    self.data.qpos[self.qpos_ids].copy(),
                    True,
                    last_pos_error,
                    last_rot_error,
                    iteration,
                )

            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.site_id)
            jacobian = np.vstack(
                [jac_pos[:, self.dof_ids], self.orientation_weight * jac_rot[:, self.dof_ids]]
            )
            error = np.concatenate([position_error, self.orientation_weight * rotation_error])
            normal = jacobian @ jacobian.T + (self.damping**2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(normal, error)

            if rest_qpos is not None:
                rest = np.asarray(rest_qpos, dtype=np.float64)
                projector = np.eye(7) - np.linalg.pinv(jacobian) @ jacobian
                delta += 0.015 * projector @ (rest - self.data.qpos[self.qpos_ids])

            max_abs = float(np.max(np.abs(delta)))
            if max_abs > 0.22:
                delta *= 0.22 / max_abs
            q = self.data.qpos[self.qpos_ids] + self.step_size * delta
            self.data.qpos[self.qpos_ids] = np.clip(q, self.lower + 1e-4, self.upper - 1e-4)

        return IKResult(
            self.data.qpos[self.qpos_ids].copy(),
            False,
            last_pos_error,
            last_rot_error,
            max_iterations,
        )


def move_toward(current: np.ndarray, target: np.ndarray, max_step: float = 0.045) -> np.ndarray:
    """Rate-limit a joint-space command without an additional trajectory dependency."""
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = np.clip(target - current, -max_step, max_step)
    return current + delta


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the world-frame logarithm map of a 3x3 rotation matrix."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1e-7:
        return 0.5 * skew
    if np.pi - angle < 1e-5:
        # Near pi, the skew term is ill-conditioned; recover an unsigned axis
        # from the diagonal and choose signs using the off-diagonal terms.
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        axis[0] = np.copysign(axis[0], rotation[2, 1] - rotation[1, 2] or 1.0)
        axis[1] = np.copysign(axis[1], rotation[0, 2] - rotation[2, 0] or 1.0)
        axis[2] = np.copysign(axis[2], rotation[1, 0] - rotation[0, 1] or 1.0)
        norm = np.linalg.norm(axis)
        return angle * axis / max(norm, 1e-12)
    return angle * skew / (2.0 * np.sin(angle))
