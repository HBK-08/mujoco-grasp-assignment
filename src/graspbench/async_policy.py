"""Run slow policy decisions without pausing the physics control loop.

The policy API remains deliberately simple and synchronous: students implement
``reset()`` and ``act(observation)``.  This driver invokes at most one ``act``
call at a time on a worker thread.  While a VLM, SAM service, or remote planner
is pending, the simulator continues at the requested control frequency using a
short action horizon and then a local joint-position hold.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from time import perf_counter
from typing import Any

import mujoco
import numpy as np

from .types import (
    CameraObservation,
    JointPositionCommand,
    Observation,
    PolicyDecision,
)


class AsyncPolicyDriver:
    """Single-worker policy executor with a bounded safe-command horizon.

    A returned command may be reused for only ``max_command_hold_steps``.
    Afterwards the arm holds its *current* joint configuration, preserving the
    latest gripper state.  Completed decisions are rejected if the robot has
    moved too far from the observation on which the decision was computed.
    """

    def __init__(
        self,
        policy: Any,
        *,
        decision_interval_steps: int = 1,
        max_command_hold_steps: int = 8,
        max_joint_drift_rad: float = 0.15,
        max_ee_drift_m: float = 0.025,
    ) -> None:
        if decision_interval_steps < 1:
            raise ValueError("decision_interval_steps must be positive")
        if max_command_hold_steps < 0:
            raise ValueError("max_command_hold_steps cannot be negative")
        self.policy = policy
        self.decision_interval_steps = int(decision_interval_steps)
        self.max_command_hold_steps = int(max_command_hold_steps)
        self.max_joint_drift_rad = float(max_joint_drift_rad)
        self.max_ee_drift_m = float(max_ee_drift_m)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy_inference")
        self._future: Future[PolicyDecision] | None = None
        self._request_observation: Observation | None = None
        self._request_started_at: float | None = None
        self._step = 0
        self._next_request_step = 0
        self._last_decision: PolicyDecision | None = None
        self._last_decision_step = 0
        self._terminal = False
        self._last_latency_s: float | None = None
        self._submitted = 0
        self._accepted = 0
        self._stale_dropped = 0
        self._model_errors = 0
        self._safe_hold_steps = 0
        self._reused_command_steps = 0

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        """Reset policy state before creating the first asynchronous request."""
        self._cancel_pending()
        self.policy.reset(task, model)
        self._future = None
        self._request_observation = None
        self._request_started_at = None
        self._step = 0
        self._next_request_step = 0
        self._last_decision = None
        self._last_decision_step = 0
        self._terminal = False
        self._last_latency_s = None
        self._submitted = 0
        self._accepted = 0
        self._stale_dropped = 0
        self._model_errors = 0
        self._safe_hold_steps = 0
        self._reused_command_steps = 0

    def act(self, observation: Observation) -> PolicyDecision:
        """Poll one completed decision, schedule one request, and return control now."""
        self._step += 1
        completed = self._collect_completed(observation)
        if completed is not None:
            return completed

        if not self._terminal and self._future is None and self._step >= self._next_request_step:
            self._submit(observation)

        return self._fallback(observation)

    @property
    def inference_pending(self) -> bool:
        return self._future is not None

    def summary(self) -> dict[str, int | float | None]:
        return {
            "submitted": self._submitted,
            "accepted": self._accepted,
            "stale_dropped": self._stale_dropped,
            "model_errors": self._model_errors,
            "safe_hold_steps": self._safe_hold_steps,
            "reused_command_steps": self._reused_command_steps,
            "last_latency_s": self._last_latency_s,
        }

    def close(self) -> None:
        self._cancel_pending()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _submit(self, observation: Observation) -> None:
        snapshot = _copy_observation(observation)
        self._future = self._executor.submit(self.policy.act, snapshot)
        self._request_observation = snapshot
        self._request_started_at = perf_counter()
        self._next_request_step = self._step + self.decision_interval_steps
        self._submitted += 1

    def _collect_completed(self, observation: Observation) -> PolicyDecision | None:
        if self._future is None or not self._future.done():
            return None

        future = self._future
        request_observation = self._request_observation
        started_at = self._request_started_at
        self._future = None
        self._request_observation = None
        self._request_started_at = None
        self._last_latency_s = perf_counter() - started_at if started_at is not None else None

        try:
            decision = future.result()
        except Exception as exc:  # noqa: BLE001 - model failures must stop safely.
            self._model_errors += 1
            self._terminal = True
            return self._with_async_debug(
                PolicyDecision(
                    command=JointPositionCommand(
                        observation.joint_position, _clamp_gripper(observation.gripper_opening)
                    ),
                    stage="async_model_error",
                    rationale=f"Asynchronous policy call failed; holding safely: {exc}",
                    done=True,
                ),
                source="model_error",
            )

        if request_observation is None or not self._is_fresh(request_observation, observation):
            self._stale_dropped += 1
            return None

        self._last_decision = decision
        self._last_decision_step = self._step
        self._accepted += 1
        self._terminal = decision.done
        return self._with_async_debug(decision, source="fresh_decision")

    def _fallback(self, observation: Observation) -> PolicyDecision:
        if self._last_decision is not None:
            decision_age = self._step - self._last_decision_step
            if decision_age <= self.max_command_hold_steps:
                self._reused_command_steps += 1
                return self._with_async_debug(self._last_decision, source="command_reuse")
            gripper_opening = _command_gripper_opening(
                self._last_decision, fallback=observation.gripper_opening
            )
            target_id = self._last_decision.target_id
        else:
            gripper_opening = _clamp_gripper(observation.gripper_opening)
            target_id = None

        self._safe_hold_steps += 1
        return self._with_async_debug(
            PolicyDecision(
                command=JointPositionCommand(
                    observation.joint_position, _clamp_gripper(gripper_opening)
                ),
                stage="async_wait" if self._future is not None else "async_hold",
                rationale="Waiting for asynchronous planning; holding the latest safe joint pose.",
                target_id=target_id,
            ),
            source="safe_hold",
        )

    def _is_fresh(self, request: Observation, current: Observation) -> bool:
        joint_drift = float(np.max(np.abs(current.joint_position - request.joint_position)))
        ee_drift = float(np.linalg.norm(current.ee_position - request.ee_position))
        return joint_drift <= self.max_joint_drift_rad and ee_drift <= self.max_ee_drift_m

    def _with_async_debug(self, decision: PolicyDecision, *, source: str) -> PolicyDecision:
        async_debug = {
            "source": source,
            "inference_pending": self.inference_pending,
            "last_latency_s": self._last_latency_s,
            "step": self._step,
        }
        return replace(decision, debug={**decision.debug, "async_control": async_debug})

    def _cancel_pending(self) -> None:
        if self._future is not None:
            self._future.cancel()
        self._future = None


def _command_gripper_opening(decision: PolicyDecision, *, fallback: float) -> float:
    command = decision.command
    if isinstance(command, JointPositionCommand):
        return _clamp_gripper(command.gripper_opening)
    return _clamp_gripper(command.gripper_opening) if hasattr(command, "gripper_opening") else _clamp_gripper(fallback)


def _clamp_gripper(opening: float) -> float:
    return float(np.clip(opening, 0.0, 1.0))


def _copy_observation(observation: Observation) -> Observation:
    """Give the worker immutable-by-convention image and state snapshots."""
    cameras = tuple(
        CameraObservation(
            name=camera.name,
            rgb=camera.rgb.copy(),
            depth=camera.depth.copy(),
            position=camera.position.copy(),
            rotation=camera.rotation.copy(),
            fovy_degrees=camera.fovy_degrees,
        )
        for camera in observation.cameras
    )
    return Observation(
        time=observation.time,
        instruction=observation.instruction,
        joint_position=observation.joint_position.copy(),
        joint_velocity=observation.joint_velocity.copy(),
        ee_position=observation.ee_position.copy(),
        ee_quaternion=observation.ee_quaternion.copy(),
        gripper_opening=observation.gripper_opening,
        cameras=cameras,
    )
