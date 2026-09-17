from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DetectedObject:
    name: str
    color: str
    shape: str
    position: np.ndarray
    quaternion: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "color": self.color,
            "shape": self.shape,
            "position": self.position.tolist(),
            "quaternion": self.quaternion.tolist(),
        }


@dataclass(frozen=True)
class CameraObservation:
    name: str
    rgb: np.ndarray
    depth: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    fovy_degrees: float

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rgb_shape": list(self.rgb.shape),
            "depth_shape": list(self.depth.shape),
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "fovy_degrees": self.fovy_degrees,
        }


@dataclass(frozen=True)
class Observation:
    time: float
    instruction: str
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    ee_position: np.ndarray
    ee_quaternion: np.ndarray
    gripper_opening: float
    cameras: tuple[CameraObservation, ...]

    def compact_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "joint_position": self.joint_position.tolist(),
            "joint_velocity": self.joint_velocity.tolist(),
            "ee_position": self.ee_position.tolist(),
            "ee_quaternion": self.ee_quaternion.tolist(),
            "gripper_opening": self.gripper_opening,
            # Never serialize image arrays into JSONL logs. The metadata is enough
            # to audit which camera payload the policy received.
            "cameras": [camera.metadata_dict() for camera in self.cameras],
        }


@dataclass(frozen=True)
class JointPositionCommand:
    joint_position: np.ndarray
    gripper_opening: float = 1.0

    def __post_init__(self) -> None:
        q = np.asarray(self.joint_position, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(f"joint_position must have shape (7,), got {q.shape}")
        if not 0.0 <= float(self.gripper_opening) <= 1.0:
            raise ValueError("gripper_opening must be in [0, 1]")
        object.__setattr__(self, "joint_position", q)


@dataclass(frozen=True)
class CartesianDeltaCommand:
    """Optional VLA-friendly action; use CartesianDeltaAdapter to turn it into joint targets."""

    translation: np.ndarray
    rotation_vector: np.ndarray
    gripper_opening: float

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation, dtype=np.float64)
        rotation = np.asarray(self.rotation_vector, dtype=np.float64)
        if translation.shape != (3,) or rotation.shape != (3,):
            raise ValueError("translation and rotation_vector must both have shape (3,)")
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "rotation_vector", rotation)


@dataclass
class PolicyDecision:
    command: JointPositionCommand | CartesianDeltaCommand
    stage: str
    rationale: str
    target_id: str | None = None
    done: bool = False
    request_retry: bool = False
    debug: dict[str, Any] = field(default_factory=dict)

    def compact_dict(self) -> dict[str, Any]:
        command = self.command
        if isinstance(command, JointPositionCommand):
            command_dict = {
                "mode": "joint_position",
                "joint_position": command.joint_position.tolist(),
                "gripper_opening": command.gripper_opening,
            }
        else:
            command_dict = {
                "mode": "cartesian_delta",
                "translation": command.translation.tolist(),
                "rotation_vector": command.rotation_vector.tolist(),
                "gripper_opening": command.gripper_opening,
            }
        return {
            "command": command_dict,
            "stage": self.stage,
            "rationale": self.rationale,
            "target_id": self.target_id,
            "done": self.done,
            "request_retry": self.request_retry,
            "debug": self.debug,
        }
