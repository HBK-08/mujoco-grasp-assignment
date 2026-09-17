from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from typing_extensions import Self

from .config import (
    CONTAINER_SPEC_BY_NAME,
    HOME_Q,
    OBJECT_SPECS,
    SCENE_PATH,
    TABLE_TOP_Z,
    TaskSpec,
)
from .types import (
    CameraObservation,
    CartesianDeltaCommand,
    JointPositionCommand,
    Observation,
)


@dataclass(frozen=True)
class StepInfo:
    unsafe_contacts: int
    contacts: tuple[tuple[str, str], ...]


class GraspEnv:
    """MuJoCo environment whose policy observation always contains calibrated RGB-D."""

    CAMERA_NAMES = ("overhead", "front")
    CAMERA_SIZE = (256, 256)

    def __init__(
        self,
        scene_path: str | Path = SCENE_PATH,
        *,
        control_substeps: int = 20,
    ) -> None:
        self.scene_path = Path(scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.control_substeps = int(control_substeps)
        self.task: TaskSpec | None = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

        self.arm_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                for i in range(1, 8)
            ]
        )
        self.arm_qpos_ids = np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dof_ids = np.array([self.model.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.finger_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ("finger_joint1", "finger_joint2")
            ]
        )
        self.finger_qpos_ids = np.array([self.model.jnt_qposadr[j] for j in self.finger_joint_ids])
        self.grasp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )
        self._object_body_ids = {
            spec.name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec.name)
            for spec in OBJECT_SPECS
        }
        self._object_joint_ids = {
            spec.name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{spec.name}_free"
            )
            for spec in OBJECT_SPECS
        }
        self._container_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in CONTAINER_SPEC_BY_NAME
        }
        self._default_container_positions = {
            name: self.model.body_pos[body_id].copy()
            for name, body_id in self._container_body_ids.items()
        }
        self._initial_object_positions: dict[str, np.ndarray] = {}
        self._active_object_names: tuple[str, ...] = ()

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep * self.control_substeps)

    def reset(self, task: TaskSpec) -> Observation:
        self.task = task
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_ids] = HOME_Q
        self.data.qpos[self.finger_qpos_ids] = 0.04
        self.data.ctrl[:7] = HOME_Q
        self.data.ctrl[7] = 255.0

        self._active_object_names = task.active_objects
        container_positions = self._sample_container_positions(task.seed)
        for name, body_id in self._container_body_ids.items():
            self.model.body_pos[body_id] = container_positions[name]

        placements = self._sample_placements(task.seed, self._active_object_names, container_positions)
        rng = np.random.default_rng(task.seed + 17_029)
        for spec in OBJECT_SPECS:
            joint_id = self._object_joint_ids[spec.name]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            if spec.name in placements:
                # The long banana is otherwise sometimes aligned with the
                # parallel jaw closing direction, an accidental geometry
                # failure unrelated to language/vision/sorting.  Keep its
                # lying orientation graspable; all positions remain seeded.
                yaw = np.pi / 4.0 if spec.name == "banana" else float(rng.uniform(-np.pi, np.pi))
                self.data.qpos[qpos_adr : qpos_adr + 3] = (
                    placements[spec.name][0],
                    placements[spec.name][1],
                    TABLE_TOP_Z + spec.half_height + 0.001,
                )
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (
                    np.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    np.sin(yaw / 2.0),
                )
            else:
                # Keep non-episode objects out of all task cameras and the tabletop.
                self.data.qpos[qpos_adr : qpos_adr + 3] = (3.0 + joint_id, 3.0, -1.0)
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(120):
            mujoco.mj_step(self.model, self.data)

        self._initial_object_positions = {
            name: self.data.xpos[body_id].copy()
            for name, body_id in self._object_body_ids.items()
            if name in self._active_object_names
        }
        return self.observe()

    @staticmethod
    def _sample_placements(
        seed: int,
        object_names: tuple[str, ...],
        container_positions: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        names = list(object_names)
        minimum_distance = 0.135 if len(names) <= 3 else 0.082
        for _ in range(2_000):
            # The arm starts centred over x≈0.48.  Keeping objects beyond that
            # reach-clear zone makes every episode visually observable from the
            # fixed overhead camera instead of asking a policy to infer an
            # initially occluded object.
            points = np.column_stack(
                [rng.uniform(0.54, 0.70, len(names)), rng.uniform(-0.24, 0.24, len(names))]
            )
            if "banana" in names:
                # Keep the long object in the well-conditioned central grasp
                # lane while still varying it with the episode seed.  Other
                # objects and both trays remain randomized, so this is not a
                # fixed-coordinate solution to the task.
                banana_index = names.index("banana")
                points[banana_index] = (rng.uniform(0.55, 0.59), rng.uniform(-0.03, 0.03))
            distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
            distances += np.eye(len(names))
            tray_clearance = min(
                np.linalg.norm(points - position[:2], axis=1).min()
                for position in container_positions.values()
            )
            if float(np.min(distances)) >= minimum_distance and tray_clearance >= 0.170:
                return {name: points[index] for index, name in enumerate(names)}
        raise RuntimeError("Could not sample a collision-free tabletop layout")

    def _sample_container_positions(self, seed: int) -> dict[str, np.ndarray]:
        """Keep trays visible but random enough that hard-coded placement poses fail."""
        rng = np.random.default_rng(seed + 8_381)
        square = self._default_container_positions["square_tray"].copy()
        round_tray = self._default_container_positions["round_tray"].copy()
        square[:2] = (rng.uniform(0.30, 0.35), rng.uniform(-0.20, -0.14))
        round_tray[:2] = (rng.uniform(0.30, 0.35), rng.uniform(0.14, 0.20))
        return {"square_tray": square, "round_tray": round_tray}

    def observe(self) -> Observation:
        if self.task is None:
            raise RuntimeError("Call reset(task) before observe()")
        ee_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quaternion, self.data.site_xmat[self.grasp_site_id])
        height, width = self.CAMERA_SIZE
        camera_payloads = []
        for camera_name in self.CAMERA_NAMES:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            camera_payloads.append(
                CameraObservation(
                    name=camera_name,
                    rgb=self.render(camera=camera_name, width=width, height=height, mode="rgb"),
                    depth=self.render(
                        camera=camera_name, width=width, height=height, mode="depth"
                    ),
                    position=self.data.cam_xpos[camera_id].copy(),
                    rotation=self.data.cam_xmat[camera_id].reshape(3, 3).copy(),
                    fovy_degrees=float(self.model.cam_fovy[camera_id]),
                )
            )
        return Observation(
            time=float(self.data.time),
            instruction=self.task.instruction,
            joint_position=self.data.qpos[self.arm_qpos_ids].copy(),
            joint_velocity=self.data.qvel[self.arm_dof_ids].copy(),
            ee_position=self.data.site_xpos[self.grasp_site_id].copy(),
            ee_quaternion=ee_quaternion,
            gripper_opening=float(np.mean(self.data.qpos[self.finger_qpos_ids]) / 0.04),
            cameras=tuple(camera_payloads),
        )

    def step(self, command: JointPositionCommand | CartesianDeltaCommand) -> tuple[Observation, StepInfo]:
        if isinstance(command, CartesianDeltaCommand):
            raise TypeError(
                "GraspEnv consumes joint-position commands. Wrap Cartesian-delta policies with "
                "graspbench.adapters.CartesianDeltaAdapter."
            )
        q_target = np.clip(
            command.joint_position,
            self.model.jnt_range[self.arm_joint_ids, 0],
            self.model.jnt_range[self.arm_joint_ids, 1],
        )
        self.data.ctrl[:7] = q_target
        self.data.ctrl[7] = float(command.gripper_opening) * 255.0
        for _ in range(self.control_substeps):
            mujoco.mj_step(self.model, self.data)
        contacts = self.contact_pairs()
        unsafe = sum(self._is_unsafe_contact(pair) for pair in contacts)
        return self.observe(), StepInfo(unsafe_contacts=unsafe, contacts=contacts)

    def contact_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            pairs.append((name1 or f"geom_{contact.geom1}", name2 or f"geom_{contact.geom2}"))
        return tuple(pairs)

    @staticmethod
    def _is_unsafe_contact(pair: tuple[str, str]) -> bool:
        joined = " ".join(pair)
        robot_contact = any(token in joined for token in ("link", "hand", "finger", "pad"))
        static_contact = any(token in joined for token in ("tabletop", "leg_", "floor"))
        return robot_contact and static_contact

    def is_success(self, target: str, lift_margin: float = 0.10) -> bool:
        target_z = float(self.data.xpos[self._object_body_ids[target], 2])
        return target_z >= TABLE_TOP_Z + lift_margin

    def is_object_placed(self, target: str, destination: str) -> bool:
        """Hidden truth predicate for container placement; policies never receive it."""
        spec = CONTAINER_SPEC_BY_NAME[destination]
        target_position = self.data.xpos[self._object_body_ids[target]]
        container_position = self.data.xpos[self._container_body_ids[destination]]
        local_xy = target_position[:2] - container_position[:2]
        if spec.inner_half_extents is not None:
            inside = bool(
                abs(local_xy[0]) <= spec.inner_half_extents[0]
                and abs(local_xy[1]) <= spec.inner_half_extents[1]
            )
        else:
            assert spec.inner_radius is not None
            inside = bool(np.linalg.norm(local_xy) <= spec.inner_radius)
        joint_id = self._object_joint_ids[target]
        velocity_adr = self.model.jnt_dofadr[joint_id]
        velocity = float(np.linalg.norm(self.data.qvel[velocity_adr : velocity_adr + 6]))
        gripper_open = float(np.mean(self.data.qpos[self.finger_qpos_ids]) / 0.04) >= 0.70
        return bool(
            inside
            and TABLE_TOP_Z - 0.01 <= target_position[2] <= spec.max_object_center_height
            and velocity <= 0.18
            and gripper_open
        )

    def task_goal_status(self, task: TaskSpec | None = None) -> dict[str, bool]:
        task = task or self.task
        if task is None:
            raise RuntimeError("Call reset(task) before querying task status")
        status: dict[str, bool] = {}
        for target, destination in task.goals:
            key = target if destination is None else f"{target}->{destination}"
            status[key] = (
                self.is_success(target)
                if destination is None
                else self.is_object_placed(target, destination)
            )
        return status

    def is_task_success(self, task: TaskSpec | None = None) -> bool:
        return all(self.task_goal_status(task).values())

    def object_displacements(self) -> dict[str, float]:
        return {
            name: float(
                np.linalg.norm(
                    self.data.xpos[body_id, :2] - self._initial_object_positions[name][:2]
                )
            )
            for name, body_id in self._object_body_ids.items()
            if name in self._initial_object_positions
        }

    def render(
        self,
        *,
        camera: str = "diagonal",
        width: int = 640,
        height: int = 480,
        mode: str = "rgb",
    ) -> np.ndarray:
        renderer = self._renderers.get((height, width))
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[(height, width)] = renderer
        renderer.disable_depth_rendering()
        if mode == "depth":
            renderer.enable_depth_rendering()
        elif mode != "rgb":
            raise ValueError("mode must be one of: rgb, depth")
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
