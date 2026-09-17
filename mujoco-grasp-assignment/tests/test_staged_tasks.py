from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from graspbench.config import TaskSpec
from graspbench.env import GraspEnv


def _move_object(env: GraspEnv, name: str, position: np.ndarray) -> None:
    joint_id = env._object_joint_ids[name]
    qpos_adr = env.model.jnt_qposadr[joint_id]
    dof_adr = env.model.jnt_dofadr[joint_id]
    env.data.qpos[qpos_adr : qpos_adr + 3] = position
    env.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
    env.data.qvel[dof_adr : dof_adr + 6] = 0.0


def test_task_public_contract_hides_destinations_and_sort_answers() -> None:
    task = TaskSpec(
        "sort",
        "sort fruit and tools",
        "apple",
        9,
        targets=("apple", "orange", "mustard_bottle", "potted_meat_can"),
        destinations=("round_tray", "round_tray", "square_tray", "square_tray"),
    )
    assert task.task_kind == "sort"
    assert task.public_dict() == {
        "episode_id": "sort",
        "instruction": "sort fruit and tools",
        "split": "public",
    }
    assert task.active_objects == ("apple", "orange", "mustard_bottle", "potted_meat_can")
    assert task.control_budget == 3000

    lift = TaskSpec("lift", "lift apple", "apple", 10)
    assert lift.control_budget == 800


def test_container_truth_predicate_distinguishes_the_requested_tray() -> None:
    task = TaskSpec(
        "place",
        "put the red cube in the square tray",
        "red_cube",
        12,
        destination="square_tray",
        scene_objects=("red_cube", "green_cylinder", "blue_box"),
    )
    with GraspEnv() as env:
        env.reset(task)
        square = env.data.xpos[env._container_body_ids["square_tray"]]
        round_tray = env.data.xpos[env._container_body_ids["round_tray"]]
        _move_object(env, "red_cube", square + np.array([0.0, 0.0, 0.042]))
        mujoco.mj_forward(env.model, env.data)
        assert env.is_object_placed("red_cube", "square_tray")
        assert not env.is_object_placed("red_cube", "round_tray")
        _move_object(env, "red_cube", round_tray + np.array([0.0, 0.0, 0.042]))
        mujoco.mj_forward(env.model, env.data)
        assert not env.is_task_success(task)


def test_sort_evaluator_requires_every_hidden_assignment() -> None:
    task = TaskSpec(
        "sort",
        "sort fruit into the round tray and tools into the square tray",
        "apple",
        13,
        targets=("apple", "orange", "mustard_bottle", "potted_meat_can"),
        destinations=("round_tray", "round_tray", "square_tray", "square_tray"),
    )
    with GraspEnv() as env:
        env.reset(task)
        round_center = env.data.xpos[env._container_body_ids["round_tray"]]
        square_center = env.data.xpos[env._container_body_ids["square_tray"]]
        _move_object(env, "apple", round_center + np.array([0.065, 0.0, 0.050]))
        _move_object(env, "orange", round_center + np.array([-0.048, 0.0, 0.050]))
        _move_object(env, "mustard_bottle", square_center + np.array([-0.055, -0.045, 0.081]))
        _move_object(env, "potted_meat_can", square_center + np.array([0.055, 0.045, 0.055]))
        mujoco.mj_forward(env.model, env.data)
        assert all(env.task_goal_status(task).values())
        assert env.is_task_success(task)


def test_textured_ycb_objects_render_in_task_two_scene() -> None:
    task = TaskSpec(
        "ycb",
        "pick the apple",
        "apple",
        14,
        scene_objects=("banana", "apple", "orange"),
    )
    with GraspEnv() as env:
        observation = env.reset(task)
        overhead = next(camera.rgb for camera in observation.cameras if camera.name == "overhead")
        assert overhead.shape == (256, 256, 3)
        assert np.unique(overhead.reshape(-1, 3), axis=0).shape[0] > 100


def test_task_two_public_set_requires_real_object_placement() -> None:
    task_file = Path(__file__).resolve().parents[1] / "configs" / "task2_ycb_public.json"
    payload = json.loads(task_file.read_text(encoding="utf-8"))
    tasks = [TaskSpec.from_dict(item) for item in payload["episodes"]]
    assert len(tasks) == 6
    assert {task.target for task in tasks} == {"banana", "apple"}
    assert {task.destination for task in tasks} == {"round_tray", "square_tray"}
    assert all(task.task_kind == "place" for task in tasks)


def test_banana_uses_six_vhacd_collision_hulls() -> None:
    with GraspEnv() as env:
        banana_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "banana")
        collision_geoms = [
            mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_GEOM, f"banana_collision_{index:02d}"
            )
            for index in range(6)
        ]
        assert all(env.model.geom_bodyid[geom_id] == banana_body for geom_id in collision_geoms)
        assert all(
            env.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            for geom_id in collision_geoms
        )
        assert all(env.model.geom_dataid[geom_id] >= 0 for geom_id in collision_geoms)
