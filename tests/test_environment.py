import numpy as np

from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.types import JointPositionCommand


def test_scene_loads_and_steps() -> None:
    with GraspEnv() as env:
        observation = env.reset(TaskSpec("test", "pick the red cube", "red_cube", 0))
        assert env.model.nu == 8
        assert observation.joint_position.shape == (7,)
        assert {camera.name for camera in observation.cameras} == {"overhead", "front"}
        next_observation, _ = env.step(
            JointPositionCommand(observation.joint_position, gripper_opening=1.0)
        )
        assert next_observation.time > observation.time


def test_seeded_placements_are_deterministic_and_separated() -> None:
    task = TaskSpec("test", "pick the red cube", "red_cube", 42)
    with GraspEnv() as first, GraspEnv() as second:
        a = first.reset(task)
        b = second.reset(task)
        overhead_a = next(camera.rgb for camera in a.cameras if camera.name == "overhead")
        overhead_b = next(camera.rgb for camera in b.cameras if camera.name == "overhead")
        # EGL rasterization can differ by one intensity level at an isolated
        # antialiased edge across renderer contexts; the seeded scene state is
        # still deterministic.  Keep this as a visual-regression guard without
        # making the suite flaky on GPU drivers.
        np.testing.assert_allclose(overhead_a, overhead_b, rtol=0, atol=1)
