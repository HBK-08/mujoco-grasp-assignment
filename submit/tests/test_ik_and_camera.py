import numpy as np

from graspbench.camera import unproject_pixel
from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.ik import DampedLeastSquaresIK


def test_reachable_top_down_ik() -> None:
    with GraspEnv() as env:
        observation = env.reset(TaskSpec("ik", "pick red cube", "red_cube", 1))
        result = DampedLeastSquaresIK(env.model).solve(
            observation.joint_position,
            np.array([0.53, 0.08, 0.56]),
            observation.ee_quaternion,
        )
        assert result.converged
        assert result.position_error < 0.002


def test_overhead_center_pixel_unprojects_to_table() -> None:
    with GraspEnv() as env:
        observation = env.reset(TaskSpec("vision", "pick red cube", "red_cube", 2))
        camera = next(camera for camera in observation.cameras if camera.name == "overhead")
        point = unproject_pixel(camera, (127.5, 127.5))
        np.testing.assert_allclose(point, np.array([0.52, 0.0, 0.4]), atol=2e-3)
