from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.types import JointPositionCommand


def main() -> None:
    task = TaskSpec("viewer", "抓取桌面上的红色方块", "red_cube", 3)
    with GraspEnv() as env:
        observation = env.reset(task)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                start = time.perf_counter()
                observation, _ = env.step(
                    JointPositionCommand(observation.joint_position, gripper_opening=1.0)
                )
                viewer.sync()
                remaining = env.control_dt - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)


if __name__ == "__main__":
    main()

