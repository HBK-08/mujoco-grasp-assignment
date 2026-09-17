from __future__ import annotations

from time import perf_counter

from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.types import JointPositionCommand


def main() -> None:
    task = TaskSpec("benchmark", "pick the red cube", "red_cube", 0)
    with GraspEnv() as env:
        observation = env.reset(task)
        steps = 500
        start = perf_counter()
        for _ in range(steps):
            observation, _ = env.step(
                JointPositionCommand(observation.joint_position, gripper_opening=1.0)
            )
        elapsed = perf_counter() - start
        simulated = steps * env.control_dt
        print(f"default RGB-D loop wall time: {elapsed:.3f} s")
        print(f"simulated time:    {simulated:.3f} s")
        print(f"real-time factor:  {simulated / elapsed:.1f}x")


if __name__ == "__main__":
    main()
