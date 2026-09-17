from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.types import JointPositionCommand
from policies.student_policy import StudentPolicy


def test_student_policy_contract_runs() -> None:
    with GraspEnv() as env:
        task = TaskSpec("student", "pick the red cube", "red_cube", 0)
        observation = env.reset(task)
        policy = StudentPolicy()
        policy.reset(task.public_dict(), env.model)
        decision = policy.act(observation)
        assert isinstance(decision.command, JointPositionCommand)
        assert decision.stage

