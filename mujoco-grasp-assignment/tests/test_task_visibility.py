from __future__ import annotations

from graspbench.config import HOME_Q, TaskSpec
from graspbench.env import GraspEnv
from graspbench.runner import run_episode
from graspbench.types import JointPositionCommand, Observation, PolicyDecision


class TaskVisibilityProbe:
    """A policy probe that fails if the evaluator leaks answer-only task fields."""

    def reset(self, task: dict, model) -> None:
        assert set(task) == {"episode_id", "instruction", "split"}
        assert "target" not in task
        assert "seed" not in task
        self.task = task

    def act(self, observation: Observation) -> PolicyDecision:
        return PolicyDecision(
            command=JointPositionCommand(HOME_Q, 1.0),
            stage="safe_stop",
            rationale="Task visibility probe; do not move.",
            done=True,
        )


def test_runner_hides_target_and_seed_from_policy_but_scores_against_them(tmp_path) -> None:
    task = TaskSpec("private_answer", "pick the red cube", "red_cube", 991, split="integration")
    with GraspEnv() as env:
        summary = run_episode(
            env,
            TaskVisibilityProbe(),
            task,
            output_dir=tmp_path,
            max_steps=3,
        )

    assert summary["expected_target"] == "red_cube"  # evaluator-side metric only
    assert summary["selected_target"] is None
    assert not summary["target_selection_correct"]
    assert not summary["success"]
    assert summary["terminal_reason"] == "policy_done"
