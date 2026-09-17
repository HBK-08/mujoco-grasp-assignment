from __future__ import annotations

from dataclasses import replace
from threading import Event
from time import sleep

import numpy as np

from graspbench.async_policy import AsyncPolicyDriver
from graspbench.config import HOME_Q, TaskSpec
from graspbench.env import GraspEnv
from graspbench.runner import run_episode
from graspbench.types import JointPositionCommand, Observation, PolicyDecision


class BlockingPolicy:
    """A deterministic stand-in for a slow remote VLM/SAM request."""

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def reset(self, task: dict, model) -> None:
        self.task = task

    def act(self, observation: Observation) -> PolicyDecision:
        self.started.set()
        assert self.release.wait(timeout=1.0)
        return PolicyDecision(
            command=JointPositionCommand(HOME_Q, 1.0),
            stage="remote_ready",
            rationale="Slow remote request completed.",
            target_id="red_cube",
        )


class ImmediateStopPolicy:
    def reset(self, task: dict, model) -> None:
        self.task = task

    def act(self, observation: Observation) -> PolicyDecision:
        return PolicyDecision(
            command=JointPositionCommand(observation.joint_position, 1.0),
            stage="safe_stop",
            rationale="Quick policy result.",
            done=True,
        )


def test_async_driver_keeps_physics_stepping_while_remote_policy_waits() -> None:
    policy = BlockingPolicy()
    driver = AsyncPolicyDriver(policy, max_command_hold_steps=0)
    task = TaskSpec("async", "pick the red cube", "red_cube", 4)
    try:
        with GraspEnv() as env:
            observation = env.reset(task)
            driver.reset(task.public_dict(), env.model)
            decision = driver.act(observation)
            assert decision.stage == "async_wait"
            assert policy.started.wait(timeout=0.5)

            before = env.data.time
            observation, _ = env.step(decision.command)
            assert env.data.time > before
            slightly_overopen = replace(observation, gripper_opening=1.00001)
            waiting_decision = driver.act(slightly_overopen)
            assert waiting_decision.stage == "async_wait"
            assert waiting_decision.command.gripper_opening == 1.0

            policy.release.set()
            for _ in range(50):
                decision = driver.act(observation)
                if decision.stage == "remote_ready":
                    break
                sleep(0.005)
            assert decision.stage == "remote_ready"
            assert decision.debug["async_control"]["source"] == "fresh_decision"
            assert driver.summary()["accepted"] == 1
    finally:
        driver.close()


def test_async_driver_discards_a_decision_from_a_moved_robot_state() -> None:
    policy = BlockingPolicy()
    driver = AsyncPolicyDriver(policy, max_ee_drift_m=0.01)
    task = TaskSpec("async_stale", "pick the red cube", "red_cube", 5)
    try:
        with GraspEnv() as env:
            observation = env.reset(task)
            driver.reset(task.public_dict(), env.model)
            driver.act(observation)
            assert policy.started.wait(timeout=0.5)
            moved_observation = replace(
                observation, ee_position=observation.ee_position + np.array([0.05, 0.0, 0.0])
            )
            policy.release.set()
            for _ in range(50):
                driver.act(moved_observation)
                if driver.summary()["stale_dropped"]:
                    break
                sleep(0.005)
            assert driver.summary()["stale_dropped"] == 1
            assert driver.summary()["accepted"] == 0
    finally:
        driver.close()


def test_runner_async_mode_paces_for_a_worker_result(tmp_path) -> None:
    task = TaskSpec("async_runner", "pick the red cube", "red_cube", 6)
    with GraspEnv() as env:
        summary = run_episode(
            env,
            ImmediateStopPolicy(),
            task,
            output_dir=tmp_path,
            max_steps=4,
            async_decisions=True,
            realtime=True,
        )
    assert summary["terminal_reason"] == "policy_done"
    assert summary["execution_mode"] == "async"
    assert summary["async_control"]["accepted"] == 1
