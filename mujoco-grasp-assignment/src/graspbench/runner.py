from __future__ import annotations

from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from .async_policy import AsyncPolicyDriver
from .config import OBJECT_SPECS, TaskSpec
from .env import GraspEnv
from .logging import EpisodeLogger


def run_episode(
    env: GraspEnv,
    policy,
    task: TaskSpec,
    *,
    output_dir: str | Path,
    max_steps: int = 800,
    record_video: bool = False,
    render_stride: int = 3,
    success_hold_steps: int = 10,
    async_decisions: bool = False,
    realtime: bool = False,
    async_command_hold_steps: int = 8,
) -> dict[str, Any]:
    """Evaluate one episode, optionally keeping physics live during slow inference.

    ``async_decisions`` preserves the student-facing ``policy.act`` contract:
    the framework owns exactly one worker and continues simulation using the
    latest short-horizon command followed by a local safe hold.  ``realtime``
    should be enabled for remote VLM/SAM evaluation so wall-clock inference has
    time to return before the virtual episode budget is exhausted.
    """
    observation = env.reset(task)
    async_driver = (
        AsyncPolicyDriver(policy, max_command_hold_steps=async_command_hold_steps)
        if async_decisions
        else None
    )
    if async_driver is not None:
        async_driver.reset(task.public_dict(), env.model)
    else:
        policy.reset(task.public_dict(), env.model)
    wall_start = perf_counter()
    success_streak = 0
    unsafe_contacts = 0
    retries = 0
    selected_target = None
    terminal_reason = "max_steps"
    last_stage = "reset"
    verification_observed = False
    planned_actions: list[dict] | None = None

    with EpisodeLogger(
        output_dir,
        task,
        record_video=record_video,
        video_fps=max(1, round(1.0 / (env.control_dt * render_stride))),
    ) as logger:
        if record_video:
            logger.append_frame(env.render(camera="diagonal", width=640, height=480))
        try:
            for step in range(max_steps):
                decision = (
                    async_driver.act(observation) if async_driver is not None else policy.act(observation)
                )
                last_stage = decision.stage
                verification_observed = verification_observed or decision.stage in {
                    "verify",
                    "verify_place",
                }
                if decision.target_id is not None:
                    selected_target = decision.target_id
                task_plan = decision.debug.get("task_plan")
                if isinstance(task_plan, dict) and isinstance(task_plan.get("actions"), list):
                    planned_actions = task_plan["actions"]
                if decision.request_retry:
                    retries += 1
                observation, step_info = env.step(decision.command)
                unsafe_contacts += step_info.unsafe_contacts
                logger.write_step(
                    step,
                    observation,
                    decision,
                    unsafe_contacts=step_info.unsafe_contacts,
                )
                if record_video and step % render_stride == 0:
                    logger.append_frame(env.render(camera="diagonal", width=640, height=480))

                if env.is_task_success(task):
                    success_streak += 1
                else:
                    success_streak = 0
                if success_streak >= success_hold_steps and verification_observed:
                    terminal_reason = "success"
                    break
                if decision.done:
                    terminal_reason = "policy_done"
                    break
                if realtime:
                    deadline = wall_start + (step + 1) * env.control_dt
                    remaining = deadline - perf_counter()
                    if remaining > 0:
                        sleep(remaining)
            else:
                step = max_steps - 1
        finally:
            if async_driver is not None:
                async_driver.close()

        displacements = env.object_displacements()
        goal_names = {target for target, _ in task.goals}
        non_target_displacement = max(
            (value for name, value in displacements.items() if name not in goal_names), default=0.0
        )
        expected_actions = [
            {"pick_id": target, "place_id": destination} for target, destination in task.goals
        ]
        plan_matches = (
            sorted(planned_actions, key=lambda item: (item["pick_id"], str(item["place_id"])))
            == sorted(expected_actions, key=lambda item: (item["pick_id"], str(item["place_id"])))
            if planned_actions is not None
            else None
        )
        summary = {
            "episode_id": task.episode_id,
            "split": task.split,
            "seed": task.seed,
            "instruction": task.instruction,
            "expected_target": task.target,
            "selected_target": selected_target,
            "target_selection_correct": (
                bool(plan_matches) if task.task_kind == "sort" else selected_target == task.target
            ),
            "task_plan_matches_hidden_goal": plan_matches,
            "success": success_streak >= success_hold_steps and verification_observed,
            "terminal_reason": terminal_reason,
            "control_steps": step + 1,
            "control_budget": max_steps,
            "sim_time_s": float(env.data.time),
            "wall_time_s": perf_counter() - wall_start,
            "execution_mode": "async" if async_driver is not None else "sync",
            "realtime": realtime,
            "async_control": async_driver.summary() if async_driver is not None else None,
            "last_stage": last_stage,
            "verification_observed": verification_observed,
            "retry_count": retries,
            "unsafe_contact_count": unsafe_contacts,
            "object_xy_displacement_m": displacements,
            "max_non_target_xy_displacement_m": non_target_displacement,
            "object_catalog": [spec.name for spec in OBJECT_SPECS],
            "task_kind": task.task_kind,
            "hidden_goals": [
                {"target": target, "destination": destination} for target, destination in task.goals
            ],
            "goal_status": env.task_goal_status(task),
            "completed_goal_count": sum(env.task_goal_status(task).values()),
        }
        logger.write_summary(summary)
    return summary
