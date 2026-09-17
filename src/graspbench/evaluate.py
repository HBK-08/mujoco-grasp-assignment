from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Iterable
from pathlib import Path

from .config import TaskSpec
from .env import GraspEnv
from .runner import run_episode


def load_tasks(path: str | Path) -> list[TaskSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TaskSpec.from_dict(item) for item in payload["episodes"]]


def load_policy(spec: str):
    try:
        module_name, class_name = spec.split(":", 1)
    except ValueError as exc:
        raise ValueError("Policy must use module:Class syntax") from exc
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def aggregate(summaries: Iterable[dict]) -> dict:
    rows = list(summaries)
    if not rows:
        return {"episodes": 0}
    plan_rows = [row for row in rows if row["task_plan_matches_hidden_goal"] is not None]
    return {
        "episodes": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "target_selection_accuracy": sum(row["target_selection_correct"] for row in rows)
        / len(rows),
        "verification_rate": sum(row["verification_observed"] for row in rows) / len(rows),
        "mean_control_steps": sum(row["control_steps"] for row in rows) / len(rows),
        "mean_wall_time_s": sum(row["wall_time_s"] for row in rows) / len(rows),
        "total_unsafe_contacts": sum(row["unsafe_contact_count"] for row in rows),
        "mean_non_target_displacement_m": sum(
            row["max_non_target_xy_displacement_m"] for row in rows
        )
        / len(rows),
        "goal_completion_rate": sum(row["completed_goal_count"] for row in rows)
        / sum(len(row["hidden_goals"]) for row in rows),
        "task_plan_match_rate": (
            sum(row["task_plan_matches_hidden_goal"] is True for row in plan_rows) / len(plan_rows)
            if plan_rows
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a grasp policy on deterministic tasks")
    parser.add_argument("--policy", default="policies.student_policy:StudentPolicy")
    parser.add_argument("--tasks", default="configs/public_tasks.json")
    parser.add_argument("--output", default="runs/public_eval")
    parser.add_argument(
        "--execution-mode",
        choices=("async", "sync"),
        default="async",
        help="Run policy inference on one worker while physics continues (default: async).",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pace control steps to simulation time; defaults to on for async execution.",
    )
    parser.add_argument(
        "--async-command-hold-steps",
        type=int,
        default=8,
        help="Maximum 25 Hz control steps to reuse a command before local safe hold.",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the task budget (800 for Task 1/2, 3000 for Task 3).",
    )
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    if args.episodes is not None:
        tasks = tasks[: args.episodes]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    async_decisions = args.execution_mode == "async"
    realtime = async_decisions if args.realtime is None else args.realtime
    with GraspEnv() as env:
        for task in tasks:
            policy = load_policy(args.policy)
            summary = run_episode(
                env,
                policy,
                task,
                output_dir=output_dir,
                max_steps=args.max_steps if args.max_steps is not None else task.control_budget,
                record_video=args.video,
                async_decisions=async_decisions,
                realtime=realtime,
                async_command_hold_steps=args.async_command_hold_steps,
            )
            summaries.append(summary)
            print(
                f"{task.episode_id}: success={summary['success']} "
                f"target={summary['selected_target']} steps={summary['control_steps']}"
            )

    report = aggregate(summaries)
    report["policy"] = args.policy
    report["task_file"] = str(args.tasks)
    report["execution_mode"] = args.execution_mode
    report["realtime"] = realtime
    (output_dir / "aggregate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
