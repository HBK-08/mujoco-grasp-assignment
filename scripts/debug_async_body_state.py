#!/usr/bin/env python3
"""Development-only async rollout trace with MuJoCo body-state diagnostics.

The body pose is printed by this outer harness and is never included in the
policy observation or used to select an action.  Standard evaluation continues
to use ``graspbench.evaluate`` and its unchanged physical success predicate.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    # Match scripts/run_student.sh so this development trace exercises the
    # same NVIDIA EGL renderer as the standard evaluator instead of silently
    # falling back to a CPU Mesa implementation in minimal containers.
    nvidia_egl_root = Path(
        os.environ.get("NVIDIA_EGL_ROOT", "/opt/nvidia-egl-580.173.02")
    )
    nvidia_lib = nvidia_egl_root / "usr/lib/x86_64-linux-gnu"
    nvidia_vendor = (
        nvidia_egl_root
        / "usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )
    if (nvidia_lib / "libEGL_nvidia.so.0").is_file():
        prior_library_path = os.environ.get("LD_LIBRARY_PATH")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{nvidia_lib}:{prior_library_path}"
            if prior_library_path
            else str(nvidia_lib)
        )
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(nvidia_vendor))
        # The ELF loader snapshots LD_LIBRARY_PATH at process startup. Restart
        # once before importing MuJoCo so the NVIDIA libraries are effective.
        if os.environ.get("GRASPBENCH_GPU_ENV_READY") != "1":
            os.environ["GRASPBENCH_GPU_ENV_READY"] = "1"
            os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

import mujoco
import numpy as np

from graspbench.async_policy import AsyncPolicyDriver
from graspbench.env import GraspEnv
from graspbench.evaluate import load_policy, load_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="configs/debug_task3_scene2.json")
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--body", default="mustard_bottle")
    parser.add_argument("--action-index", type=int, default=2)
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    if not 0 <= args.episode_index < len(tasks):
        parser.error(
            f"--episode-index must be in [0, {len(tasks) - 1}]"
        )
    task = tasks[args.episode_index]
    policy = load_policy("policies.student_policy:StudentPolicy")
    with GraspEnv() as env:
        observation = env.reset(task)
        driver = AsyncPolicyDriver(policy, max_command_hold_steps=8)
        driver.reset(task.public_dict(), env.model)
        body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, args.body
        )
        joint_id = int(env.model.body_jntadr[body_id])
        velocity_adr = int(env.model.jnt_dofadr[joint_id])
        wall_start = perf_counter()
        last_key: tuple[object, object] | None = None
        try:
            for step in range(args.max_steps):
                decision = driver.act(observation)
                action_index = decision.debug.get("action_index")
                key = (action_index, decision.stage)
                trace = action_index == args.action_index and (
                    key != last_key or step % 10 == 0
                )
                observation, _ = env.step(decision.command)
                if trace:
                    pos = env.data.xpos[body_id]
                    quat = env.data.xquat[body_id]
                    velocity = env.data.qvel[velocity_adr : velocity_adr + 6]
                    print(
                        f"{step:04d} {decision.stage:16s} "
                        f"body=({pos[0]:.5f},{pos[1]:.5f},{pos[2]:.5f}) "
                        f"quat=({quat[0]:.4f},{quat[1]:.4f},{quat[2]:.4f},{quat[3]:.4f}) "
                        f"ee=({observation.ee_position[0]:.5f},"
                        f"{observation.ee_position[1]:.5f},"
                        f"{observation.ee_position[2]:.5f}) "
                        f"opening={observation.gripper_opening:.4f} "
                        f"vlin=({velocity[0]:.4f},{velocity[1]:.4f},"
                        f"{velocity[2]:.4f}) "
                        f"vang=({velocity[3]:.4f},{velocity[4]:.4f},"
                        f"{velocity[5]:.4f}) "
                        f"vnorm={float(np.linalg.norm(velocity)):.4f}",
                        flush=True,
                    )
                last_key = key
                if decision.done:
                    break
                deadline = wall_start + (step + 1) * env.control_dt
                remaining = deadline - perf_counter()
                if remaining > 0:
                    sleep(remaining)
        finally:
            driver.close()


if __name__ == "__main__":
    main()
