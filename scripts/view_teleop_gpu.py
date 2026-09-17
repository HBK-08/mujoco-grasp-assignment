"""Keyboard teleoperation starter for the GPU-rendered MuJoCo VNC display.

This is an onboarding tool, not an evaluation policy.  It deliberately keeps
the task goal hidden from the controller: a student watches the rendered scene
and commands the gripper by hand.  The script converts bounded Cartesian
increments to joint targets with the same numerical IK utility exposed to
student policies.
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.ik import DampedLeastSquaresIK, move_toward
from graspbench.types import JointPositionCommand

TASK_SETS = {
    "task1": (
        TaskSpec(
            "teleop_task1_red",
            "把红色方块放进方盘。",
            "red_cube",
            131,
            destination="square_tray",
            scene_objects=("red_cube", "green_cylinder", "blue_box"),
        ),
        TaskSpec(
            "teleop_task1_green",
            "Place the green cylinder into the round tray.",
            "green_cylinder",
            132,
            destination="round_tray",
            scene_objects=("red_cube", "green_cylinder", "blue_box"),
        ),
    ),
    "task2": (
        TaskSpec(
            "teleop_task2_banana",
            "抓起桌面上的香蕉。",
            "banana",
            161,
            scene_objects=("banana", "apple"),
        ),
        TaskSpec(
            "teleop_task2_apple",
            "Pick up the apple.",
            "apple",
            162,
            scene_objects=("banana", "apple"),
        ),
    ),
    "task3": (
        TaskSpec(
            "teleop_task3_sort",
            "把所有水果放进圆盘，把所有包装食品放进方盘。",
            "apple",
            191,
            targets=("apple", "orange", "mustard_bottle", "potted_meat_can"),
            destinations=("round_tray", "round_tray", "square_tray", "square_tray"),
        ),
    ),
}

_KEY_TO_DELTA = {
    "w": np.array((0.012, 0.0, 0.0)),
    "s": np.array((-0.012, 0.0, 0.0)),
    "a": np.array((0.0, 0.012, 0.0)),
    "d": np.array((0.0, -0.012, 0.0)),
    "r": np.array((0.0, 0.0, 0.012)),
    "f": np.array((0.0, 0.0, -0.012)),
}
_KEY_TO_ROTATION = {
    "q": np.array((0.0, 0.0, 0.08)),
    "e": np.array((0.0, 0.0, -0.08)),
}
_WORKSPACE_LOW = np.array((0.22, -0.42, 0.37))
_WORKSPACE_HIGH = np.array((0.82, 0.42, 0.82))


class TeleopDemo:
    """A focused VNC window that makes the control contract tangible."""

    def __init__(self, tasks: tuple[TaskSpec, ...]) -> None:
        self.root = tk.Tk()
        self.root.title("GraspBench — keyboard teleoperation")
        self.status = tk.StringVar(value="正在初始化 GPU 渲染器…")
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        tk.Label(
            self.root,
            text="W/S: 前后  A/D: 左右  R/F: 上下  Q/E: 末端偏航  Space: 开/合夹爪  H: 重置  N: 下一任务  Esc: 退出",
            anchor="w",
            font=("Sans", 11),
        ).pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(self.root, textvariable=self.status, anchor="w", font=("Sans", 12)).pack(
            fill="x", padx=10, pady=(0, 8)
        )
        self.root.bind_all("<KeyPress>", self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.env = GraspEnv()
        self.ik = DampedLeastSquaresIK(self.env.model)
        self.tasks = tasks
        self.task_index = 0
        self.task: TaskSpec | None = None
        self.observation = None
        self.command: JointPositionCommand | None = None
        self.next_control_at: float | None = None
        self.closed = False
        self.start_episode()

    def start_episode(self) -> None:
        self.task = self.tasks[self.task_index % len(self.tasks)]
        self.task_index += 1
        self.observation = self.env.reset(self.task)
        self.command = JointPositionCommand(
            self.observation.joint_position.copy(), self.observation.gripper_opening
        )
        self.next_control_at = time.monotonic()
        self.status.set(
            f"{self.task.episode_id} | {self.task.instruction} | 点击窗口后按键操作；当前夹爪：张开"
        )
        self.root.focus_force()

    def on_key_press(self, event: tk.Event) -> None:
        if self.closed or self.observation is None or self.command is None:
            return
        key = event.keysym.lower()
        if key == "escape":
            self.close()
            return
        if key == "h":
            self.task_index -= 1
            self.start_episode()
            return
        if key == "n":
            self.start_episode()
            return
        if key == "space":
            opening = 0.0 if self.command.gripper_opening >= 0.5 else 1.0
            self.command = JointPositionCommand(self.command.joint_position, opening)
            self._update_status("夹爪闭合" if opening < 0.5 else "夹爪张开")
            return
        if key not in _KEY_TO_DELTA and key not in _KEY_TO_ROTATION:
            return

        translation = _KEY_TO_DELTA.get(key, np.zeros(3))
        rotation = _KEY_TO_ROTATION.get(key, np.zeros(3))
        target_position = np.clip(
            self.observation.ee_position + translation,
            _WORKSPACE_LOW,
            _WORKSPACE_HIGH,
        )
        target_quaternion = self._target_quaternion(rotation)
        result = self.ik.solve(
            self.observation.joint_position,
            target_position,
            target_quaternion,
            max_iterations=80,
        )
        self.command = JointPositionCommand(
            move_toward(self.observation.joint_position, result.joint_position, max_step=0.08),
            self.command.gripper_opening,
        )
        message = "IK 收敛" if result.converged else "IK 近似到达（已限速）"
        self._update_status(f"{message} | 末端位置 {target_position.round(3).tolist()}")

    def _target_quaternion(self, rotation: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(rotation))
        if angle < 1e-10:
            return self.observation.ee_quaternion.copy()
        delta_quaternion = np.empty(4)
        mujoco.mju_axisAngle2Quat(delta_quaternion, rotation / angle, angle)
        target_quaternion = np.empty(4)
        mujoco.mju_mulQuat(target_quaternion, delta_quaternion, self.observation.ee_quaternion)
        return target_quaternion

    def _update_status(self, message: str) -> None:
        assert self.task is not None
        assert self.command is not None
        gripper = "闭合" if self.command.gripper_opening < 0.5 else "张开"
        success = " | 目标已满足" if self.env.is_task_success(self.task) else ""
        self.status.set(f"{self.task.episode_id} | {message} | 夹爪：{gripper}{success}")

    def tick(self) -> None:
        if self.closed:
            return
        self._render_frame()
        assert self.command is not None
        assert self.next_control_at is not None
        if time.monotonic() >= self.next_control_at:
            self.observation, _ = self.env.step(self.command)
            self.next_control_at += self.env.control_dt
        self.root.after(16, self.tick)

    def _render_frame(self) -> None:
        frame = self.env.render(camera="diagonal", width=960, height=720)
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.env.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.after(0, self.tick)
        self.root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GPU-rendered keyboard teleoperation in a VNC desktop"
    )
    parser.add_argument("--task-set", choices=sorted(TASK_SETS), default="task1")
    args = parser.parse_args()
    TeleopDemo(TASK_SETS[args.task_set]).run()
