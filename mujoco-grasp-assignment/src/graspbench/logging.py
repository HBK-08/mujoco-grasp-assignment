from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from typing_extensions import Self

from .config import TaskSpec
from .types import Observation, PolicyDecision


class EpisodeLogger:
    def __init__(
        self,
        output_dir: str | Path,
        task: TaskSpec,
        *,
        video_fps: int = 20,
        record_video: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir) / task.episode_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.log_file = (self.output_dir / "events.jsonl").open("w", encoding="utf-8")
        self.video = None
        if record_video:
            self.video = imageio.get_writer(
                self.output_dir / "rollout.mp4",
                fps=video_fps,
                codec="libx264",
                quality=7,
                macro_block_size=16,
            )

    def write_step(
        self,
        step: int,
        observation: Observation,
        decision: PolicyDecision,
        *,
        unsafe_contacts: int,
    ) -> None:
        record = {
            "event": "control_step",
            "step": step,
            "observation": observation.compact_dict(),
            "decision": decision.compact_dict(),
            "unsafe_contacts": unsafe_contacts,
        }
        self.log_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def append_frame(self, frame: np.ndarray) -> None:
        if self.video is not None:
            self.video.append_data(frame)

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def close(self) -> None:
        self.log_file.close()
        if self.video is not None:
            self.video.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
