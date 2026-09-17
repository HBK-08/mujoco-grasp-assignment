from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio

from .config import TaskSpec
from .env import GraspEnv


def inspect_main() -> None:
    parser = argparse.ArgumentParser(description="Reset the scene and save camera observations")
    parser.add_argument("--output", default="runs/inspect")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    task = TaskSpec("inspect", "抓取桌面上的红色方块", "red_cube", args.seed)
    with GraspEnv() as env:
        observation = env.reset(task)
        for camera in ("front", "overhead", "diagonal"):
            iio.imwrite(output / f"{camera}.png", env.render(camera=camera))
        (output / "state.json").write_text(
            json.dumps(observation.compact_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote observations to {output.resolve()}")


if __name__ == "__main__":
    inspect_main()

