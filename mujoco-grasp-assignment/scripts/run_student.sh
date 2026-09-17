#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${MUJOCO_GL:-}" && "$(uname -s)" == "Linux" ]]; then
  export MUJOCO_GL=egl
fi
exec .venv/bin/python -m graspbench.evaluate \
  --policy policies.student_policy:StudentPolicy \
  --tasks configs/public_tasks.json \
  --output runs/student "$@"
