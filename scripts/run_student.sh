#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${MUJOCO_GL:-}" && "$(uname -s)" == "Linux" ]]; then
  export MUJOCO_GL=egl
fi

# Some container images expose the NVIDIA kernel/compute driver but omit its
# EGL userspace libraries.  When the matching isolated bundle is present, make
# it visible to GLVND without changing the host driver installation.  Other
# machines keep their normal system EGL discovery path.
NVIDIA_EGL_ROOT="${NVIDIA_EGL_ROOT:-/opt/nvidia-egl-580.173.02}"
if [[ -f "$NVIDIA_EGL_ROOT/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_EGL_ROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-$NVIDIA_EGL_ROOT/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
fi
exec .venv/bin/python -m graspbench.evaluate \
  --policy policies.student_policy:StudentPolicy \
  --tasks configs/public_tasks.json \
  --execution-mode async \
  --realtime \
  --output runs/student "$@"
