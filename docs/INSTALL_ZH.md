# 完整安装与使用教程

## 1. 硬件与系统要求

最低建议：

- 64 位 Windows 10/11、Ubuntu 22.04+ 或当前受支持的 macOS；
- Python 3.10 或 3.11；
- 8 GB 内存；
- 仿真客户端使用集成显卡即可。MuJoCo 物理计算主要在 CPU 上；只有显示和图像观测需要 OpenGL；
- 约 1 GB 可用磁盘空间（环境、Python wheel、录像和运行日志）。

仿真客户端不需要 CUDA、PyTorch、ROS 2 或真实机械臂。SAM、商用 VLM 或 VLA
运行在课程服务器/云服务时，学生笔记本只需要网络访问。选择在本地加载模型的组需自行满足
模型服务的显存与依赖要求。

## 2. 获取工程后先检查 Python

```bash
python3 --version
```

应显示 `Python 3.10.x` 或 `Python 3.11.x`。不要使用仓库外已有的混乱环境，
每个组在项目根目录创建 `.venv`。

## 3. Linux / macOS 安装

```bash
cd mujoco-grasp-assignment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

也可以运行等价脚本：

```bash
bash scripts/setup.sh
```

## 4. Windows PowerShell 安装

```powershell
cd mujoco-grasp-assignment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

若系统禁止激活脚本，不必修改全局策略，直接使用完整解释器路径：

```powershell
.venv\Scripts\python -m graspbench.cli --output runs/inspect
```

## 5. 配置真实模型服务

教师参考方案调用一个支持图像输入的 OpenAI-compatible Chat Completions endpoint 和 SAM3：

```text
VLM: ${GRASPBENCH_VLM_BASE_URL}/chat/completions
SAM3: http://127.0.0.1:8765/infer
```

在已经部署服务的教师机器上，把 `.env.example` 的三个 VLM 变量以环境变量方式导出：

```bash
export GRASPBENCH_VLM_BASE_URL=https://YOUR_HOST/compatible-mode/v1
export GRASPBENCH_VLM_API_KEY='set-this-in-your-shell-or-secret-manager'
export GRASPBENCH_VLM_MODEL=your-image-capable-model
export GRASPBENCH_SAM3_URL=http://127.0.0.1:8765/infer
python scripts/check_model_services.py
```

检查会调用 VLM 的 `/models` 和 SAM3 的 `/healthz`，最后两行显示 `VLM ready` 和 `SAM3 ready`。

若 VLM 或 SAM3 adapter 位于其他服务器，先通过 VPN、SSH tunnel 或网关把它们暴露为
学生机可访问的 HTTP endpoint，再设置：

```bash
export GRASPBENCH_VLM_BASE_URL=http://MODEL_HOST:8000/v1
export GRASPBENCH_VLM_API_KEY=read-from-a-secret-manager
export GRASPBENCH_VLM_MODEL=your-image-capable-model
export GRASPBENCH_SAM3_URL=http://MODEL_HOST:8765/infer
python scripts/check_model_services.py
```

VLM 请求使用 OpenAI-compatible `/chat/completions` 的 `image_url` content part，并要求返回
`target_id` 和 `reason` JSON。SAM3 adapter 的请求必须包含 JPEG base64 和 `prompts`；响应必须包含按原图尺寸排列的
`boxes`、`scores` 和 little-endian bit-packed masks。协议细节及返回示例见项目根目录
`src/graspbench/perception.py`。若使用商用 VLM、OpenAI-compatible Qwen-VL 或其他 SAM 服务，应在
`src/graspbench/perception.py` 外围实现相同结构化 adapter，不要伪造本地结果。

只运行 starter policy 和仿真公共测试时不需要模型服务；运行 `VLMAndIKTemplate` 或教师答案
时必须先通过上述检查。模型服务失败会安全停止，不存在隐藏回退。

## 6. 第一次 smoke test

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -m graspbench.cli --output runs/inspect --seed 3
pytest
```

预期输出：

- MuJoCo 版本 `3.3.7`；
- `runs/inspect/` 下有 `front.png`、`overhead.png`、`diagonal.png`、`state.json`；
- 公共单元测试通过。

## 7. OpenGL 与无头运行

先尝试不设置变量。若 Linux 服务器/VNC 中出现 GLFW display 错误：

```bash
MUJOCO_GL=egl python -m graspbench.cli --output runs/inspect
```

有桌面/VNC、希望打开交互 viewer 时：

```bash
MUJOCO_GL=glfw python scripts/view_scene.py
```

在 VNC 中直接用键盘操纵夹爪（同样使用 EGL/GPU 渲染画面）时：

```bash
MUJOCO_GL=egl python scripts/view_teleop_gpu.py --task-set task1
```

完整键位、VNC display 示例和它与正式自动评测的边界见 `docs/TELEOP_GUIDE_ZH.md`。

无 EGL 驱动但允许安装 Mesa 的 Linux 机器，可安装软件渲染库并使用：

```bash
sudo apt-get install libgl1-mesa-glx libosmesa6
MUJOCO_GL=osmesa python -m graspbench.cli --output runs/inspect
```

不要在同一进程中反复切换 `MUJOCO_GL`；它必须在导入 `mujoco` 前设置。

## 8. 运行学生策略

单回合快速检查：

```bash
python -m graspbench.evaluate \
  --policy policies.student_policy:StudentPolicy \
  --tasks configs/public_tasks.json \
  --episodes 1 \
  --output runs/smoke
```

全部公开任务：

```bash
python -m graspbench.evaluate --output runs/public_eval
```

录制 MP4：

```bash
python -m graspbench.evaluate --episodes 1 --video --output runs/video_demo
```

VLM/RGB-D 路线：

```bash
python -m graspbench.evaluate \
  --policy policies.vlm_ik_template:VLMAndIKTemplate \
  --episodes 1 --max-steps 2 --execution-mode sync
```

这条命令只用于验证一次真实服务调用和结构化返回，所以显式使用阻塞模式；正常远端模型评测保持
默认的异步实时模式，具体见 `ASYNC_CONTROL_ZH.md`。

每步观测固定包含 256×256 overhead 与 front RGB-D；图像数组不会写进 JSONL，
日志只保存分辨率、相机名和策略输出的结构化感知证据。

## 9. 输出在哪里

每回合目录包含：

- `events.jsonl`：观测摘要、结构化动作、stage、rationale、重试和接触；
- `summary.json`：成功、目标选择、步数、耗时、危险接触和非目标扰动；
- `rollout.mp4`：仅在 `--video` 时生成。

批量目录还包含 `aggregate.json`。提交前用相同命令从空 `runs/` 目录复现结果。

## 10. 常见错误

`ModuleNotFoundError: graspbench`：没有在项目根目录执行 `pip install -e .`，
或当前终端未使用 `.venv`。

`GLFWError: DISPLAY is missing`：无头 Linux 使用 `MUJOCO_GL=egl`；VNC 则确认
`echo $DISPLAY` 有值并使用 `glfw`。

画面全黑/崩溃：先关闭其他 viewer 进程，用 `grasp-inspect` 测试固定相机；远程机器
保持默认 256×256 双相机观测，并减少录像、关闭其他图形程序。

仿真爆炸：不要把关节目标一次跳变数弧度；使用 `graspbench.ik.move_toward` 限速，
并保持默认 0.002 秒物理步长。

抓到后立即掉落：检查夹爪闭合方向、抓取高度、物体宽度、摩擦与抬升加速度；
先从 `events.jsonl` 区分对准失败、夹空、滑移和验证错误。

`Model service failure`：先运行 `python scripts/check_model_services.py`。检查 endpoint、SSH
tunnel、模型名、服务端日志和超时；不要用固定坐标或简单图像通道判断绕过失败。
