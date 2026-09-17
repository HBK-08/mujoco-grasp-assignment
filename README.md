# MuJoCo 语言条件抓取大作业

本提交实现一条可本地复现的闭环路线：公开自然语言指令确定候选类别，微调
YOLOv8n 从当前顶视 RGB 图像检测目标物体与容器，RGB-D 反投影恢复世界坐标，
DLS IK 与有限状态机完成抓取、搬运、放置和视觉验证。检测失败时安全停止；放置
验证失败时最多重新感知并重试一次。

## 安装

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

## 运行

先在终端 1 启动本地 YOLO 服务：

```bash
source .venv/bin/activate
YOLO_DEVICE=0 python yolo/infer_server.py
```

终端 2 按老师规定的标准异步实时协议运行公开评测：

```bash
source .venv/bin/activate
MUJOCO_EGL_DEVICE_ID=1 bash scripts/run_student.sh
```

三阶段分别复现：

```bash
MUJOCO_EGL_DEVICE_ID=1 bash scripts/run_student.sh --tasks configs/task1_place_public.json --output runs/task1
MUJOCO_EGL_DEVICE_ID=1 bash scripts/run_student.sh --tasks configs/task2_ycb_public.json --output runs/task2
MUJOCO_EGL_DEVICE_ID=1 bash scripts/run_student.sh --tasks configs/task3_sort_public.json --output runs/task3
```

最终成绩必须以 `--execution-mode async --realtime` 为准。策略按观测中的仿真时间累计
阶段时钟，因此异步工作线程导致的决策复用不会把闭合、沉降和复检时间缩短。若只想定位
控制问题，可以额外使用 `--execution-mode sync --no-realtime` 做对照，但该结果不作为标准
评测结果。

上述命令在当前工作站上把 YOLO 固定到 GPU 0、MuJoCo EGL 固定到 GPU 1，避免两个进程
争用同一张卡。其他机器可按实际空闲卡修改两个编号；只有在没有 CUDA 时才把
`YOLO_DEVICE` 显式改成 `cpu`。接口地址、权重、置信度和端口可通过 `.env.example`
中的环境变量覆盖。Linux 上 `scripts/run_student.sh` 默认使用 EGL；若系统只暴露 NVIDIA 计算驱动而缺少
EGL 用户态库，可以通过 `NVIDIA_EGL_ROOT` 指向与内核驱动同版本的隔离库目录。当前
机器使用 `/opt/nvidia-egl-580.173.02`，脚本会自动发现它；也可用
`MUJOCO_EGL_DEVICE_ID` 明确选择 MuJoCo 渲染卡。

## 实现边界

- 规定入口为 `policies.student_policy:StudentPolicy`。
- `reset()` 只使用公开字典中的 `instruction`，不使用隐藏目标、随机种子或任务答案。
- YOLO 输出的类别、框和置信度直接决定目标选择；三维位置来自当前 RGB-D 图像。
- 控制阶段包含 `pregrasp`、`approach`、`descend`、`close`、`lift`、
  `place`、`post_place_check` 和 `verify_place`。
- 评测器、成功判定、场景、任务配置、机器人执行器和老师原始测试均保持不变。
- 每个检测和复检事件会把模型名称、框、置信度、延迟与 RGB-D 几何信息写入 JSONL 日志。

随提交提供的 `yolo/best.pt` 是 12 类微调 YOLOv8n 权重（约 6.2 MB）；运行依赖
Ultralytics，适用其 AGPL-3.0 许可。老师的完整任务、评分和报告要求仍以 `docs/`
中的原始说明为准。
