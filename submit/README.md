# MuJoCo 语言条件抓取大作业

本提交实现一条可本地复现的闭环路线：公开自然语言指令确定候选类别，微调
YOLOv8n 从当前顶视 RGB 图像检测目标物体与容器，RGB-D 反投影恢复世界坐标，
DLS IK 与有限状态机完成抓取、搬运、放置和视觉验证。检测失败时安全停止；放置
验证失败时最多重新感知并重试一次。

# 团队成员

| 姓名 | 学号 |
| ---- | ---- |
|      |      |
|      |      |
|      |      |
|      |      |

# 目录

- 核心入口：`policies/student_policy.py`
- Task 3 分拣和恢复：`policies/sort_policy.py`
- YOLO/RGB-D 感知：`policies/yolo_perception.py`
- 放置验证：`policies/placement_verification.py`
- 正式结果：`results/aggregates/`
- 成功与失败日志：`results/logs/`
- 主演示录像：`video/demo_3_to_5min.mp4`
- 全部公开回合录像：`video/full_rollouts/`

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
EGL 用户态库，可以通过 `NVIDIA_EGL_ROOT` 指向与内核驱动同版本的隔离库目录；正常安装
了 NVIDIA EGL 的机器不需要设置该变量。可用 `MUJOCO_EGL_DEVICE_ID` 明确选择 MuJoCo
渲染卡。提交包不依赖课程工作站上的固定绝对路径。

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

## 提交包内容

- `report/report.pdf`：实验报告；提交前必须补全首页姓名、学号和团队分工。
- `video/demo_3_to_5min.mp4`：约 4 分 26.5 秒，覆盖 Task 1、Task 2、Task 3 的四个完整成功回合。
- `video/full_rollouts/`：全部 15 个公开 episode 的独立完整录像，文件名包含 `episode_id` 和 seed。
- `video/README.md`：主演示时间轴、15 个视频与 seed/summary 的对应索引。
- `results/aggregates/`：Task 1、Task 2、Task 3 完整公开集正式结果。
- `results/summaries/`：正式公开集全部 15 个 episode 的轻量摘要。
- `results/logs/`：一份代表性成功日志和一份真实失败日志。
- `results/video_run/`：全量录像批次的三个 aggregate 和全部 15 个 summary；该批次同样为 15/15。
- `TEAM.md`：团队成员和可核查贡献；提交前必须填写。
- `SUBMISSION_CHECKLIST.md`：最终提交前检查项。

提交包不包含 `.venv`、真实密钥、隐藏任务、旧调试运行或教师参考答案。公开集成绩不能代表隐藏集成绩。视频只是可视化证据，正式成绩以 `results/aggregates/` 为准。
