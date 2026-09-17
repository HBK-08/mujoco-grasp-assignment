# 视频证据索引

视频分为两层：`demo_3_to_5min.mp4` 是供老师快速查看的主演示；`full_rollouts/` 保存全部 15 个公开 episode 的完整原始录像。正式成绩仍以 `../results/aggregates/` 中的完整公开集结果为准。

## 主演示视频

文件：`demo.mp4`

- 时长：约 4 分 26.5 秒
- 编码：H.264，640×480，8 fps
- 协议：所有片段均来自标准 `async + realtime` 成功回合
- 没有加速、跳帧式剪辑或删除单个回合的中间执行过程；仅将四个完整回合首尾拼接

| 时间范围（约） | Task | Episode | Seed | 指令概述 | 结果 |
|---|---|---|---:|---|---|
| 00:00–00:14 | Task 1 | `t1_public_003` | 33 | 蓝色长方体放入圆盘 | success |
| 00:14–00:36 | Task 2 | `t2_public_001` | 61 | 香蕉放入圆盘 | success |
| 00:36–02:24 | Task 3 | `t3_public_001` | 91 | 中文四物体语义分拣 | success |
| 02:24–04:27 | Task 3 | `t3_public_002` | 92 | 英文四物体语义分拣 | success |

## 全部公开 episode 原始录像

录像批次独立运行在当前冻结代码上，三组结果均为成功率 100%、核心率指标 100%、0 次不安全接触、`async + realtime`。对应的聚合结果和逐 episode 摘要位于 `../results/video_run/`。

### Task 1

| 视频 | Seed | 步数/预算 | 时长（约） | 结果 |
|---|---:|---:|---:|---|
| `full_rollouts/task1/t1_public_001_seed31.mp4` | 31 | 551/800 | 23.13 s | success |
| `full_rollouts/task1/t1_public_002_seed32.mp4` | 32 | 407/800 | 17.13 s | success |
| `full_rollouts/task1/t1_public_003_seed33.mp4` | 33 | 329/800 | 13.88 s | success |
| `full_rollouts/task1/t1_public_004_seed34.mp4` | 34 | 305/800 | 12.88 s | success |
| `full_rollouts/task1/t1_public_005_seed35.mp4` | 35 | 356/800 | 15.00 s | success |
| `full_rollouts/task1/t1_public_006_seed36.mp4` | 36 | 409/800 | 17.25 s | success |

### Task 2

| 视频 | Seed | 步数/预算 | 时长（约） | 结果 |
|---|---:|---:|---:|---|
| `full_rollouts/task2/t2_public_001_seed61.mp4` | 61 | 534/800 | 22.38 s | success |
| `full_rollouts/task2/t2_public_002_seed62.mp4` | 62 | 488/800 | 20.50 s | success |
| `full_rollouts/task2/t2_public_003_seed63.mp4` | 63 | 703/800 | 29.50 s | success |
| `full_rollouts/task2/t2_public_004_seed64.mp4` | 64 | 492/800 | 20.63 s | success |
| `full_rollouts/task2/t2_public_005_seed65.mp4` | 65 | 799/800 | 33.50 s | success |
| `full_rollouts/task2/t2_public_006_seed66.mp4` | 66 | 461/800 | 19.38 s | success |

### Task 3

| 视频 | Seed | 步数/预算 | 时长（约） | 结果 |
|---|---:|---:|---:|---|
| `full_rollouts/task3/t3_public_001_seed91.mp4` | 91 | 2579/3000 | 107.63 s | success |
| `full_rollouts/task3/t3_public_002_seed92.mp4` | 92 | 2944/3000 | 122.88 s | success |
| `full_rollouts/task3/t3_public_003_seed93.mp4` | 93 | 2727/3000 | 113.75 s | success |

## 结果边界

- `results/aggregates/` 与 `results/summaries/` 是此前完整正式公开集验收的结果。
- `results/video_run/` 是为了生成上述 15 个视频而进行的独立全量录像批次；该批次同样为 15/15。
- `results/logs/` 保留一份代表性成功事件日志和一份修复前真实失败事件日志。
- 不根据公开视频声称隐藏集成绩。

