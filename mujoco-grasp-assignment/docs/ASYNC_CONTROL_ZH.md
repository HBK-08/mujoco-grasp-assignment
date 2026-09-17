# 异步模型决策与实时控制

远端 VLM、SAM 或 Agent 的网络调用可能耗时数百毫秒到数秒。它们不能阻塞物理控制循环。
本项目的标准评测命令默认启用框架级异步执行：学生仍只需实现同步的
`reset()` 与 `act(observation)`，不需要自行管理线程。

```text
25 Hz 仿真主线程                         单一 policy worker
-----------------                         --------------------
observation_t ── snapshot ─────────────>  policy.act(snapshot)
    │                                          │ VLM / SAM / Agent
    │ fresh decision? <────────────────────────┘
    ├─ 是：执行最新经过状态检查的命令
    └─ 否：短暂复用上一命令；超时后保持当前关节位置
```

## 框架安全语义

- 每个 policy 同时至多一个 `act()` 在运行，因此学生可继续写普通的有状态状态机；
- worker 收到独立的 RGB-D 与本体观测快照，不能读取或修改 `GraspEnv`；
- 最新命令最多复用 8 个控制步（默认 25 Hz 下为 0.32 s）；之后发送“保持当前关节位置”的
  本地安全命令，夹爪保持最近语义；
- 若推理期间末端移动超过 2.5 cm 或任一关节变化超过 0.15 rad，结果视为过期并丢弃，重新
  基于最新观测请求；
- policy worker 抛出异常时，框架立即安全保持并结束该回合；
- `events.jsonl` 中每个决策的 `debug.async_control` 会标记它来自新结果、短暂复用还是安全保持；
  `summary.json` 汇总提交数、接受数、过期丢弃数、等待时长和安全保持步数。

这不是让旧命令无限运动的 action chunking；安全保持是本地控制层的停止权。

## 学生应如何实现

保持接口不变：

```python
def reset(self, task: dict, model: mujoco.MjModel) -> None: ...
def act(self, observation: Observation) -> PolicyDecision: ...
```

在 `act()` 中可以直接调用 VLM/SAM；框架会把整次调用放在唯一 worker 中。推荐将远端调用仅放在
`INITIALIZE`、目标歧义、抓取/放置验证失败等事件上。高频的轨迹插值、IK、夹爪控制和碰撞门控应
保持本地、快速且确定。

不要在学生 policy 内再创建每步一个线程、每帧发一次 HTTP 请求，或持有 `env.data` / renderer 引用。
所有规划只能使用传入的 `Observation` 快照和只读 `model`。

## 运行方式

```bash
# 默认：异步 worker + 真实 25 Hz 控制节拍，适用于远端模型。
python -m graspbench.evaluate --policy policies.student_policy:StudentPolicy

# 纯本地、无模型的快速调试；不应用于远端模型效果报告。
python -m graspbench.evaluate --no-realtime

# 兼容旧式阻塞 policy 的对照运行。
python -m graspbench.evaluate --execution-mode sync
```

`--async-command-hold-steps N` 可以调整命令短暂复用窗口。报告中必须记录该值、模型延迟和过期
结果比例。
