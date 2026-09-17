# 系统架构与接口

## 固定层与可替换层

```text
TaskSpec(instruction, seed)               evaluator keeps target private
              |
              v
Observation ------------------------------------------------------+
  overhead/front RGB-D + calibration + proprioception             |
              |                                                   |
              v                                                   |
StudentPolicy / Agent                                             |
  target selector -> plan/state -> controller or VLA              |
              |                                                   |
              v                                                   |
AsyncPolicyDriver (one worker; snapshot / freshness gate)        |
  fresh command -> short reuse -> local safe hold                 |
              |                                                   |
              v                                                   |
PolicyDecision                                                    |
  command + stage + rationale + target_id + retry/done             |
              |                                                   |
              v                                                   |
GraspEnv -> MuJoCo physics -> new observation --------------------+
              |
              v
events.jsonl + summary.json + optional rollout.mp4
```

固定层是场景、任务随机化、低层 position actuator、成功判定和指标。学生可替换 target
selector、几何恢复、IK/规划、状态机、Agent runtime 或整个 VLA policy。

## 唯一观测契约

每个 `Observation` 固定包含以下信息：

- `overhead` 与 `front` 两路 256×256 RGB；
- 与 RGB 对齐的 metric depth；
- 相机内参与相机到世界坐标变换；
- 关节位置/速度、末端位姿与夹爪开度。

物体类别、目标身份和物体世界坐标不属于策略观测。VLM+SAM+IK、单一 VLM+IK 和 VLA
都通过同一个接口运行，评测命令不切换观测类型。教师参考实现中 OpenAI-compatible VLM
接收当前 overhead 图像和 SAM3 候选后进行决策，SAM3 mask 再与 depth 融合得到坐标。

## 坐标与动作

- 世界坐标：Z 向上，桌面顶面 `z=0.4 m`；
- `grasp_site`：两指垫中心，用于末端位姿和 Jacobian；
- MuJoCo 四元数顺序：`[w, x, y, z]`；
- RGB 像素：`(u, v)`，u 向右、v 向下；
- `unproject_pixel` 把 metric depth 像素反投影到世界坐标；
- `JointPositionCommand`：7 个 arm joint 目标 + `[0,1]` 夹爪开度；
- `CartesianDeltaCommand`：三维平移、旋转向量、夹爪，可由 adapter 转换。

## 为什么高层 Agent 不直接控制每个物理步

物理步长为 2 ms，默认每 20 个物理步控制一次（25 Hz）。网络 API 延迟通常是 100 ms 到数秒，
把它塞进物理线程会造成仿真暂停、状态过期、成本飙升和不稳定。评测器默认通过
`AsyncPolicyDriver` 在单一 worker 调用 `policy.act()`，物理线程持续运行：短暂复用已验证命令，
随后保持当前关节位置，并丢弃状态漂移后的过期结果。详细语义见
`ASYNC_CONTROL_ZH.md`。Agent/VLM 应在任务开始、目标歧义、验证失败时事件触发；本地 IK/状态机
以稳定频率执行并始终保留停止权。
