# 评测协议与评分建议（100 分）

## 固定评测协议

- 开发时只使用公开任务；教师隐藏集使用不同 seed 和语言改写；
- 提交前冻结代码、prompt、阈值、重试上限和控制参数；
- Task 1/2 每回合最多 800 控制步，Task 3 最多 3000 步（四次 pick/place/verify）；Task 1/2
  均需目标位于指定盘内、松爪且静止连续 10 步；Task 3 需所有隐藏物体—容器配对同时成立连续
  10 步。三者都要求进入 `VERIFY`
  或 `VERIFY_PLACE` 阶段；
- 报告 target selection、task success、goal completion、task plan match、control steps、wall
  time、retry、unsafe contact、non-target displacement；远端模型还须报告异步请求数、接受数、过期
  丢弃数、安全保持步数、命令复用窗口和模型延迟；
- 标准 `grasp-eval` 使用异步 worker 和 25 Hz 实时控制；如为对照使用 `--execution-mode sync` 或
  `--no-realtime`，必须在报告中标明，且后者不能代表远端模型的实际闭环表现；
- 失败必须保留；禁止在 hidden rollout 后继续调参再报告同一 hidden 分数。

## 建议评分

| 维度 | 分数 | 核心证据 |
|---|---:|---|
| 可安装与可复现 | 10 | 干净环境安装、固定依赖、命令可复现 |
| 系统架构与接口 | 10 | 模块边界、结构化 schema、配置管理 |
| 任务理解与 Agent/决策 | 15 | 指令 grounding、状态、工具调用、边界 |
| 感知与几何 | 15 | 模型请求/响应、mask/box/point、坐标与不确定性 |
| 规划、IK 与执行 | 15 | 可达性、限速、夹爪、轨迹合理性 |
| 闭环验证与恢复 | 15 | 成功判定、失败 taxonomy、重试/停止 |
| 实验与数据证据 | 15 | 重复评测、对照/消融、失败分析、日志 |
| 报告、视频与分工 | 5 | 表达清楚、诚实边界、贡献可核查 |

模型越大不自动得分。VLM+SAM+IK 若闭环稳定、实验严谨、证据可审计，可以得到高分；
VLA 若只有成功视频、无复现与验证，不能得到高分。用固定坐标或简单图像通道判断伪装成
模型识别，不计感知与闭环得分。

Task 3 的一两个物体归位只反映在 `goal_completion_rate`，不计作完整 `success_rate`。
`task_plan_match_rate` 仅用于诊断语言/视觉计划是否正确，不能替代 MuJoCo 的物理成功判定。
详情见 `STAGED_TASKS_ZH.md`。

## 建议及格门槛

- 能安装；
- target selection accuracy >= 2/3；
- public success rate >= 50%；
- 无通过修改评测器作弊；
- 有完整日志、报告、演示和分工。

## 加分方向（不突破 100）

优先用于区分同档项目：跨 seed 隐藏评测、歧义澄清、抓取失败后有效重规划、
API 断网降级、VLA latency-aware action chunk、安全门控、统计置信区间、可复现消融。
