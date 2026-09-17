# 与 10 天课程的知识点对齐

| 课程日 | 工程落点 |
|---|---|
| Day 1 感知-决策-执行 | `Observation -> PolicyDecision -> Env.step -> Observation` |
| Day 2 仿真与接口 | MJCF、关节/任务空间、reset/step、相机和本体状态 |
| Day 3 多模态感知 | RGB-D、目标分割、VLM strict JSON、像素反投影 |
| Day 4 运动学与抓取 | Panda 7-DoF、site Jacobian、DLS IK、pregrasp/夹爪/抬升 |
| Day 5 闭环 | 显式 stage、验证、重试、状态更新 |
| Day 6 Agent | 结构化工具结果、目标 grounding、runtime 状态、停止规则 |
| Day 7 数据评测 | JSONL、episode、指标分层、公开/隐藏切分、失败 taxonomy |
| Day 8 轻量实现 | CPU 物理、低分辨率、事件触发 API、无 CUDA 默认路径 |
| Day 9 复杂决策 | 可扩展歧义、澄清、多目标顺序和策略修正 |
| Day 10 集成答辩 | 一键复现、报告模板、视频、边界与分工 |

reference solution 使用 OpenAI-compatible VLM 结构化决策、SAM3 目标 mask、metric depth 反投影、
front SAM3 动作结果跟踪、DLS IK 和闭环控制。VLM+SAM+IK、单一 VLM+IK 与 VLA
均使用完全相同的观测契约和评测入口，学生应做系统级指标对照，而不是只比较单模块模型精度。
