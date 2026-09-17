# 资源复用与设计取舍

## 复用内容

- Google DeepMind MuJoCo Menagerie 的 Franka Emika Panda MJCF、碰撞与视觉 mesh、
  position actuator 和夹爪 tendon；上游 Apache-2.0 许可证保留；
- MuJoCo 官方 Python bindings、Renderer、site Jacobian 与 passive viewer；
- 采用研究 benchmark 常见的 deterministic episode seed、公开/隐藏 split、task success、
  efficiency、disturbance、safety 和逐阶段诊断方式。

## 为什么没有引入完整 RL benchmark

robosuite 的 Lift/PickPlace、ManiSkill、RoboCasa 都是成熟选择；若目标是训练 RL/IL/VLA，
应优先考虑它们。本课程任务只有 10 天、面向 3–5 人本科团队，且必须在集成显卡笔记本可运行。
因此仿真客户端把依赖压到 MuJoCo、NumPy 与录像工具，让学生直接看到 MJCF、坐标、IK 和
日志接口。OpenAI-compatible VLM、SAM3、商用 VLM 或 VLA 通过 HTTP 服务接入，模型算力与学生笔记本解耦；
这不是用轻量图像启发式替代模型感知。

引入完整框架会带来任务注册、Gym/RL wrapper、额外资产/渲染依赖和版本耦合；这些并不是
本作业“自然语言到闭环抓取”的核心学习目标。此选择牺牲大规模 benchmark 可比性，换取
安装成功率、可读性和可替换性。报告若使用外部 benchmark，可通过 policy adapter 另做扩展。

## 场景为什么只用三个基础物体

三个对象足以制造目标属性、语言改写和随机布局，又能用 top-down parallel-jaw grasp 得到稳定
ground truth。透明、柔性、带把手物体应作为加分扩展，而不应让最低作业被高保真接触拖垮。

## 版本与许可证

- 课程 pin `mujoco==3.3.7`；升级必须重新跑物理回归；
- vendored Panda 记录在 `src/graspbench/assets/vendor/NOTICE.md`；
- mesh 未修改；课程只改 include 路径、添加 `grasp_site`、移除与新增 freejoint 不兼容的 keyframe。
