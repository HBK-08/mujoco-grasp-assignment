# 三条推荐实现路线

所有路线都必须使用真实图像模型或真实 VLA 推理。固定坐标、像素常量、RGB 通道条件和
评测答案不能作为识别实现。

## A. LLM + SAM + RGB-D + IK（教师参考路线）

SAM3、SAM2 在线服务或其他提示分割服务读取实际 RGB，返回 mask/box/score；OpenAI-compatible
VLM 根据自然语言、当前图像和候选选择目标；mask 与 metric depth 恢复世界坐标；本地 DLS IK 和状态机负责
实时执行。动作后再次调用 SAM 跟踪目标，失败时重感知、重规划或停止。

优点：模型证据、几何与控制边界清楚；模型部署在服务器时，学生笔记本只运行 MuJoCo 和
HTTP 客户端。教师答案使用可配置的 OpenAI-compatible VLM + SAM3。

建议实验：不同 SAM prompt、VLM 模型/提示词、模型 score 门限、单次感知与动作后重感知、无重试与重试、
不同 API 延迟和丢包率。

## B. 单一 VLM + RGB-D + IK

使用 Qwen-VL、GPT、Gemini、Claude 或其他视觉模型直接读取 RGB 与指令，返回严格 JSON，
例如目标 box/point、类别、置信度和简短依据；用 depth 与 `unproject_pixel` 恢复世界坐标；
后续使用本地 IK 和状态机。执行后应再次调用 VLM 或分割服务验证结果。

关键工程要求：

- prompt 明确 JSON schema，拒绝无法解析的自由文本；
- bbox/point 必须检查图像边界，depth 必须检查有限值与桌面范围；
- API timeout 后保持安全姿态，不沿旧计划继续；
- 相同图像和指令可以缓存，报告调用次数、P50/P95 延迟与费用；
- API key 只从环境变量读取；
- 禁止在 VLM 失败后静默切换到固定坐标或简单图像通道判断。

## C. VLA（端到端或分层）

`policies/vla_template.py` 定义常见 7D 末端增量 action contract；用
`CartesianDeltaAdapter` 限幅并转为关节位置。可使用本地模型、远端推理服务，或让 VLA
只负责阶段策略而由本地安全层执行。

必须明确：训练数据/权重来源、输入分辨率、动作归一化、控制频率、action chunking、
延迟、夹爪语义和终止条件。端到端不等于免除验证、安全门控和日志。

学生端没有独显时，把模型部署在课程服务器或云端；本地保留 MuJoCo、动作限幅、碰撞检查和
紧急停止。模型服务延迟较高，应低频产生目标或 action chunk，不能阻塞每个物理步。

## Agent 可加在哪里

Agent 适合：解析约束、选择感知工具、检查模型响应、在候选间决策、决定是否重试或澄清、
维护 stage 和记忆。Agent 不适合：生成未经校验的高频关节序列、绕过碰撞/限位或把自由文本
直接作为控制协议。
