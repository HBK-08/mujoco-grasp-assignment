# 测试与验收分层

本项目的测试不是以“对象能否构造”代替任务完成。测试分为三层，命令与通过条件不同。

## 1. 离线单元与接口测试

```bash
pytest
```

不调用任何模型服务。它验证场景、相机反投影、IK、动作接口、VLM 的图像请求和 JSON 校验、
缓存，以及最重要的权限边界：`policy.reset()` 只能得到 `episode_id`、`instruction` 和 `split`，
不能拿到 `target` 或 `seed`。这些测试通过只表示底座可用，不代表完成抓取。

其中异步控制测试还验证：远端决策未完成时物理时间仍前进、策略结果被单 worker 串行化、状态漂移后
的结果会丢弃，以及夹爪安全保持开度会被约束在 `[0, 1]`。

## 2. 实时视觉-控制任务验收

```bash
GRASPBENCH_SAM3_URL=http://127.0.0.1:8767/infer \
GRASPBENCH_RUN_SAM3_INTEGRATION=1 \
pytest -m 'integration and not live_model'
```

该测试选择三个未出现在 public task 文件中的随机 seed（101、211、307），分别指定红色方块、
绿色圆柱和蓝色长方体。策略由当帧 SAM3 RGB mask 与 depth 恢复位置；测试的语言模块只解析
指令文本，拿不到评测答案、随机 seed、物体坐标或 MuJoCo data。

每个回合必须同时满足：

- `selected_target` 等于评测器保留的目标；
- 目标稳定抬升 10 个控制步；
- 出现 `raise_clear → move_above → descend → close → lift → verify` 的任务阶段；
- 已进入 `VERIFY` 才能计为成功；
- 危险接触为 0，非目标物体水平扰动小于 1 cm；
- 事件日志包含 SAM3 的当前帧 mask 几何证据。

## 3. 实时 VLM + SAM3 全链路验收

在 shell 或密钥管理器中设置 `GRASPBENCH_VLM_BASE_URL`、`GRASPBENCH_VLM_API_KEY`、
`GRASPBENCH_VLM_MODEL` 和 `GRASPBENCH_SAM3_URL` 后：

```bash
python scripts/check_model_services.py
GRASPBENCH_RUN_LIVE_VLM_INTEGRATION=1 pytest -m live_model
```

它在 409、503、617 三个额外随机布局中运行真实 OpenAI-compatible VLM 与 SAM3。除第二层
的全部成功条件外，还要求日志中的 grounding 服务是 `openai_compatible_vlm`，并记录模型、
endpoint、原始 JSON（不含 API key）与 SAM3 几何摘要。

完整公开集和教师隐藏集的批量分数，仍需通过 `graspbench.evaluate` 单独执行；不能将本层三回合
smoke test 当成最终成绩。
