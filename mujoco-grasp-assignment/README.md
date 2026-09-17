# MuJoCo 语言条件桌面抓取 Assignment

你将实现一个从自然语言任务到抓取执行与结果验证的闭环系统。框架提供 Franka Panda、
桌面、简单形体与带材质的真实物体、双相机 RGB-D 与本体观测、数值 IK 工具、结构化 policy contract、日志、
录像和公开评测；需要完成的入口是 `policies/student_policy.py`。

## Quick start

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m graspbench.cli --output runs/inspect
pytest
python -m graspbench.evaluate --episodes 1 --output runs/smoke
```

starter policy 只保持安全姿态，成功率为 0 是正常的。完整要求与教程：

- `docs/INSTALL_ZH.md`
- `docs/ASSIGNMENT_ZH.md`
- `docs/ARCHITECTURE_ZH.md`
- `docs/ASYNC_CONTROL_ZH.md`（远端 VLM/SAM 不阻塞仿真的标准执行方式）
- `docs/TELEOP_GUIDE_ZH.md`（无需 API 的键盘遥控：先理解真实物理与 IK）
- `docs/EXTENSIONS_AND_ACADEMIC_INTEGRITY_ZH.md`（VLA、人形平台与课程内自选拓展的边界）
- `docs/METHOD_TRACKS_ZH.md`
- `docs/EVALUATION_AND_RUBRIC_ZH.md`
- `docs/STAGED_TASKS_ZH.md`
- `docs/TESTING_ZH.md`
- `docs/REPORT_TEMPLATE_ZH.md`

模型路线必须接入真实 VLM/SAM/VLA 服务。运行随包提供的 OpenAI-compatible VLM+SAM3 模板前先执行：

```bash
python scripts/check_model_services.py
```

学生笔记本无需承载模型权重；可以使用课程服务器或商用在线服务。固定坐标和简单图像通道
判断不算模型识别。

三阶段公开评测：

```bash
python -m graspbench.evaluate --tasks configs/task1_place_public.json --output runs/task1
python -m graspbench.evaluate --tasks configs/task2_ycb_public.json --output runs/task2
python -m graspbench.evaluate --tasks configs/task3_sort_public.json --output runs/task3
```

Task 3 只有所有物体都进入正确盘子才算完整成功；评测细则见 `docs/STAGED_TASKS_ZH.md`。

机器人模型来自 Google DeepMind
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)，
许可证和本地改动记录在 `src/graspbench/assets/vendor/`。课程固定 MuJoCo 3.3.7 以保证复现。

不要提交 API key、`.venv`、大模型权重或整个 `runs/`。提交代表性日志、汇总结果与演示视频即可。
