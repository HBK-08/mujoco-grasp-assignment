# 面向学生发布包：教师检查清单

发布文件由项目根目录执行 `python scripts/build_student_release.py` 生成：

```text
dist/mujoco-grasp-assignment/       可解压、可直接发放的目录
dist/mujoco-grasp-assignment-student.zip
```

发布前确认：

1. 在干净目录解压 ZIP，按 `README.md` 安装并运行 `pytest`；
2. `policies/student_policy.py` 仍是安全保持的 starter，未携带 `instructor_solution/`；
3. `.env.example` 不含真实密钥；
4. `docs/TELEOP_GUIDE_ZH.md` 的 VNC display 与本次课程服务器一致；
5. 公开任务、隐藏任务和评分细则的版本号已由教师记录；
6. 仅教师保留 `instructor_solution/`、隐藏 task JSON、真实模型密钥和演示服务器运维脚本。

教师参考策略、`instructor_solution/`、隐藏任务和教师端 VNC 推理演示均不进入学生包。学生实际可直接运行的是 `view_scene.py` 与不需要 API 的 `view_teleop_gpu.py`。
