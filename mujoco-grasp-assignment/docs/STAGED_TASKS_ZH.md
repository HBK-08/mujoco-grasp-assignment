# 三阶段作业与评测

作业从单物体放置逐步扩展到真实物体抓取和语义分拣。每个回合只向 policy
暴露 `episode_id`、自然语言 `instruction` 和 `split`；目标物体、目的盘、随机种子
与最终判定均在评测器侧保存。不得从任务 JSON、文件名或随机种子读取答案。

| 阶段 | 公开集 | 场景与任务 | 成功条件 |
|---|---|---|---|
| Task 1 | `configs/task1_place_public.json` | 红方块、绿圆柱、蓝长方体；按指令放入黄方盘或紫圆盘 | 指定物体在指定盘内、已松爪、静止，连续 10 个控制步成立 |
| Task 2 | `configs/task2_ycb_public.json` | 带材质的 YCB 香蕉和苹果；按指令抓取并放进指定托盘 | 指定物体在指定盘内、已松爪、静止，连续 10 个控制步成立 |
| Task 3 | `configs/task3_sort_public.json` | 苹果、橙子、芥末瓶、午餐肉罐；水果归圆盘、包装食品归方盘 | 四个“物体—容器”隐藏配对全部同时成立，连续 10 个控制步成立 |

每个公开文件含开发用的 6、6、3 个确定性场景。Task 1/2 每回合最多 800 控制步；
Task 3 因为连续执行四个 pick/place/verify 闭环，每回合最多 3000 控制步。教师使用独立
seed 和语言改写的隐藏集；提交后不应根据隐藏集结果继续调参。

## 运行

```bash
# Task 1：单物体 + 指定容器
python -m graspbench.evaluate --tasks configs/task1_place_public.json \
  --output runs/task1

# Task 2：真实水果抓取与放置
python -m graspbench.evaluate --tasks configs/task2_ycb_public.json \
  --output runs/task2

# Task 3：多物体语义分拣
python -m graspbench.evaluate --tasks configs/task3_sort_public.json \
  --output runs/task3
```

`aggregate.json` 的核心指标为：

- `success_rate`：完整回合任务成功率；Task 3 只有四件物体全部归位才记为成功。
- `goal_completion_rate`：所有隐藏目标配对中已完成的比例，便于区分“已完成一半分拣”和完整成功。
- `task_plan_match_rate`：policy 若输出结构化 `task_plan.actions`，其 pick/place 配对和评测器答案一致的比例；它只用于诊断，不能替代物理成功。
- `target_selection_accuracy`：Task 1/2 检查选中的单个物体；Task 3 检查整套计划是否匹配。
- `verification_rate`、控制步数、模型/墙钟时间、重试次数、危险接触和非目标物体位移。

评测器不以“进入 VERIFY stage”或模型口头声称成功作为通过条件；该 stage 仅是要求 policy
提供的闭环证据。最终通过始终来自 MuJoCo 的隐藏几何、速度和夹爪状态判定。

## 推荐实现边界

1. 用图像和指令生成候选物体/容器及其世界坐标，不能读取 `TaskSpec` 私有字段。
2. VLM 对整张 overhead RGB 与候选表输出结构化动作，例如
   `{"actions":[{"pick_id":"apple","place_id":"round_tray"}]}`。
3. 用 SAM mask + 深度 + 标定恢复抓取与放置点；IK/轨迹控制只消费这些估计。
4. 抓取后重新观测，确认抬升或盘内归位；失败时重感知、重规划或安全停止。

Task 1 的关键是语言条件下的对象和容器对应，Task 2 排除纯色几何捷径并要求完成放置，Task 3 则检验多步
规划、类别关系、放置槽分配和逐项验证。它们是同一闭环的难度递进，不是三段彼此独立的
调试脚本。
