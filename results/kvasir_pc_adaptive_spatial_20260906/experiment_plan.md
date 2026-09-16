# Kvasir PC 优化：第一轮受控消融

日期：2026-09-06。远端：`violet@222.31.141.50`。

实验目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_pc_adaptive_spatial_20260906`。

## 问题判断

历史 test 上 PC-only b0–b6 Router Dice 为 0.861688，oracle 为 0.911715；TP-only oracle 为 0.907426，TP+PC oracle 为 0.921471。这说明 PC 候选包含有价值的 mask，也说明当前候选选择仍有改进空间。

不过，候选池的 oracle 与 selected Dice 之差不能单独证明 Top-8 构图分数就是原因。修改 correspondence score 会改变 anchor 与桥接路径，需要重新评价生成的候选本身、oracle，以及最终选路三个层面。

本轮不使用学生网络，不叠加 Soft Top-K、前背景对比或双向对应。它们保留为后续独立假设，不在当前三版本结果中混合。

## 三个版本

图像特征为 SAM3-base trunk@256，18×18=324 个 patch；沿用原 8 个标注 anchor 前景原型。

- **v0**：原 Top-8 相似度均值，复用已冻结的历史候选及传播结果作为对照。
- **v1**：`K = clamp(round(anchor原始GT前景面积占比 × 324), 1, 324)`，取最高 K 个相似度的均值。
- **v2**：`score = v1 - 0.05 × (1 - compactness)`；其中 `compactness = K / Top-K patch最小轴对齐包围矩形的面积`。面积以 patch 格点计。

并列相似度按较小的展平 patch 索引优先。惩罚系数 0.05 在验证和测试前固定，没有搜索参数。

8 个 anchor 的自适应 K 依次为：`18, 33, 16, 12, 13, 33, 2, 8`。

此定义的限制：anchor 面积只是 target 面积的先验；包围矩形占有率是集中度代理，无法保证对应病灶或正确形态。不能预设自适应 K 或空间惩罚一定更好。

## 固定项

- 原 train/validation/test = 800/100/100。
- 原训练标注 8 张、无标注 792 张，名单不变。
- SAM3-base checkpoint，不使用 LoRA 或学生网络。
- 传播 canvas=256，beam width=32，b0–b6。
- 默认 KNN patch-mean 特征与图像间相似度保持原缓存。
- 只替换 PC anchor-conditioned score；TP 候选作为固定对照。
- anchor GT 可用于原型、面积与提示；target GT 不进入构图和选路。
- 不重新跑 train propagation；训练图像仅提供 bridge RGB 候选。
- 仅使用 GPU1；GPU0 的已有其他进程不受本脚本管理。

## 验证顺序

1. 检查原数据清单、checkpoint 和特征缓存 SHA256。
2. 针对自适应 K 的边界、相同 appearance 下紧凑/分散 patch 的分数关系运行单元检查。
3. 重新提取 train+validation 的 patch 相似度图；在进入 test 前不读取 test RGB。用重新计算的 Top-8 分数核对历史特征，最大绝对误差必须不超过 1e-5。
4. 对 beam search 作等价向量化加速。用原 v0 分数重建完整 700 条 validation 路线，必须全部与历史 route_id 一致。实际检查中 700 条 route_id 完全一致，但历史 float32 路径分数存在最大 5.96e-8 的数值差异，因此路径分数容差调整为 1e-6；route_id 仍要求逐条完全一致。此调整发生在任何新路线传播和效果评估之前。
5. 用 v1/v2 各生成 100×7=700 条 validation 路线。完全相同的路线可复用同一 base teacher 的传播记录；其路径分数字段必须更新为新构图分数，新路线执行完整 forward/return cycle。
6. validation 按 target ID 做 5 折（seed=2026），同一 target 的全部桥长和模式始终属于同一折。每折只用其他 80 张 target 拟合 ridge=1.0 的传播质量 Router，预测未参与拟合的 20 张。
7. 报告 PC-only、固定 TP+PC 的折外选择 Dice、每桥长 Dice、oracle、改变的 anchor/route 数量，以及按 target 配对的 bootstrap 差值区间。

## Test 准入规则

主要验证指标：**PC-only b0–b6 折外选择 Dice**。

新版本须同时满足：

- 比 v0 提高至少 **0.003**；
- 固定 b3–b6 的平均 Dice 相比 v0 下降不超过 **0.005**。

通过者中选择 PC-only 折外 Dice 更高的版本；完全并列时选更简单的 v1。若无版本通过，本轮停止在 validation，并如实记录阴性结果。

若通过，则冻结版本，执行该版本 test；用各自完整 100 张 validation 拟合最终 Router，比较 v0/胜出版本的 PC-only 与 TP+PC，均采用 b0–b6。无论 test 是否改善都报告，不根据 test 再调参数。

## 实现与运行状态

- 核心实现：本目录 `pc_adaptive_core.py`。
- 调度、提特征、构图、推理与统计：本目录 `run_pc_ablation.py`。
- 远端完整代码快照：实验目录 `code/`。
- 远端当前调度 PID：`261727`（首次启动为 `260741`，数值容差修正后已恢复）；实际状态以远端进程、`pipeline.log` 和标记文件为准。
- 原实验脚本和历史结果保持原样；改动写入独立实验目录。

关键产物：`config.json`、`feature_replay_validation.json`、`v0_route_identity_audit.json`、`v*/route_audit_validation.json`、`v*/reuse_audit_validation.json`、`validation_folds.json`、`validation_decision.json`、`report.md`。通过验证后另有 `test_results.json` 和 test 最终 masks。

`COMPLETE` 表示本轮按预定规则执行完毕；`FAILED` 表示需要检查异常，不能据此声称算法退化或改善。

## 已完成的前置检查与构路观察

- train+validation 共 900 张图的 Top-8 重提分数与原缓存最大绝对误差为 **0.0**。
- v0 全部 700 条 validation route_id 与原路线一致；路径分数最大数值差异为 **5.96e-8**。
- 加速 Ridge 与原实现的验证集路线分数最大差异为 **1.59e-13**，选中的 route_id 变化数为 **0**。此处仅检查实现一致性，不把拟合内 Dice 当验证效果。
- v1 改变 **611/700** 条路线、**437/700** 个 anchor 选择；v2 改变 **637/700** 条路线、**444/700** 个 anchor 选择。
- v0 中最小 anchor（K=2 对应图像）被选中 **213/700** 次；v1 为 **636/700**，v2 为 **643/700**。

该集中现象提示不同 K 的均值分数可能存在跨 anchor 可比性问题：较小 K 更容易获得高分。因此不预设 v1/v2 有效，必须等待候选传播和折外选路结果。空间包围矩形约束在当前系数下没有消除这一现象。

v1 复用 91 条完全相同的既有传播路线，需要实际推理 609 条新路线；当前已进入这部分 validation 推理。v2 将在 v1 完成后运行，允许复用与 v0、TP、v1 完全一致的推理路径。
