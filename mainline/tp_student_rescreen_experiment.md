# SAM3 TP + 单学生：全量重筛训练池实验

2026-09-10 已完成全量重筛、标签生成、816 epoch 训练，以及 best/final 的完整 test 评估。

目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/tp_student_rescreen_20260910`

## 冻结的规则

对原划分的全部 792 张无标注训练图统一审核，不把是否属于原 580 张池作为准入条件。8 张 GT 身份保持。

审核学生为上一阶段软标签模型的 validation-best（epoch 556，validation Dice 0.827330）；checkpoint SHA256 为 `c03b6a63c754df9bc75a6d904bb4b64cabdf7a0e29508b37d8cbbed2a698fc58`。未使用 test 更高的 final 替代教师。

| 类别 | 返回一致性 | 候选与其他六条 TP 的平均 Dice | 与学生二值预测 Dice |
|---|---:|---:|---:|
| A | ≥0.95 | ≥0.95 | 不作为硬门槛 |
| B | ≥0.95 | ≥0.80 | ≥0.85 |

候选必须非空；B 类要求学生预测非空。有 A 候选优先在 A 中选，否则在 B 中选；同级使用原冻结 TP Router 排序。没有合格候选则不入池。阈值在查看本轮 validation 诊断前固定，未因结果进行调整。

## 重筛结果

| 项目 | 数量 |
|---|---:|
| 审核图像 | 792 |
| 审核 TP 候选 | 5,544 |
| A 类接纳 | 538 |
| B 类接纳 | 82 |
| 未接纳 C 类 | 172 |
| 新伪标签池 | **620** |
| 原 580 张保留 | 580 |
| 原池退出 | 0 |
| 原池之外新增 | **40** |
| 保留图像中主 mask 改变 | **22** |

原 580 张逐一重新审核后恰好仍全部通过，并非程序自动保留。其中 538 张为 A、42 张为 B；新增 40 张均通过 B 条件。

## 标签与训练

所有接纳图使用同一公式：`Y = .75*M + .25*P_TP`。M 为本次选中的主 mask；P_TP 为该图全部返回一致性 ≥0.95 的 TP 候选的等权平均。学生只参与准入，不混入最终监督目标。

所有样本与像素等权，A/B 不形成不同权重或采样流。620 张伪标签加 8 张 GT，共 628 张；统一打乱，每轮各出现一次。batch size=12，最后一批保留 4 张；53 iter/epoch，816 epoch 共 **43,248 iter**。

复用上一阶段保存的相同初始 U-Net 权重，而非从审核学生继续微调。其余沿用同一代码配方：逐图 BCE + Soft Dice 后平均，无伪标签系数、无 ramp；SGD 初始学习率 0.01，momentum 0.9，weight_decay 1e-4，学习率按 epoch 线性衰减；每 4 epoch 验证一次。训练代码只为可变池大小替换了固定计数断言及统计。

固定 epoch 意味着总 iter 由上一阶段 39,984 增至 43,248。因此这是固定 epoch 下的重筛配方比较，不是固定迭代预算比较。

## 验证与评估

预检查通过：A 不被学生否决、B 必须满足共同支持、所有 792 张决策可重算、训练输入哈希与标签二值形状一致、所有样本权重为 1、真实一轮 628 张无重复且 GT 恰好 8 张、最后一批 4 张、初始权重一致、CUDA 前向/反向正常。

冻结规则在 validation 上接纳 75/100 张，接纳子集的 SAM3 主 mask 平均 Dice 为 0.912085。该数值只描述接纳子集，不是新学生的完整 validation 或 test 成绩，也不是 OOF 估计。审核学生的完整 validation Dice 复现为 0.827330；诊断后未调整阈值。

训练结束后先冻结 validation-best 和预先固定的 epoch816 final，再各评估完整 test 一次。best 为主要模型选择口径，final 为固定训练终点补充。会分别与上一阶段软标签 best/final 做对应比较，不按 test 重新选择准入门槛。

## 文件

- `policy.json`、`teacher.json`：规则与冻结审核学生。
- `screening/train/`：全部候选分数及每张图决策。
- `data/pool_summary.json`、`data/membership_changes.jsonl`：池变化。
- `data/train_manifest.jsonl`、`data/label_audit.jsonl`：正式标签及清单。
- `preflight_results.json`：程序、数据与采样检查。
- `runs/soft/status.json`、`train_epochs.jsonl`、`validation.jsonl`：新学生进度。
- 完成后 `results.json`、`report.md`、`completion_audit.json`，汇总到 `new_project/tp_student_rescreen_results.md`。


## 完成结果

620 张伪标签 + 8 GT，816 epoch 共 43,248 iter，所有轮次无重复遍历。validation-best（epoch576）validation Dice 0.826621、test Dice 0.858330；final（epoch816）validation Dice 0.810647、test Dice 0.869478。对应旧 580 张软标签池 test 为 0.851243 和 0.859530，分别提高 0.007086 和 0.009948。最终模型及 200 张 test 输出 mask 核查通过。详见 [tp_student_rescreen_results.md](tp_student_rescreen_results.md)。
