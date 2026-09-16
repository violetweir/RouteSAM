# 单学生 S：硬标签 / 合格候选软标签实验

2026-09-09：标签生成与预检查完成，两个正式训练任务已启动。

服务器目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/single_student_hard_soft_20260909`

## 两个对照

| 版本 | 580 张伪标签的监督目标 |
|---|---|
| hard | 冻结 Router 在返回合格候选中选中的主 mask M |
| soft | Y = 0.75 M + 0.25 P，P 为返回合格 TP 候选的平均 mask |

两组均使用同一组 8 张 GT 和 580 张无标注图。同一初始 U-Net 权重、seed=2026、同一 epoch 图像顺序、同一逐图增强随机种子；唯一设计差异为伪标签目标。

## 实际训练配置

- 全部 588 张图统一打乱，每轮各出现一次，没有 GT 重采样。
- Batch size=12；49 iter/epoch；816 epoch，共 39,984 iter。
- 所有样本和像素等权；无伪标签额外系数、无 ramp、无质量加权。
- 统一逐图 BCE + Soft Dice，然后对 batch 求平均；软目标不阈值化。
- SGD：初始学习率 0.01，momentum=0.9，weight_decay=1e-4。
- 学习率按 epoch 线性衰减：第 e 轮使用 `0.01*(1-(e-1)/816)`，轮内固定。
- 每 4 epoch（196 iter）验证完整 100 张 validation；最高平均 Dice 保存 best，同分保留更早 checkpoint。
- 保留原弱增强用于 GT、强增强用于伪标签；两组对应图像的增强一致。
- 所有图像和标签先统一至 256×256，使用 PIL nearest；训练/validation/test 的输入与 GT 处理口径一致。
- 两组均完成训练并固定各自 validation-best 后，各评估完整 test 一次。本轮不构建扩展池或运行 B7。

## 标签与程序核查

580 张主 mask 与既有清单逐项一致，Router 选择已重新计算确认，所有软标签按 0.5 阈值化后仍等于主 mask。

合格候选数分布：1 个=42 张，2 个=54 张，3 个=64 张，4 个=67 张，5 个=41 张，6 个=30 张，7 个=282 张。538 张具有非二值监督像素；42 张单候选图保持硬标签。平均每图约 0.716% 的像素被软化。

预检查覆盖了全部 588 张输入哈希、软标签解码、逐图等权损失、损失对 batch 排列不变、两组增强一致、实际一轮无重复遍历，以及真实 batch size 12 的 CUDA 前向/反向。检查中一轮 49 个 batch 有 41 个不含 GT，这是统一采样的正常结果。

## 查看结果

- `status.json`、`pipeline.log`：总体状态。
- `runs/hard/status.json`、`runs/soft/status.json`：各组实时 epoch、iter、loss。
- `runs/*/train_epochs.jsonl`：每轮样本计数、顺序哈希和 loss。
- `runs/*/validation.jsonl`：每次验证结果。
- `data/summary.json`、`data/label_audit.jsonl`、`preflight_results.json`：数据和程序检查。
- 完成后生成 `results.json`、`report.md`、`completion_audit.json`、每图 test 指标及 mask，并汇总到 `new_project/single_student_experiment_results.md`。

本实验的正式 epoch 配方不同于历史 S2/S3，也不同于此前修复监督错位的 X3-pool-v2 对照；不能将那些结果直接当作本实验结果。


## 实验完成更新

硬、软两组均完成 816 epoch（39,984 iter），随后固定各自 validation-best 并评估完整 test。硬标签 validation/test Dice 为 0.819562/0.848592，软标签为 0.827330/0.851243。软标签 test 平均提高 0.002651，配对 95% 区间跨零，尚不能确认稳定提升。完整结果见 [single_student_experiment_results.md](single_student_experiment_results.md)。
