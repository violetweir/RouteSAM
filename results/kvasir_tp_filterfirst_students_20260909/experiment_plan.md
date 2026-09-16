# Kvasir TP：先筛返回候选，再 Router 选优

2026-09-09 已完成建池、预检查并启动完整学生对照实验。当前结果尚未产生。

## 本次改动

沿用 SAM3-base Target Pooling b0–b6 的候选和冻结的独立 Router。对于每张无标注训练图，仍要求七个候选的整体一致性 q_multi ≥ 0.90；先保留返回一致性 q_return ≥ 0.95 的候选，再由 Router 在其中选分数最高的一张。如果没有合格候选，则不进入首批伪标签池。

首批伪标签由 448 张增加到 580 张，新增 132 张。原 448 张的候选、mask 和图像权重均逐项保持。数据仍为原 train/validation/test = 800/100/100，训练内有标注 8 张、无标注 792 张。

## 已核查

- 580 张选中 mask 的哈希校验通过，筛选决策独立重算一致。
- S3 的 580 张软共识标签已经重新生成。
- S2/S3 均读入 8 张有标注图和 580 张伪标签图；训练目标不读取无标注图的 GT。
- uint16 软标签解码及像素权重预检查通过。
- S2 已实际开始迭代训练，后续阶段由后台流水线顺序执行。

## 后续顺序

S2 40,000 步 → 导出最佳/最终快照 → S3 40,000 步 → 导出最佳/最终快照 → 四成员委员会审核剩余 212 张 → 重建并去重 X3 清单 → X3 40,000 步 → validation 比较 → 冻结 checkpoint 和 B7 规则 → 完整 test 评估一次。

学生结构、初始化方式、seed、损失、批次、训练步数、委员会规则和 B7 公式沿用原 448 张实验。最终 X3 数量须等待新委员会结果；不能直接把 132 加到原 X3 数量上。

## 对照值与判断

| 指标 | 原 448 张实验 |
|---|---:|
| X3 validation Dice（导出入口） | 0.824269 |
| B7 validation Dice | 0.839690 |
| X3 test Dice | 0.867161 |
| B7 test Dice | 0.892639 |
| 单 TP 独立 Router test Dice | 0.885433 |

覆盖率提高不等于质量提高。此前 validation OOF 筛选诊断中，接纳量从 51 增至 71，接纳子集平均 Dice 从 0.942856 降至 0.913386；这些是筛选子集质量，不是完整 validation 分数。本轮通过完整重训检验覆盖率和标签质量的实际取舍，不自动替换既有基线。

## 运行位置

服务器：`violet@222.31.141.50`

独立实验目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_tp_filterfirst_students_20260909`

状态：`status.json`；逐阶段日志：`logs/`；主日志：`pipeline.log`；完成结果：`results.json` 和 `report.md`；失败诊断：`FAILED.txt`（仅失败时生成）。

完成后报告自动归档到 `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/reproduction_reports/Kvasir_TP_filterfirst_students_20260909.md`。
