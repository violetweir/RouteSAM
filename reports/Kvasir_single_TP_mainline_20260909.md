# Kvasir 单 TP 主线确认

确认日期：2026-09-09。用户决定：采用单 TP 路线。

后续采用 **SAM3-base Target Pooling b0–b6 + 独立 Router**。每张目标图保留七个 TP 候选，由原独立 Router 选择最终 mask。

## 已冻结基线

| 指标 | 值 |
|---|---:|
| Test Dice，无学生 | 0.885433 |
| Test IoU，无学生 | 0.828274 |
| Test 候选 Oracle Dice，仅事后诊断 | 0.907426 |
| Validation 5 折 OOF Dice | 0.851755 |

沿用原 train/validation/test = 800/100/100，以及原 8 张有标注、792 张无标注训练图身份。复用已冻结 Router、候选与最终 mask。

冻结产物：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_tp_b0_b6_router_baseline_20260907`。

以上指标是 SAM3 单 TP + 独立 Router 的成绩。此前学生网络及 B7 的成绩保留在各自报告中，不混入本基线。

## 本次取舍

PC、备用 TP anchor（B）、TP/PC 检索与路径对照（C/D）、以及辅助增益选择器和目标端评分实验保留为探索记录，停止作为当前主线继续推进。

本次为路线选择与归档，未重新划分数据、训练模型或产生新的 test 成绩。后续工作以本单 TP 主线为依据。
