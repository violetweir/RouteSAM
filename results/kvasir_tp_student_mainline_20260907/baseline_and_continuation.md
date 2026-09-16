# Kvasir 新基线与 TP-only 学生续跑

## 冻结基线

SAM3-base Target Pooling b0–b6 + 独立 Router；test 100张，Dice **0.885433**，IoU **0.828274**。

原数据划分800/100/100，人工标注8张、无标注训练792张，均保持原身份。SAM3特征与传播256，beam width32，每图7个TP候选；Router为原validation拟合的ridge=1、无模式位，最终选一个mask。

远程封存：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_tp_b0_b6_router_baseline_20260907`。保存checkpoint哈希、协议、Router、100个最终mask与来源校验。

## 复用与重建

- 复用同checkpoint、同anchor/桥路径、同256画布的原TP候选：792×7=5544条训练传播结果。
- 用冻结的TP独立Router重新选择，并按文档固定`q_multi>=0.90`、`q_return>=0.95`筛选；新池为**448/792**。
- S3共识重新按每图7个TP候选生成；不使用旧TP+PC伪标签池、共识图或学生权重。
- 验证集同阈值保留52/100。其筛选后Dice是全validation拟合Router的描述性统计，不作为独立泛化成绩。

## 本轮流程

1. S2：8张人工GT + 448张Router选中硬伪标签。
2. S3：同样的8张GT + 448张TP候选共识软标签；像素权重为`max(0.1, exp(-4*variance))`。
3. 导出S2/S3的validation-best及final训练集预测，组成文档中的四成员委员会。
4. 对剩余344张无标注训练图进行Tier A/B/C筛选。
5. X3：原448 + Tier A + Tier B重新建池训练，GT/original/new每批3/3/6。
6. 冻结普通validation Dice最高的X3 checkpoint；使用固定B7公式，在相同TP b0–b6候选中选mask。
7. 评估全部100张test：X3单图Dice/IoU、X3+B7 Dice/IoU，以及与冻结Router基线的差值。

S2/S3/X3各40000次迭代，每200次验证，seed2026。GPU1已有任务，本轮固定GPU0顺序运行。S2/S3训练结束时的test评估已延后；不根据test选checkpoint。后续SAM3 LoRA不属于本次Step3–Step10队列。

## 训练前修复

实际检查发现旧T24读取器对16位PNG调用`convert('L')`，将S3概率的8个等级饱和成0/255，同时将像素权重全部饱和为255。

本轮在独立脚本中按位深正确归一化：uint16除65535，uint8除255；S2二值标签读取结果不变。S3软标签与权重的原始等级均保留。X3和委员会也共用该读取器。修复前脚本及配置存于`code_before_uint16_fix`，修复发生于任何学生训练之前。

因此本轮同时包含TP-only新池和软标签读取修复；与旧双路线学生的差值不能全部归因于TP-only。

## 产物与状态

远程主目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_tp_student_mainline_20260907`。

- `config.json`：固定参数与脚本哈希。
- `preflight.json`：8+448输入身份、目标路径及概率解码检查。
- `status.json`、`events.jsonl`：当前阶段与历史事件。
- `students/S2`、`students/S3`、`students/X3`：训练日志、验证指标及checkpoint。
- `audit`：委员会结果；`b7_validation`、`b7_test`：最终选路与逐图指标。
- 完成后写入`reproduction_reports/Kvasir_TP_student_mainline_20260907.md`。
