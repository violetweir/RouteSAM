> 本目录是**主线路**的实验树（Kvasir-SEG / ISIC2018 / BUSI 1% 锚点实验）。
> 主线路总览见 [`../docs/cross_dataset_1pct.md`](../docs/cross_dataset_1pct.md)，
> 全部实验索引见 [`EXPERIMENTS.md`](EXPERIMENTS.md)，Kvasir-SEG 实验总表见
> [`../docs/kvasir_program.md`](../docs/kvasir_program.md)。
> 本文是其中 TP-only 伪标签池部分的方法记录。

# SAM3-base TP-only：先筛合格候选，再由 Router 选优

本文记录 Kvasir 首批伪标签池从 448 张扩展至 580 张的方法。核心改动是调整候选筛选与 Router 选优的先后顺序。

## 1. 方法变化

| 环节 | 原来 | 改进后 |
|---|---|---|
| 每图候选 | TP b0–b6，共 7 张 mask | 相同 |
| 筛选顺序 | Router 先选一张，再检查是否合格 | 先筛合格候选，Router 再选 |
| 首批训练图数量 | 448 张 | 580 张 |

沿用 **SAM3-base Target Pooling b0–b6 + 独立 Router**。不重新生成 TP 候选，不改变冻结 Router，不重新划分数据。

原数据划分保持为 train 800 张、validation 100 张、test 100 张；训练集内部保持原来的 8 张有标注图和 792 张无标注图。首批伪标签池从这 792 张无标注图中筛选。

## 2. 两个筛选条件

### 图像整体一致性：q_multi ≥ 0.90

对于同一目标图像，计算 TP b0–b6 七张候选 mask 两两之间的 Dice，再对全部 21 个组合求平均，得到该图像的整体一致性 q_multi。

这个分数判断七条传播路径对目标区域的预测是否整体一致。此处 q_multi 是图像级分数，不能与后续委员会中“单个候选相对其他六个候选的平均 Dice”混用。

### 候选返回一致性：q_return ≥ 0.95

将某条路径得到的目标 mask 作为反向传播提示，沿对应路径返回 anchor 图像，比较返回预测与 anchor 的已知 GT，得到该候选的返回一致性。现有质量文件中对应字段为 `q_cycle`。

七个候选各自拥有一个返回一致性分数。计算使用 anchor 的已知标注，不使用目标无标注训练图的 GT。

这些一致性条件是质量筛选依据，不保证通过的目标 mask 一定正确。

## 3. 原方法：先选，再检查

对每张无标注训练图：

1. 生成或读取 TP b0–b6 的七张候选 mask。
2. 用冻结 Router 对全部候选打分，选最高分候选。
3. 检查图像整体 q_multi 是否 ≥ 0.90，以及选中候选的 q_return 是否 ≥ 0.95。
4. 两项都满足才接纳；否则该图像不进入首批伪标签池。

按此规则，792 张无标注图中有 **448 张**进入首批池。

其不足在于：Router 首选候选未通过返回检查时，即使其他候选满足条件，该图像也会被排除。

## 4. 改进方法：先筛，再选

对每张无标注训练图：

1. 读取相同的七张 TP 候选 mask。
2. 如果图像整体 q_multi < 0.90，则该图像不进入首批池。
3. 如果图像整体条件满足，保留 q_return ≥ 0.95 的候选，形成合格候选集合。
4. 如果合格集合为空，则该图像不进入首批池。
5. 如果集合非空，用同一个冻结 Router 在其中选择分数最高的一张，作为该图像的主伪标签。

```text
每张无标注训练图的七张 TP 候选
                ↓
图像整体一致性 q_multi ≥ 0.90？
    否 → 暂不进入首批池
    是
                ↓
保留 q_return ≥ 0.95 的候选
    无合格候选 → 暂不进入首批池
    有合格候选
                ↓
冻结 Router 在合格候选中选择最高分 mask
                ↓
图像及选中 mask 进入首批伪标签池
```

Router 的“最高分”表示模型估计的优先级，并不是使用目标 GT 找到真实 Dice 最高的候选。

## 5. 为什么从 448 张变成 580 张

原流程中，有 160 张图像满足整体一致性条件，但 Router 首选候选未通过返回一致性检查。其中 **132 张**存在其他返回合格的候选。

调整筛选顺序后，这 132 张图像得到可用的候选，因而：

**新首批伪标签池 = 原 448 张 + 新接纳 132 张 = 580 张。**

原 448 张仍被接纳，其选中候选和 mask 经核对保持一致。剩余 **212 张**暂不进入首批伪标签池，可供后续扩展阶段进一步审核。

此处增加的是训练覆盖量，不能直接视为标签精度或最终分割性能提升。

## 6. 这套池在既有 S2/S3 中的使用方式

- **S2：**使用这 580 张图像及各自被选中的 TP mask，作为硬伪标签训练。
- **S3：**使用相同的 580 张图像；当前既有实现仍以每图全部七张 TP mask 的等权平均作为软标签。

上述入池改动没有同时将 S3 改为“只融合合格候选”，也没有实现候选可靠性加权融合；这些属于后续待研究的标签生成改进，不纳入本文已经完成的方法。

这套筛选规则用于训练伪标签准入。test 上的独立 Router 基线仍对完整 test 集输出预测，不按训练准入条件删除测试图像。

## 7. 对应实现与产物

服务器项目根目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`

- 筛选顺序诊断：`work/kvasir_tp_pseudo_filter_20260909/`
- 已生成的 580 张首批清单：`work/kvasir_tp_filterfirst_students_20260909/pseudo_manifest_original.jsonl`
- 首批池核查：`work/kvasir_tp_filterfirst_students_20260909/pool_audit.json`
- 冻结 Router：`work/kvasir_tp_filterfirst_students_20260909/router.json`

本文聚焦已经实现的 448→580 首批池筛选逻辑，不以旧学生实验的性能证明该规则有效。学生训练实现修复与后续 X3 重构分别记录在 `reproduction_reports/Kvasir_X3_pool_v2_plan_20260909.md`。

## 8. 已确认的后续单学生训练方案

S2/S3 合并为单学生 S，使用 588 张图统一随机采样、样本等权、batch size 12、训练 816 epoch（39,984 iter）。具体记录见 [single_student_training.md](single_student_training.md)。这是已确认的训练配方；伪标签构造已确定为 `Y = 0.75*M + 0.25*P`，保留主 mask 硬标签对照，详见 [pseudo_label_construction.md](pseudo_label_construction.md)。正式标签已生成并核查，硬/软对照训练已启动，见 [single_student_experiment.md](single_student_experiment.md)。


## 实验完成更新

硬、软两组均完成 816 epoch（39,984 iter），随后固定各自 validation-best 并评估完整 test。硬标签 validation/test Dice 为 0.819562/0.848592，软标签为 0.827330/0.851243。软标签 test 平均提高 0.002651，配对 95% 区间跨零，尚不能确认稳定提升。完整结果见 [single_student_experiment_results.md](single_student_experiment_results.md)。


## 全量重筛实验已启动

2026-09-10：用冻结软标签学生与 TP 候选共同重筛全部 792 张，接纳 620 张，其中原 580 张仍通过、新增 40 张，原池 22 张主 mask 改变。新学生使用 628 张图统一等权训练 816 epoch。规则、核查与输出位置见 [tp_student_rescreen_experiment.md](tp_student_rescreen_experiment.md)。

全量重筛实验现已完成，best/final test Dice 为 **0.858330 / 0.869478**，详见 [tp_student_rescreen_results.md](tp_student_rescreen_results.md)。
