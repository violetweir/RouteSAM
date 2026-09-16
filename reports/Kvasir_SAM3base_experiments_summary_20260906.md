# Kvasir-SEG：SAM3-base 当前实验汇总

整理日期：2026-09-06（Asia/Shanghai）  
服务器：`violet@222.31.141.50`  
远端项目：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`

本文汇总本次实际完成的三组实验：两条伪视频路线的 test 重跑、不使用学生网络的最终 mask 选择，以及 SAM3-base 单图空文本/有文本预测。所有结果均来自实际运行产物，指标以 0–1 表示。

## 1. 主要结果

| 实验 | 设置 | Test Dice | Test IoU | 评测分辨率 |
|---|---|---:|---:|---|
| 单图 Direct | 空文本 `""` | 0.361394 | 0.304767 | 1008×1008 |
| 单图 Direct | 文本 `colon polyp` | 0.461427 | 0.432159 | 1008×1008 |
| Target pooling 最终 mask | 传播质量 Router，b3–b6 | 0.876869 | 0.815768 | 256×256 |
| Patch Correspondence 最终 mask | 传播质量 Router，b3–b6 | 0.874294 | 0.810286 | 256×256 |
| 两条路线联合最终 mask | 传播质量 Router，b3–b6 | **0.881029** | **0.817542** | 256×256 |
| Target pooling 最终 mask | 传播质量 Router，b0–b6 | **0.885433** | **0.828274** | 256×256 |
| Patch Correspondence 最终 mask | 传播质量 Router，b0–b6 | 0.861688 | 0.799622 | 256×256 |
| 两条路线联合最终 mask | 传播质量 Router，b0–b6 | 0.874083 | 0.813986 | 256×256 |

本次结果支持以下观察：

1. 两条路线的 1,400 条 test 传播全部完成，14 个“路线类型×桥长”平均 Dice 与旧实验完全一致。
2. 按旧报告无学生阶段的 b3–b6 设置，两路线联合选出的最终 mask 达到 **88.10% Dice、81.75% IoU**。
3. 在本次预先指定并完成的六组 Router 设置中，Target pooling 单路线 b0–b6 的测试 Dice 最高，为 **88.54%**。这是测试结果的描述，不代表已经通过新的验证集方案选择确立了最优设置。
4. 单图预测中，`colon polyp` 相比空文本提高 **10.0033 个百分点 Dice**。

两条路线使用有标注 anchor 和伪视频传播；单图 Direct 没有 anchor，并且分辨率不同。因此表中两类结果是不同输入协议的表现，不能把全部差值归因于选路方法。

## 2. 固定数据与模型

### 2.1 数据划分

| 项目 | 数量 | 本次处理 |
|---|---:|---|
| Train | 800 | 沿用原划分，没有重新划分 |
| Validation | 100 | 沿用原划分；Router 读取历史传播结果 |
| Test | 100 | 本次全部实验均评测这同一批图像 |
| 训练集人工标注 supports / anchors | 8 | 沿用原名单 |
| 训练集按无标注使用的图像 | 792 | 原 train 减去原 8 个 supports |

原协议目录：

```text
/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_1pct_anchors/protocol/
├── merged_manifest.jsonl
├── support_manifest.jsonl
├── frozen_labeled_images.txt
└── protocol_summary.json
```

核验结果：1000 个唯一样本 ID；三个 split 的 ID 无交集；8 个 supports 全属于 train；人工标注图像列表与 support manifest 完全一致；清单内图像与 mask 文件均存在。

| 文件 | SHA256 |
|---|---|
| merged_manifest.jsonl | `eb74d0ea534450d7b97ae93319260b7cbbd78e5b74b2b615c777b1263eb91322` |
| support_manifest.jsonl | `15746c3f00a897ed36a2a32cfb1108a697db2e08bff0d99b88af459b6c701dab` |
| frozen_labeled_images.txt | `1968d718cd1877beddbf59ea1c972972e51a75702da646617b10a8ffd2786c0d` |

### 2.2 模型

本次所有 SAM3 推理使用未微调的 base checkpoint：

```text
/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
```

SHA256：`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。

本次没有使用 LoRA e33、X3/X4 学生网络、学生预测或 B7 学生辅助评分。无学生的 Router 实验仍包含一个使用历史 validation 标签拟合的岭回归评分器，不能称为“完全没有拟合任何评分器”。

## 3. 实验 A：两条路线的 test 传播重跑

### 3.1 配置与路线含义

| 项目 | 配置 |
|---|---|
| 选路特征 | 冻结 SAM3-base trunk@256 |
| KNN descriptor | L2-normalized patch mean |
| Beam width | 32 |
| 路线类型 | Target pooling、Patch Correspondence |
| 路线范围 | 两类型均为 b0–b6 |
| 传播模型 | SAM3-base |
| 传播 canvas | 256×256 |
| 推理提示 | 仅 anchor 帧输入 box，`text_str=None` |
| 实际重跑范围 | 仅 test；每类型 100×7=700 条，共 1400 条 |
| 执行方式 | 复用原固定 routes，重新执行 forward 与 return-cycle 推理 |

两类型共用同一个 SAM3-base 传播模型，差异在候选路线构造：

- **Target pooling**：使用标注 anchor 的前景原型，对目标 patch 特征按相似度加权池化，再计算条件分数。
- **Patch Correspondence**：使用 patch 与 anchor 前景原型相似度最高的 8 项均值作为条件分数。

帧序列定义：

```text
b0: [anchor, target]
bn: [anchor, bridge_1, ..., bridge_n, target]
```

b0 仍是包含 anchor 的两帧传播。它与仅输入 target 图像的单图 Direct 不是同一方法。

此次固定路线来自：

```text
/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6/
```

两类型 test 路线 SHA256：

- Target pooling：`89db1347a13699f139cad501b5a6aa0aaee7b1cd6ddb621407dd6e31515e6e6f`
- Patch Correspondence：`15708488c9eb1cb7155e3157de00939e048d50b3958447546556b9a63fbe4e54`

### 3.2 Test 结果

每个表格单元均为 100 张 test 图的平均 Dice。

| 桥长 | Target pooling | Patch Correspondence | 两类型算术平均 |
|---|---:|---:|---:|
| b0 | 0.798144 | 0.706645 | 0.752395 |
| b1 | 0.846623 | 0.754897 | 0.800760 |
| b2 | 0.870032 | 0.788470 | 0.829251 |
| b3 | 0.862253 | **0.859413** | 0.860833 |
| b4 | 0.873930 | 0.857625 | **0.865778** |
| b5 | 0.869552 | 0.855339 | 0.862446 |
| b6 | **0.874004** | 0.853581 | 0.863793 |

“两类型算术平均”是同一桥长下 200 条结果的均值，不是最终 mask 的融合或选路结果。

Target pooling 在所有桥长上的平均 Dice 都更高。测试结果中，Target pooling 最高为 b6，Patch Correspondence 最高为 b3；这里没有根据 test 重新定义推理时的固定桥长。

### 3.3 完成与核验

- 完成时间：2026-09-06 05:14:02（Asia/Shanghai）。
- 两个进程退出码均为 0。
- 每类型 700 条唯一路线均成功，共 1400 条。
- 本次 14 个平均 Dice 与旧结果的差值均为 `0.0`。
- 原数据和原实验结果没有覆盖；本次结果写入独立目录。

## 4. 实验 B：不使用学生网络，得到每张图的最终 mask

### 4.1 从旧报告恢复的方法

旧报告在学生网络之前使用 **Propagation-Quality Router（传播质量选路评分器）**：

```text
SAM3-base 两路线生成候选 mask
    → 提取每条路线的路径与传播质量特征
    → Ridge Router 给候选评分
    → 对每张 target 选分数最高的一个候选
    → 该候选的 SAM3 mask 作为最终 mask
```

最终输出是一个已有候选 mask，没有进行像素投票或 mask 平均，也没有使用学生 mask。

来源：项目 `README.md` 的 `Candidate-Invariant Propagation-Quality Router`，以及 `reproduction_reports/C0_256_base_reproduction.md` 的 Step 2。

### 4.2 评分器与评测协议

- 沿用旧实现 `scripts/analyze_propagation_quality_router.py`，岭回归正则系数 `ridge=1.0`。
- 特征包括桥长、KNN 瓶颈/均值相似度、SAM 分数、回环一致性、面积轨迹、空 mask 数、连通域、质心/框变化和相邻帧 Dice 等。
- 只读取历史 **SAM3-base@256 validation** 传播记录拟合评分器；本次没有重新推理 validation，也没有使用 test 标签拟合。
- 两个单路线实验各自拟合评分器；联合候选池的评分器额外包含模式及其与桥长的交互特征。
- 预先固定 b3–b6 和 b0–b6 两个候选范围，不根据 test 分数改范围或调参数。
- 选路键为 `max(router_score, -bridge_count, route_id)`。
- 实际选路函数不接收 target GT 指标或 target GT 路径；先保存选定的 mask，再对照 GT 计算 Dice/IoU。
- 全部设置均覆盖 100 张 test 图，没有按置信度剔除难例。

候选数量：

| 范围 | 单路线每张 target 候选数 | 两路线联合每张 target 候选数 |
|---|---:|---:|
| b3–b6 | 4 | 8 |
| b0–b6 | 7 | 14 |

### 4.3 最终 mask 结果

| 候选范围 | 候选类型 | Test Dice | Test IoU | GT Oracle（仅分析） |
|---|---|---:|---:|---:|
| b3–b6 | Target pooling | 0.876869 | 0.815768 | 0.902689 |
| b3–b6 | Patch Correspondence | 0.874294 | 0.810286 | 0.895989 |
| b3–b6 | 两路线联合 | **0.881029** | **0.817542** | 0.919337 |
| b0–b6 | Target pooling | **0.885433** | **0.828274** | 0.907426 |
| b0–b6 | Patch Correspondence | 0.861688 | 0.799622 | 0.911715 |
| b0–b6 | 两路线联合 | 0.874083 | 0.813986 | 0.921471 |

GT Oracle 表示事后利用 GT 为每个 target 选出候选池中最佳 mask 的上限，仅用于分析，不是可部署推理结果。

联合选择的模式分布：

| 候选范围 | 选中 Target pooling | 选中 Patch Correspondence |
|---|---:|---:|
| b3–b6 | 81 | 19 |
| b0–b6 | 78 | 22 |

b0–b6 联合候选池的 oracle 高于 b3–b6，但实际 Router Dice 更低，说明增加候选没有自动带来更好的最终排序。单路线与联合路线不仅候选池不同，评分器的拟合输入也不同，差值不能只归因于增加了某一种候选。

### 4.4 保存与核验

六组设置均保存 100 张最终 mask，共 600 张 256×256 二值 PNG；每组包含选路清单、逐图 Dice/IoU 和汇总 JSON。从最终保存 mask 重新计算的 Dice 与对应传播记录一致，独立汇总也与原 Router evaluator 一致。

## 5. 实验 C：SAM3-base 单图空文本 / 有文本对照

### 5.1 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | 未微调 SAM3-base image model |
| 输入 | 当前 test 图像，单图 |
| 输入 / 评测分辨率 | 1008×1008 / 1008×1008 |
| 组 1 | `query_text=""` |
| 组 2 | `query_text="colon polyp"` |
| 类别概率阈值 | 0.5 |
| mask 概率阈值 | 0.5 |
| 多 query 合并 | 保留类别概率达阈值的 queries，对前景概率取最大值后阈值化 |
| 无 query 达阈值时 | 输出空 mask |
| 附加输入 | 无 anchor、桥接、点、框、学生或 Router |

提示词 `colon polyp`（结肠息肉）在测试前按历史 Direct 协议确定，没有在 test 上搜索提示词。

空文本组仍保留 SAM3 接口中的 query 机制。因此它应称为“空文本对照”，不能写成完全删除 query 的无条件自动分割。

### 5.2 Test 结果

| 文本提示 | Test Dice | Test IoU | Dice 中位数 | 非空预测 | 空预测 |
|---|---:|---:|---:|---:|---:|
| 空文本 `""` | 0.361394 | 0.304767 | 0.197243 | 90/100 | 10/100 |
| `colon polyp` | **0.461427** | **0.432159** | **0.263805** | 61/100 | 39/100 |

文本组 Dice 提高 **10.0033 个百分点**。空文本虽然更经常输出非空 mask，但总体 Dice/IoU 更低；非空率本身不能作为分割准确性的替代指标。

`colon polyp` 结果与旧报告的 base Direct 1008+text Dice `0.461427` 一致。

### 5.3 保存与核验

- 两组各保存 100 张最终 mask，共 200 张 1008×1008 PNG。
- 保存逐图指标、有效 query 数、汇总和运行配置。
- 从保存 PNG 与 COCO GT 重新计算逐图 Dice/IoU，最大绝对误差为 `0.0`。
- 两组使用同样模型、阈值与分辨率，唯一实验变量是文本内容。

## 6. 指标口径与结论边界

1. 表内 Dice/IoU 是逐图计算后，在全部 100 张 test 上取算术平均，不是把所有像素合并计算的全局指标。
2. 固定桥长表中的“两类型平均”和 Router 表中的“两路线联合最终 mask”含义不同；后者每张图只输出一张 mask。
3. b0 仍含 anchor；单图 Direct 不含 anchor，两者不能都简单写成“直接预测”。
4. 两路线实验是 canvas=256；本次 Direct 是输入和评测均为 1008。当前结果没有完成统一分辨率下的严格对照。
5. 本次 SAM3 没有微调；Router 使用了历史 validation 的监督信息。未使用学生不等于没有使用标注信息。
6. b3–b6 是按旧报告恢复的无学生候选范围，b0–b6 是预先指定的补充对照；不以本次 test 的最高值反选新的正式协议。
7. 历史记录中的 e33、X3/X4 和 X3+B7 数字没有作为本次新结果。此次不涉及这些模型或评分方法。

## 7. 远端产物索引

### 7.1 路线重跑与无学生最终 mask

```text
/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_kvasir_sam3base_test_20260906/
├── config.json
├── protocol/
├── code/
├── logs/
├── quality_root/
│   ├── sam3enc_anchor_conditioned_target_pooling/
│   └── sam3enc_anchor_conditioned_patch_correspondence/
├── bridge_b0_b6_test_metrics.json
├── bridge_b0_b6_test_metrics.tsv
├── comparison_with_previous.json
├── report.md
├── COMPLETE
├── final_masks_no_student/
│   ├── protocol.json
│   ├── results.json
│   ├── input_sha256.json
│   ├── report.md
│   ├── router_b*_b6_*.json
│   ├── b3_b6/{target_pooling,patch_correspondence,combined}/
│   └── b0_b6/{target_pooling,patch_correspondence,combined}/
└── final_masks_no_student.zip
```

每个 Router 设置子目录均包含 `masks/`、`selected_masks.jsonl`、`per_target_metrics.jsonl` 和 `summary.json`。

关键代码入口：

- 重跑调度：上述目录 `run.py`。
- 传播推理：上述目录 `code/eval_route_propagation_quality.py`。
- 无学生选路：上述目录 `final_masks_no_student/evaluate.py`。
- 原 Router 实现：项目 `scripts/analyze_propagation_quality_router.py`。

### 7.2 Direct 文本对照

```text
/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_kvasir_sam3base_direct_text_ablation_20260906/
├── protocol.json
├── config.yaml
├── code/evaluate_direct.py
├── code/original_evaluator.py
├── logs/
├── empty/
│   ├── masks/
│   ├── per_image.jsonl
│   └── summary.json
├── category/
│   ├── masks/
│   ├── per_image.jsonl
│   └── summary.json
├── verification.json
├── report.md
└── COMPLETE
```

压缩包：

```text
/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_kvasir_sam3base_direct_text_ablation_20260906.zip
```

原单图评测实现：项目 `scripts/eval_sam3_lora_direct_split.py`。虽然文件名包含 LoRA，本次未传入 LoRA 权重，运行时也断言仅使用 base 模型。实验副本只添加输入核验、逐图记录与 mask 导出。

### 7.3 项目 reproduction_reports 中的报告

```text
/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/reproduction_reports/
├── Kvasir_SAM3base_20260906_no_student_final_masks.md
├── Kvasir_SAM3base_20260906_direct_text_ablation.md
└── Kvasir_SAM3base_experiments_summary_20260906.md
```

## 8. 本地交付文件

- 本汇总：`F:/medsam3/kvasir_review_20260906/Kvasir_SAM3base_experiments_summary_20260906.md`
- 路线重跑原表：`F:/medsam3/kvasir_review_20260906/test_rerun_results.md`
- 无学生最终 mask 报告：`F:/medsam3/kvasir_review_20260906/no_student_final_results.md`
- 无学生最终 mask 与逐图结果：`F:/medsam3/kvasir_review_20260906/final_masks_no_student.zip`
- 单图文本对照报告：`F:/medsam3/kvasir_review_20260906/direct_text_ablation_results.md`
- 单图预测 mask 与逐图结果：`F:/medsam3/kvasir_review_20260906/direct_text_ablation_masks_and_results.zip`

重新执行脚本时应使用新的实验输出目录；部分调度和评测脚本会拒绝覆盖已有目录，以保留本次记录。
