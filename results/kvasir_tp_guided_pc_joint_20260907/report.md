# Kvasir：TP-guided Patch Correspondence 统一选路实验

SAM3-base@256；原800/100/100 split和8个supports；无学生。每版本单一route family，b0-b6每图7个候选，测试覆盖全部100张。

Joint-v1=.75 z(TP)+.25 z(原Top8 PC)。Joint-v2=.70 z(TP)+.30 z(TP-attention加权的局部多patch对应)。tau=10，local top-r=min(3,前景anchor patch数)。区域TP使用原实现的归一化pooled descriptor。

z=(s-median)/(MAD+1e-6)，每anchor/分数组件仅由训练图像拟合，排除anchor自身。两Joint版本的转移余弦也按训练图像对median/MAD标准化，避免混合量纲。原patch-mean KNN提案排序保持，联合分数进入anchor与完整beam路径评分。

Validation报告5折target-level OOF Dice。最终Router使用全部100张validation拟合并冻结，再评test；两个Joint均按用户要求测试，不按test选择权重。

| 方法 | Val OOF Dice | Test Dice | Test IoU | Test Oracle |
|---|---:|---:|---:|---:|
| tp_baseline | 0.851755 | 0.885433 | 0.828274 | 0.907426 |
| joint_v1 | 0.818057 | 0.850575 | 0.786759 | 0.890218 |
| joint_v2 | 0.823666 | 0.847341 | 0.782477 | 0.885651 |

## Test 固定桥长

| Bridge | TP baseline | Joint-v1 | Joint-v2 |
|---|---:|---:|---:|
| b0 | 0.798144 | 0.738989 | 0.737740 |
| b1 | 0.846623 | 0.774150 | 0.796951 |
| b2 | 0.870032 | 0.775304 | 0.788859 |
| b3 | 0.862253 | 0.789116 | 0.822101 |
| b4 | 0.873930 | 0.796501 | 0.839890 |
| b5 | 0.869552 | 0.817148 | 0.820074 |
| b6 | 0.874004 | 0.852381 | 0.836326 |

Oracle只用GT事后分析，不能作为实际预测性能。最终mask为单一路线候选中Router最高分mask，未合并独立TP/PC候选池。全部保存mask已按原GT重新计算并核对Dice；本轮未做train传播。

Anchor patch数量差异仍可能影响局部相似度分布；训练分布median/MAD校准是受测方案，不预设已解决anchor偏向。这个三组对照没有单独隔离“仅校准TP”的效果，不能把Joint对TP的全部差值都归于局部对应。

完整产物：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_tp_guided_pc_joint_20260907`。各版本final_test_masks、selected_test_masks.jsonl、test_per_target_metrics.jsonl、final_router.json均保留。

## 配对 Test 差异分析

以同一张 test 图为配对单位，与 TP baseline 比较；固定 seed=2026、10000 次图像 bootstrap。区间仅描述当前 100 张 test 的抽样不确定性，不用于调参。

| 方法 | Dice 差值 | 95% bootstrap 区间 | 改善/持平/下降图数 |
|---|---:|---:|---:|
| joint_v1 | -0.034858 | [-0.072235, -0.001118] | 40/2/58 |
| joint_v2 | -0.038092 | [-0.079385, -0.002880] | 45/0/55 |

## Anchor 分布诊断

| 方法 | Test 700 候选中最多的 anchor 占比 | 最终 100 masks 中最多的 anchor 占比 |
|---|---:|---:|
| tp_baseline | 56.71% | 56.00% |
| joint_v1 | 21.14% | 21.00% |
| joint_v2 | 19.71% | 20.00% |

## 本轮结论与解释边界

两组均于2026-09-07 01:38前后完成；全流程于01:38:47（UTC+8）结束。三个版本各完成验证700条和测试700条成功路线，最终每版本100个test masks，保存PNG的Dice与逐路线评估核验一致。

Joint-v1与Joint-v2相对TP的test Dice分别下降0.034858、0.038092（3.49、3.81个百分点）。本轮没有证据支持用这两个Joint版本替换原TP路线。

两组test Oracle也分别降至0.890218和0.885651，低于TP的0.907426，因此下降不仅是Router未选好，候选池本身的最好可达平均Dice也降低。单anchor占比降低只说明选路分布变得分散，不代表anchor质量提高。

b0无桥路线的Dice已经从TP的0.798144降到两组约0.739，因此问题至少涉及新的anchor/条件分数选择，而不能完全归因于桥路径。具体是否由每anchor标准化、融合权重或局部对应导致，当前对照无法拆分；b1-b6还共同改变了转移分数标度。

若继续研究，建议先在验证集隔离“仅TP校准”及“固定原TP anchor后加入局部对应”两个因素，再决定后续测试方案。本轮未追加这些新实验，也未用test结果调整本轮参数。

数值核验：原TP/Top8分数全1000张精确复现；新局部对应使用float64计算、float32存储，与独立NumPy参考误差1.30343e-8。初次混合精度核验失败在传播前已修复，详情见precision_audit_note.md和保留的attempt_01目录。
