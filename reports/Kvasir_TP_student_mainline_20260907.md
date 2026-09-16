# Kvasir TP-only 学生主线完成报告

新基线：SAM3-base TP b0-b6独立Router，test Dice0.885433。原划分与8标注/792无标注身份保持；候选为同一TP-only七路线。

按C0文档完成伪标签、S2/S3、四成员委员会、X3和B7。X3只按普通validation Dice选择best checkpoint；B7公式固定，test全100张，不作置信度过滤。

| 方法 | Test Dice | Test IoU |
|---|---:|---:|
| TP Router基线 | 0.885433 | 0.828274 |
| X3学生单图 | 0.867161 | 0.786951 |
| X3-best + TP-only B7 | 0.892639 | 0.834468 |

B7候选Oracle（仅分析）：0.907426。

完整产物：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_tp_student_mainline_20260907`。训练、验证、模型、逐图预测、最终mask、脚本哈希和事件日志均保留。

## 完成核验

流程于2026-09-07 23:27:42完成。X3在40000次训练中按普通验证Dice选择第24600次checkpoint，权重哈希与测试前冻结清单一致。100个最终test mask均核验SHA256并逐像素重新计算Dice。

与固定TP Router逐图比较：45张改善、17张持平、38张下降；平均差值+0.007207，配对bootstrap 95%区间[-0.005002, +0.025372]。区间仅描述这100张图的抽样不确定性，不用于调参，也不构成未来泛化保证。

训练内X3最佳validation Dice为0.821969；固定checkpoint经历史预测导出流程得到的validation单图Dice为0.824269。这是两个评估入口的结果，不能混写；checkpoint选择依据为前者。

本轮同时采用TP-only新伪标签池及S3的uint16软标签读取修复；与旧双路线学生的差值不能全部归因于路线变化。原无学生TP Router基线保留不覆盖。
