# BUSI 分数校准与多参考候选实验

固定自动选出的 5 张训练参考图，原 train517 / val64 / test66 划分，SAM3-base，GT box，无文本，256 评价。
校准仅使用 517 张训练 RGB 特征；Router 使用额外的 64 张验证集标注。测试集此前已有诊断分析，本轮参数只在验证集选择。

验证集选定：k2_legacy_a1；OOF Dice=0.738958；完整选择流程 nested OOF=0.709939。

| 方法 | test Dice |
|---|---:|
| selected_router | 0.752198 |
| original_router | 0.566808 |
| calibrated_top1_b0 | 0.705708 |
| original_b0 | 0.458291 |
| calibrated_top1_b1 | 0.709401 |
| original_b1 | 0.514197 |
| calibrated_top1_b2 | 0.679150 |
| original_b2 | 0.514843 |
| calibrated_top1_b3 | 0.689764 |
| original_b3 | 0.533733 |
| calibrated_top1_b4 | 0.662677 |
| original_b4 | 0.517095 |
| calibrated_top1_b5 | 0.698703 |
| original_b5 | 0.514702 |
| calibrated_top1_b6 | 0.718360 |
| original_b6 | 0.539444 |
| calibrated_top1_val_fixed | 0.662677 |

候选池 Oracle 仅作 GT 上界诊断，候选数量不同的 Oracle 分开列出：

- k=0（0 为旧路线），7 候选：Oracle Dice 0.617141
- k=1（0 为旧路线），7 候选：Oracle Dice 0.775461
- k=2（0 为旧路线），14 候选：Oracle Dice 0.823649

每图结果、验证集全部配置、bootstrap 区间及可复现命令见同目录 JSON 和 README.md。
