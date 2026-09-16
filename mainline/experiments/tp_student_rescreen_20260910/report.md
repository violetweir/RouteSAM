# TP + 单学生统一重筛：实验完成

按冻结 A/B 条件审核全部 792 张训练图，不保留旧池特权；冻结软标签 validation-best 教师只参与准入。新标签仍为 .75M+.25合格TP共识。

池变化：{'audited': 792, 'candidates': 5544, 'tiers': {'B': 82, 'A': 538, 'C': 172}, 'pseudo_count': 620, 'gt_count': 8, 'train_count': 628, 'previous_pseudo_count': 580, 'retained': 580, 'removed': 0, 'added': 40, 'retained_main_mask_changed': 22, 'teacher_sha256': 'c03b6a63c754df9bc75a6d904bb4b64cabdf7a0e29508b37d8cbbed2a698fc58', 'all_792_rescreened': True, 'no_historical_membership_in_scoring': True, 'student_not_mixed_into_targets': True, 'unlabeled_gt_masks_read': 0}。
统一等权训练816 epoch，batch12，每轮53步，共43248步；复用上一阶段初始权重。

| checkpoint | epoch | validation Dice | test Dice | 相比旧软标签池 test 差异 |
|---|---:|---:|---:|---:|
| best | 576 | 0.826621 | 0.858330 | +0.007086 |
| final | 816 | 0.810647 | 0.869478 | +0.009948 |

validation-best 为主要模型选择规则；final 为预先固定的816epoch终点补充。二者均先冻结再评估test，不按test改筛选门槛。池大小变化导致总iter变化，因此本轮是固定epoch的重筛方案比较。
