# 等权单学生：final checkpoint 的 test 补充评估

按用户要求评估既有第 816 epoch / 39,984 iter 的 final checkpoint；未重新训练、未按 test 重新选择模型。输入预处理、阈值和逐图指标与此前 best 评估相同。

| 版本 | Best test Dice | Final test Dice | Final test IoU | Final validation Dice |
|---|---:|---:|---:|---:|
| hard | 0.848592 | 0.854011 | 0.775123 | 0.812377 |
| soft | 0.851243 | 0.859530 | 0.783362 | 0.814561 |

两组均覆盖完整 100 张 test。两个 final checkpoint 均复现训练末期的 validation Dice；checkpoint 哈希和全部 200 张输出 mask 哈希核查通过。

输出目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/single_student_hard_soft_20260909/final_checkpoint_test`；逐图结果及 mask 位于 `runs/hard/test_final` 和 `runs/soft/test_final`。
