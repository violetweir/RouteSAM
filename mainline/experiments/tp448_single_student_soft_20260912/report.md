# 原448张池＋新版单学生软标签训练结果

GPU0，seed2026；448伪标签+8GT，逐图等权，816epoch=31008步。软目标=.75主mask+.25返回合格TP共识；与580张版本复用相同初始权重和共有图像软标签。

|权重|Epoch|Val Dice|Test Dice|Test IoU|
|---|---:|---:|---:|---:|
|best|476|0.828689|0.855191|0.770759|
|final|816|0.806991|0.851452|0.767965|

参考：580张新soft best Test=0.851243，final=0.859530；448张旧S3 best=0.860283，final=0.853202。固定epoch下池缩小，总步数由39984降为31008，因此不是固定优化步数的单因素比较。

路径：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/tp448_single_student_soft_20260912`
