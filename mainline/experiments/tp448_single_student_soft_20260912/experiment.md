# 原448张训练池＋新版单学生软标签配方

日期：2026-09-12。本次只训练一个soft学生，GPU0，seed2026。

## 要回答的问题

把新版单学生的训练配方用于原448张高质量池，是否优于使用580张池，以及是否超过448张旧S3？旧S3本身也使用软标签，但其共识构造、6+6采样和加权损失与本次不同。

## 固定配方

|设置|本次448张|既有580张对照|
|---|---|---|
|训练图像|原448张伪标签＋原8GT|580张伪标签＋原8GT|
|主mask|原448张选中mask保持不变|共有448张完全相同|
|软目标|Y=.75M+.25P；P仅平均返回一致性≥.95的TP候选|相同；直接复用共有图像标签|
|学生|同一U-Net、同一保存的初始化权重|相同|
|采样|每epoch统一打乱，每图各一次|相同|
|权重|样本/像素等权，无GT重采样、无额外伪标签系数、无ramp|相同|
|损失|逐图BCE＋Soft Dice后batch平均|相同|
|Batch|12|12|
|训练长度|816epoch|816epoch|
|每轮更新|38|49|
|总更新|31,008|39,984|
|学习率|SGD .01，按epoch线性衰减；momentum .9，weight decay 1e-4|相同|
|输入/指标|256×256，原NEAREST处理|相同|
|验证|每4epoch完整val100，按最高Val Dice选择best|相同|
|测试|训练完成后冻结best和final，两者各测完整test100|best/final已测|

共有图像保留其580张清单中的原索引，用于逐图增强随机种子，避免仅因删去132张图而改变共有图像的增强。每轮shuffle和batch组合会随池大小改变。

这是固定epoch的池大小配方对照，不能把差异全部归因于标签质量，因为总优化步数从39984降为31008。没有额外启动固定步数对照或第二个训练任务。

## 运行前核查

检查原448张身份、448张主mask与原记录一致；复算所有448张软目标并核对量化误差；核对数据哈希、初始化权重、共有图像增强结果、数据集互斥以及真实batch12 CUDA前向/反向。正式训练从原初始化重新加载，不使用预检查后的参数。

## 参考结果

|模型|Val-best Test Dice|Final Test Dice|
|---|---:|---:|
|448张旧S2|0.846007|0.838730|
|448张旧S3|0.860283|0.853202|
|580张新soft|0.851243|0.859530|
|本次448张新soft|等待训练完成|等待训练完成|

## 服务器位置

实验根目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/tp448_single_student_soft_20260912`

- `logs/train_soft.log`：训练日志。
- `runs/soft/status.json`：当前epoch、loss及最佳validation。
- `runs/soft/train_epochs.jsonl`：每轮学习率、样本数及更新步数。
- `runs/soft/validation.jsonl`：每次验证结果。
- `data/train_manifest.jsonl`：456张训练图及监督标签。
- `preflight.json`、`PREFLIGHT_OK`：启动核查。
- 完成后：`FROZEN_BEFORE_TEST.json`、`results.json`、`report.md`、`completion_audit.json`、`COMPLETE`。

旧448/580张实验、图像、标签和权重均保持只读；本次产物单独保存。
