# BUSI 自动选图1%：最终结果

实验完成。原train517/val64/test66不变，自动5张参考图，训练标注比例0.9671%。validation标注额外用于Router。选图RGB1008；SAM3-base KNN patch_mean256、TP、beam32、b0–b6、传播及评估256。没有SAM3微调或学生训练。

|桥长|Val Dice|Test Dice|Test IoU|
|---|---:|---:|---:|
|b0|0.494374|0.458291|0.375306|
|b1|0.560873|0.514197|0.429142|
|b2|0.590754|0.514843|0.431707|
|b3|0.608827|0.533733|0.446990|
|b4|0.620961|0.517095|0.432910|
|b5|0.644838|0.514702|0.435170|
|b6|0.604237|0.539444|0.454572|

Val Oracle：0.675247；Test Oracle：0.617141。

## Router与对照

配置只由validation图像分组交叉验证选择，选中了旧式Ridge alpha1。rank_peer三个配置没有胜出。validation旧式OOF Dice为0.617634；选参策略的嵌套OOF为0.609696。val固定最好为b5（0.644838），说明Router在验证阶段并没有超过固定b5；test上的改善不应被解释为跨划分稳定收益。

|方法|Test Dice|Test IoU|
|---|---:|---:|
|选中Router/旧式重拟合|0.566808|0.480026|
|val选定固定b5|0.514702|0.435170|
|最大返回一致性|0.530451|0.437252|
|最大候选间一致性|0.547502|0.460090|

Router比固定b5高5.2106个百分点，配对Bootstrap差值95%区间[0.012606,0.101015]。Router与Oracle仍差5.0333个百分点。b6是事后观察的test固定最好（0.539444），不据此改选桥长。

## 路径源头诊断

自动选择5张：benign (3)、benign (125)、benign (305)、malignant (195)、malignant (187)。GT未用于选择。

|参考图|Val路径数/448|Test路径数/462|
|---|---:|---:|
|BUSI::benign (3)|8|0|
|BUSI::benign (125)|0|0|
|BUSI::benign (305)|0|0|
|BUSI::malignant (195)|5|7|
|BUSI::malignant (187)|435|455|

malignant (187)占val路径97.10%，test路径98.48%。自动参考图在外观上多样，并没有转化为实际传播源头的多样性。该现象说明需要检查TP的跨参考评分和路径选择是否偏向某个参考；单凭集中度还不能证明它是性能较低的唯一原因。

当前test Oracle仅0.617141，表明候选生成质量本身也是瓶颈，不能仅依靠Router解决。下一步可在validation中比较每参考分别生成候选与跨参考评分校准，并核查SAM3在超声图上的直接提示预测；本次未自动启动额外实验。没有随机选图传播对照，暂不能判断自动5张优于随机5张。

## 完整性

val448及test462条候选全部完成；没有质量删图。Router使用所有66张test。预测mask哈希检查、旧Router实现等价性检查及模型/选择冻结检查通过。原数据与历史实验未修改。

传播目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/busi_auto5_tp_1pct_20260913`

Router目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/busi_auto5_tp_router_20260913`

模型与最终mask：Router目录下`busi/models_frozen.json`、`busi/selected_router/masks/`。完整逐图结果在`busi/selected_router/per_target.json`。
