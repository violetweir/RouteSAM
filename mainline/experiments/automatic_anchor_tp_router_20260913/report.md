# 自动选图＋SAM3-base 单TP b0–b6 Router实验

## 实验范围

复用已冻结的自动参考图及TP候选：Kvasir 8张，ISIC2018 21张。未重新选图或传播，不训练SAM3或学生。验证集分别100/259张，test分别100/260张。Router为每张图选择一个既有mask，不融合、不按质量阈值丢弃test图像。

## 方法与验证

旧式基线：28项路径、SAM置信度、返回一致性、传播形态特征，标准化后用Ridge(alpha=1)回归候选Dice。以自动选图validation候选重新拟合，未直接搬用原参考图的旧权重。新实现与原实现validation评分最大误差小于1e-6，见completion_audit.json。

改进方法rank_peer：增加8项特征——与其余6个mask的Dice均值/最小/最大/标准差、面积、与候选中位面积的绝对差、面积比对数、候选一致性乘返回一致性。将特征与Dice标签在每张图的7个候选内部中心化，学习候选相对质量；只比较alpha=1、10、100。

固定seed2026，按图像分组5折。同图7个候选始终在同折，ISIC沿用此前冻结的5折。按OOF所选mask的平均Dice选择配置，再在全部validation拟合；额外嵌套验证：外层5折、内层用剩余4折选择配置，检查选参稳定性。所有模型与test选择先冻结，之后读取上轮已计算的test指标。此前test候选结果已见过，因此本轮是已使用基准上的实验验证，不是全新盲测；没有根据本轮test成绩修改配置。

## Validation结果

|配置/指标|Kvasir|ISIC2018|
|---|---:|---:|
|legacy_a1|0.851317|0.862396|
|rank_peer_a1|0.829937|0.862954|
|rank_peer_a10|0.834230|0.866155|
|rank_peer_a100|0.835469|0.870530|
|配置选择的嵌套OOF Dice|0.832701|0.864335|

选中配置：Kvasir legacy_a1；ISIC2018 rank_peer_a100。单配置OOF参与了选参，不能将最大OOF值视为无偏泛化估计；嵌套结果更保守。Kvasir嵌套结果低于旧式基线OOF，说明当前选参不稳定。

## 完整Test结果

|方法|Kvasir Dice|Kvasir IoU|ISIC2018 Dice|ISIC2018 IoU|
|---|---:|---:|---:|---:|
|固定b6（validation选定）|0.887069|0.825453|0.852864|0.774872|
|最高返回一致性|0.872943|0.808258|0.861756|0.782495|
|最高候选间一致性|0.876956|0.813704|0.855438|0.777655|
|旧式Router在自动候选上重拟合|0.871374|0.809810|0.866573|0.787929|
|validation选中的Router|0.871374|0.809810|0.866363|0.786488|

|指标|Kvasir|ISIC2018|
|---|---:|---:|
|Oracle Dice|0.917615|0.884316|
|选中Router距Oracle|0.046241|0.017953|

## 结论

Kvasir：选中Router 0.871374，比固定b6低1.5695个百分点。相对固定b6的配对Bootstrap 95%差值区间[-0.037478, 0.001524]；无法证明Router带来收益。这次排序改进在validation上未胜出，不应用test继续追调。

ISIC2018：选中Router 0.866363，比固定b6高1.3499个百分点，95%差值区间[0.004297, 0.024660]。旧式重拟合Router为0.866573，与改进版差0.000210，改进版相对旧式差值区间[-0.007163, 0.006470]，不能声称新的排序方法优于旧式Router。按照预先确定的validation选择，报告rank_peer_a100为选定方案，不因test上旧式略高而改选。

Kvasir的Oracle仍为0.917615，ISIC为0.884316。Router仅选择既有候选，Oracle没有改变。没有对测试图像进行质量删选。

## 文件和启动方式

实验目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/automatic_anchor_tp_router_20260913`

每个数据集下：models_frozen.json保存模型、超参数及冻结时间；folds_frozen.json保存分折；test_choices_frozen.json保存读取test指标前的选择；results.json保存结果；各方法目录的per_target.json及masks/保存逐图指标与最终mask；completion_audit.json为完整性核验。

```bash
/home/violet/anaconda3/envs/sam3/bin/python -u /Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/automatic_tp_router.py
```

上述命令会创建实验目录；现已完成，直接重跑会因目录存在而停止。复现实验应将脚本R变量设为新的输出目录，保留原始结果。脚本快照为run.py。
