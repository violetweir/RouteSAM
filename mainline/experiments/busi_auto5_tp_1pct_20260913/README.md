# BUSI：自动选图1%＋SAM3-base KNN＋单TP b0–b6＋Router

2026-09-13启动，GPU1，后台主进程3552468。原数据位置：`/Data_8TB/lht/MK-UNet/BUSI/BUSI_split`。

## 数据与标注预算

|划分|图像数|用途|
|---|---:|---|
|train|517|自动选图与桥接图像池|
|val|64|候选质量评估、Router拟合和选参|
|test|66|方案冻结后的最终评估|

按train的1%四舍五入选择5张，实际0.9671%。1%指训练图像的标注预算；与此前实验相同，Router还使用独立validation标注。原始划分不变。

图像与mask一一配对。没有发现跨划分的完全相同图像文件；训练内benign (433)与malignant (145)的图像文件SHA256完全相同，选择算法只允许其中一个进入参考集合。此核查仅排查完全相同文件，未保证不存在近重复或同病例图像。

## 自动选择

只读取train RGB特征，不用类别名称、GT、伪mask或val/test信息选图。使用原图缩放到1008的SAM3-base冻结特征：整图归一化patch均值，结合每图64个固定空间位置patch的局部视觉词直方图；训练64簇词典，train IDF加权、平方根变换和L2归一化。全局/局部余弦各占0.5，greedy facility每步最大化训练集最近参考相似度均值；固定seed2026。与Kvasir/ISIC2018使用同一主选择算法。

5张名单冻结后才读取所选GT，生成参考提示及形态预览。未在BUSI重新调选择超参数。本轮未增加随机选图传播对照，因此单次自动选图的结果不能单独证明优于随机。

## 传播与评估

SAM3-base原始权重；KNN patch_mean输入256，TP路径评分，beam32，b0–b6。桥接图只取train。每条路径执行正向及返回传播，canvas256。val 64×7=448个候选，test 66×7=462个候选。全部候选冻结后计算Dice/IoU与Oracle；不按质量阈值删除test图。未训练学生或LoRA。

## Router

沿用前两套数据的预设4种配置：旧式28项特征Ridge alpha1，以及加入候选间一致性、同图中心化学习相对质量的rank_peer alpha1/10/100。按图像分组5折，以validation OOF选中mask的Dice选择配置，另外使用嵌套验证评估选参稳定性。完整validation拟合模型并冻结test选择后才读取test指标。

对照包含旧式重拟合Router、validation选定Router、validation选定固定桥长、最高返回一致性、最高候选间一致性。Oracle仅是事后候选上限。

## 服务器结果位置

传播与选图根目录：
`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/busi_auto5_tp_1pct_20260913`

- `status.json`：阶段与错误信息。
- `logs/selection_features.log`：1008特征提取。
- `logs/automatic_selection.log`：自动选图。
- `selection/SELECTIONS_FROZEN.json`：固定5张名单。
- `selected/index.html`：所选图像及GT预览。
- `logs/validation_propagation.log`、`logs/test_propagation.log`：传播日志。
- `validation_results.json`、`test_results.json`：各b0–b6及Oracle（完成后生成）。
- `quality_root/sam3enc_anchor_conditioned_target_pooling/`：路径、全部候选mask和质量记录。

Router根目录：
`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/busi_auto5_tp_router_20260913`

完成后含`busi/results.json`、模型、冻结选择以及每种方法的最终mask。

## 启动命令

```bash
/home/violet/anaconda3/envs/sam3/bin/python /Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/busi_auto_tp.py prepare
/home/violet/anaconda3/envs/sam3/bin/python -u /Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/busi_auto5_tp_1pct_20260913/pipeline.py run
```

已后台启动，不要重复启动到同一输出位置。CPU选图使用mkunet_mamba环境的sklearn，SAM3使用sam3环境；子进程自动设置GPU1及所需导入路径，完整命令记录在各`*_process.json`。
## 已完成的自动选择

固定顺序为：benign (3)、benign (125)、benign (305)、malignant (195)、malignant (187)。类别名未参与选择，3张良性和2张恶性是自动选择后的结果。GT面积占比分别为2.67%、1.12%、2.31%、17.71%、12.18%。

选图及KNN特征已完成，val的448条路径已生成，正在GPU1传播；test与Router由主流程接续运行。预览文件已同步至本地`F:\medsam3\BUSI_自动参考图5张\index.html`。
