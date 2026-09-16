# 自动选图＋SAM3-base KNN＋单TP b0–b6：Test实验

2026-09-13启动。本轮按用户最新要求，不拟合、不运行Router。

|数据集|固定自动参考图数量|Train / Val / Test|新增Test候选数|
|---|---:|---|---:|
|Kvasir|8|800 / 100 / 100|700|
|ISIC2018|21|2075 / 259 / 260|1820|

沿用2026-09-11自动选图实验已冻结的名单。自动选择使用训练图像RGB的SAM3-base 1008特征；本次不重新选图。选中图像的既有标注用于参考图提示。

KNN使用SAM3-base patch_mean特征，输入256，beam width=32；路径评分使用Target Pooling，桥长b0–b6。中间桥接图只来自train。保留先前train/val特征缓存的原值，只追加test特征。传播canvas=256，含正向预测与返回一致性记录；使用原始SAM3权重，不使用LoRA或学生。

本轮主结果：分别报告b0–b6全部test图像的平均Dice、IoU；额外计算Oracle Dice以衡量候选上限。所有候选预测冻结后才读取test GT计算指标。不会按质量阈值排除test图像，不使用test GT选择实际输出。

原参考图版本使用存档test预测，按相同256分辨率GT重算以作对照；自动参考图版本新增运行。各桥长结果不合并为Router结果。

## 服务器位置

实验根目录：
`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/automatic_anchor_tp_test_20260913`

GPU1顺序运行Kvasir、ISIC2018，保留GPU0其他任务。主进程PID：3500968。

- 总进度：`status.json`
- 总日志：`pipeline.log`
- 各数据集日志：`kvasir/logs/`、`isic2018/logs/`
- 传播日志：各数据集下`logs/test_propagation.log`
- 路径：各数据集下`quality_root/sam3enc_anchor_conditioned_target_pooling/test_pool0_stage1/routes.jsonl`
- 候选mask及质量记录：各数据集下`quality_root/sam3enc_anchor_conditioned_target_pooling/propagation_quality_test/`
- 完成后自动输出：各数据集下`results.json`、`report.md`、`completion_audit.json`和`COMPLETE`

## 启动命令

以下准备命令会创建隔离目录；现已执行，不应在同一路径重复准备。

```bash
/home/violet/anaconda3/envs/sam3/bin/python /Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/automatic_anchor_tp_test_20260913.py prepare
```

主运行入口已通过后台进程启动，不要重复启动：

```bash
/home/violet/anaconda3/envs/sam3/bin/python /Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/automatic_anchor_tp_test_20260913/pipeline.py run
```

主脚本为各子任务设置CUDA_VISIBLE_DEVICES=1和所需PYTHONPATH；完整实际子命令记录在各数据集的`*_process.json`中。