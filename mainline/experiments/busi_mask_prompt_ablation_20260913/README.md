# BUSI：完整mask提示，有/无文本两轮实验

2026-09-13启动，GPU1，主进程3577483。用户要求将原提示换成mask，并比较有文本与无文本。本轮将其解释为：从参考GT外接框，改为输入参考GT完整mask。

## 固定条件

沿用自动5张参考图、原数据train517/val64/test66及原TP b0–b6所有路径，不重新选图或调路径。SAM3-base权重不变，不训练学生或LoRA。传播canvas及评估分辨率保持256。

两轮为：

1. mask_no_text：参考GT完整mask，无文本。
2. mask_text：参考GT完整mask＋固定英文文本`breast lesion`。不根据图像类别改成良性/恶性，不搜索多种文本。

每轮生成val448及test462个候选；返回一致性以正向预测mask为输入，沿反向排列的RGB路径回到参考图，不读取目标GT。

## 接口实现与核验

SAM3现有公共add_prompt没有mask参数；本实验使用本地独立适配函数，复用官方交互对象注册逻辑，将点提示调用替换为tracker.add_new_mask，并保留用户mask条件帧，避免被原来的点修正清理逻辑误删。

每条路径验证输入mask与跟踪器内存中的原分辨率mask逐像素一致；框和点输入均为空。精确mask条件帧写入后，从下一帧开始传播，输出始终跟踪mask指定的object 0。

两轮均走完整SAM3语义检测＋跟踪流程。有文本组将文本写入官方文本输入，允许文本检测与跟踪匹配；无文本组使用原生无语义提示设置，不新增语义检测。这样避免只跑Tracker而使文本被忽略。记录每条路径的文本特征缓存、对象数及mask输入检查。

最初两条val路径（b0、b6）两种提示输出完全一致。它仅说明这两条路径暂未受文本改变，不代表文本未被编码，也不代表全数据必然相同。完整报告会统计两轮预测的逐候选一致数。

原框版本曾逐帧取最高分对象，新mask版本跟踪指定对象；因此新mask与原框比较同时包含提示方式与对象选择方式的变化，不应声称这是完全纯粹的单变量框/mask对照。新两轮之间固定其他实现，仅切换文本。

## Router及报告

每轮分别用validation做图像分组5折，按此前预设的旧式Ridge和rank_peer四种配置选参，再完整validation拟合，冻结test选择后评估。保留b0–b6 Dice/IoU、Oracle和Router及简单选择对照。

原数据、原SAM3代码、旧BUSI实验均不覆盖。小规模接口调试日志保留，完整任务另存。

## 路径和日志

根目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/busi_mask_prompt_ablation_20260913`

- 总进度：`status.json`。
- 主日志：`pipeline.log`。
- 每轮目录：`mask_no_text/`、`mask_text/`。
- 每轮日志：`logs/smoke.log`、`logs/validation_propagation.log`、`logs/test_propagation.log`、`logs/router.log`。
- 各轮候选：`quality_root/sam3enc_anchor_conditioned_target_pooling/propagation_quality_test/`。
- 各轮Router：根目录下`mask_no_text_router/`、`mask_text_router/`。
- 自动汇总：完成后根目录生成`results.json`、`report.md`、`COMPLETE`。

## 运行入口

```bash
/home/violet/anaconda3/envs/sam3/bin/python -u /Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/busi_mask_prompt_ablation_20260913/pipeline.py run
```

已后台启动，不要重复运行到同一目录。实际子命令保存在各轮的`*_process.json`。