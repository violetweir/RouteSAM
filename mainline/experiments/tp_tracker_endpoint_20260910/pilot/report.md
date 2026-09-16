# TP 传播端点微调：首轮结果

采用完整 TP 推理接口，以当前模型生成的历史记忆作为固定上下文，仅重算终点并反向传播。
训练池保持 620 张软伪标签 + 8 张 GT，每张目标一次、等权；LoRA rank4/alpha8，仅记忆注意力与解码器；AdamW 1e-5，1 epoch。
这不是整条路径的端到端反向传播。没有重新划分数据，没有使用无标注训练图 GT 或 test GT。

| Validation 指标 | SAM3-base | 微调 epoch1 |
|---|---:|---:|
| TP+B7 Dice | 0.851807 | 0.842634 |
| TP Oracle Dice | 0.869544 | 0.873182 |

按完整 validation B7 选择：epoch 0（0 表示保留 base）。本轮未测试 test。
逐图预测、候选清单、冻结选择、训练及梯度核查位于 `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/tp_tracker_endpoint_20260910/pilot`。
