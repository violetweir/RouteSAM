# SAM3 TP 传播端点微调：首轮实验

日期：2026-09-10。服务器目录：
`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/tp_tracker_endpoint_20260910/`

## 当前状态

预检完成，首轮训练已启动。最终性能以 `pilot/results.json` 和 `pilot/report.md` 为准；本文件不把训练启动视为性能验证成功。

## 预检发现与修正

在训练集固定选取 3 张目标，各检查 b0、b6，共 6 条路径，不读取无标注训练图的真实 mask。

- 原公共 TP 推理接口全部复现历史 mask，一致性 Dice 为 1。
- 直接沿用旧 `forward_tracking` 训练接口时，与公共接口的输出一致性 Dice 为 0.941157、0.979372、0.944759、0.897360、0.820794、0.431339。不能把这个接口直接当成当前 TP 基线。
- 源码显示，公共接口会将 anchor 框处理为视觉提示，经过检测、对象初始化、记忆维护与输出处理；旧接口直接把框输入 tracker，执行过程不同。
- 修正方案保留公共接口全部前向行为，截取终点实际使用的特征、对象 ID 和历史状态，在启用梯度时重算同一个终点步骤。
- 修正后，6 条路径的最终 mask 与历史输出一致；终点原始 logits 与重算 logits 的最大绝对差均为 0。记忆注意力和解码器均获得有限、非零梯度。
- 同时关闭预测器长期 BF16 autocast 上下文中的权重缓存，避免推理缓存切断梯度，以及更新后继续使用旧的缓存权重。

预检产物：`preflight/`（旧接口）、`preflight_public_v3/`（修正后）。先前调试失败的记录保留用于追溯。

## 首轮训练协议

目的：先隔离检验传播端点微调是否有效，不同时重新设计伪标签池。

| 项目 | 首轮配置 |
|---|---|
| 初始化 | 原始 SAM3-base |
| 训练池 | 沿用已冻结的 620 张软伪标签 + 8 张 GT |
| 数据划分 | 原 train800 / validation100 / test100；8 个标注身份不变 |
| 训练轮数 | 1 epoch，共 628 次更新 |
| 采样 | 每个目标出现一次、统一打乱、等权，无 GT 过采样 |
| 路径 | 伪标签目标在 b0–b6 间均衡分配；GT 目标使用其他 GT anchor 的 b0 路径 |
| 提示 | 仅 anchor 的 GT 框；bridge/target 无额外提示；无文本提示 |
| 损失 | 终点 BCE + Soft Dice；无 anchor/bridge 辅助损失 |
| 标签 | 伪标签沿用现有软标签；GT 为硬标签；不随路径改变标签 |
| 更新范围 | 记忆注意力与 mask 解码器交叉注意力 LoRA，52 个模块，90,112 个参数 |
| LoRA | rank4、alpha8、dropout0 |
| 优化器 | AdamW，lr=1e-5，weight decay=0.01，梯度裁剪1 |
| 模型行为 | 保持推理行为，公共接口生成当前模型的历史记忆 |
| 反向传播 | 只对终点重算步骤反向传播；历史记忆断开梯度 |

这是长度为 1 的截断反向传播试验，不是整条路径的端到端训练，也不声称已优化所有记忆写入环节。

首选路径数：b0=96、b1=90、b2=89、b3=89、b4=88、b5=88、b6=88。训练池内基线已知为空的 6 条候选路径不作为训练路径。若当前模型在首选路径上没有可监督的跟踪对象，按事先冻结的候选路径顺序尝试下一条；每个目标仍只贡献一次更新。所有路径都失败则报错，不静默丢弃目标。

## 验证和选择

训练后自动执行完整 validation 的 100×7 条固定 TP 路径，重新计算正向 mask 和返回一致性。使用同一个新学生 validation-best，以及原 B7 公式重新选择 mask。全部选择冻结后再用 validation GT 计算 Dice、IoU 和 Oracle。

| 对照指标 | SAM3-base + 新学生 B7 |
|---|---:|
| Validation B7 Dice | 0.851807 |
| Validation Oracle Dice | 0.869544 |

以完整 validation B7 Dice 在 base 与 epoch1 中选择，平分保留 base。本轮不运行 test，不根据 test 调整训练。

## 产物

- `pilot/config.json`：冻结配置与数据、学生哈希。
- `pilot/train_targets.jsonl`：628 个目标、监督标签和预先确定的候选路径。
- `pilot/first_backward.json`：实际训练首步梯度核查。
- `pilot/status.json`、`pilot/train_steps.jsonl`、`pilot/run.log`：状态与日志。
- `pilot/epoch1.pt`：适配器、优化器及随机状态。
- `pilot/train_audit.json`：完整 epoch 覆盖及冻结参数核查。
- `pilot/validation_epoch1/`：候选 mask、冻结 B7 选择及逐图指标。
- `pilot/results.json`、`pilot/report.md`：完成后生成的结果。

本轮的结果用于决定是否延长训练，以及是否进一步引入跨帧反向传播和新的训练池构造规则。
