# Kvasir：TP-guided PC 统一 KNN 选路实验

日期：2026-09-07。当前文件为执行前固定方案，结果另行记录。

## 数据与模型

- 沿用原 train/validation/test = 800/100/100，原 8 张标注、792 张无标注训练划分不变。
- SAM3-base 原 checkpoint，特征与传播分辨率 256；不训练学生网络。
- 训练 RGB 仅用于分数校准与桥节点；不进行训练集 mask 传播。
- 验证集用于拟合历史 Ridge Router；两个 Joint 版本均测试全部 100 张 test。
- 每个版本仅一类路线，b0–b6 每张 7 个候选，不合并两类独立候选池。

## 固定对照

1. **TP baseline**：原 Target Pooling 分数、原转移余弦、原 beam 规则，复用已完成的同路线传播结果，并核对重建路线与最终指标。
2. **Joint-v1**：`0.75 z(TP) + 0.25 z(原 PC Top8)`。
3. **Joint-v2**：`0.70 z(TP) + 0.30 z(TP-guided local PC)`。

`w_j = softmax(10 * p_a^T x_j)`。TP 使用历史定义 `p_a^T normalize(sum_j w_j x_j)`，以保证基线一致。

Joint-v2 保留 anchor GT 前景的全部独立归一化 patch token。每个 target patch 与这些 token 计算余弦，取最相似的 `r=min(3, anchor 前景 token 数)` 个求平均为 `c_j`；再计算 `local = sum_j w_j c_j`。因此 TP 指引关注位置，局部多 token 对应检验该位置的相似程度。

## 校准与选路

`z=(s-median)/(MAD+1e-6)`，每个 anchor、每种分数组件单独拟合。只使用原训练图像分数，排除 anchor 自身，每 anchor 799 个样本。验证与测试均使用冻结的训练统计量。

联合分数已无量纲，两个 Joint 的桥节点转移余弦也使用全部 319600 对训练图像的全局 median/MAD 标准化；避免在同一个 bottleneck 中混合原始余弦和 z 分数。

原 patch-mean KNN 邻居提案排序保留，条件分数用于同余弦值的次级排序。联合分数进入 anchor 选择以及完整 beam 路径的瓶颈/平均值评分；beam width=32。此次修改的是统一条件分数与路径目标，并非学习新的邻居特征。

## Router 与最终 mask

- 与历史设置相同：Ridge=1.0，特征标准化，无 route family mode bits。
- 验证诊断为 target-level 5 折 OOF，seed=2026；每折只用其余 80 张目标图拟合。
- 每版本最终 Router 使用原 100 张 validation 拟合后冻结，再推理 test。
- 测试选择过程移除所有 GT/evaluation_only 字段；每图选最高 Router 分数的一个候选 mask，不进行像素融合。
- 先写出选择清单，再读取 GT 评估；保存的最终 PNG 独立复算 Dice/IoU。
- Oracle 仅为 GT 事后分析，不代表可部署性能。本轮不根据 test 调整权重或挑选参数。

## 核验与解释边界

先核对全部图像的原 TP/PC 特征分数、局部对应的 NumPy 参考实现、TP 的 1400 条验证/测试路线以及历史最终 Dice=0.8854326463411542。

同 anchor/mask/box/bridges/target、同 checkpoint/画布的已有传播结果可以复用；联合分数与路径分数保留新值。每个新版本需要的其余路线重新推理。

本对照没有单独设置“只校准 TP”一组，不能把 Joint 相对 TP 的全部变化归因于局部对应本身。训练统计校准是否能改善 anchor 偏向，需要从实际分布与结果判断。

远程产物：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_tp_guided_pc_joint_20260907`。

最终报告写入远程 `reproduction_reports/Kvasir_TP_guided_PC_joint_20260907.md`，同时保留所有候选、路由模型、最终 mask 和逐图指标。
