# Kvasir：8-anchor b0 直接传播诊断

仅 validation 100 张；沿用原 8 个训练 anchor。无学生，无新 test 评估。

## 实际选择结果

| 方法 | Dice | 相对 TP b0 |
|---|---:|---:|
| TP_top1 | 0.773972 | +0.000000 |
| PC_top1 | 0.750490 | -0.023482 |
| eight_anchor_mask_medoid | 0.830269 | +0.056297 |
| eight_anchor_sam_confidence | 0.828685 | +0.054713 |
| OOF_fixed_anchor_training_mean | 0.821163 | +0.047191 |

## 候选上界（GT 事后选择）

| 候选 | Oracle Dice |
|---|---:|
| all8 | 0.886678 |
| TP_top2 | 0.860772 |
| TP_top3 | 0.865779 |
| PC_top2 | 0.807804 |
| TP_top1_PC_alternate | 0.816058 |

上述 b0 结果不能与 b0–b6 Router 直接当作相同预算的机制比较。全 8 anchor 结果用于定位检索瓶颈；top-k Oracle 不是实际可用选择器。

目标内部相似度与真实传播 Dice 的平均 Spearman：

{
  "TP": {
    "mean": 0.08224851021334159,
    "count": 99
  },
  "PC": {
    "mean": -0.0315938939276435,
    "count": 99
  },
  "consensus": {
    "mean": 0.25373141489929313,
    "count": 99
  }
}

全部 800 张 mask 已校验哈希并独立重算 Dice；TP/PC 的 200 个原始 b0 检索选择已复现。

## 置信度字段的新诊断

本轮每个 anchor 在所有非空 target 输出上的 final_sam_score 都只有一个取值；空输出时为 0。它在这里基本区分参考对象，不能解释为目标 mask 的连续质量估计。SAM3 的后处理另算了 out_tracker_probs，但当前返回接口没有导出它；旧脚本 select_top 读取的是 out_probs。后续若研究质量选择，应分清对象分数、逐帧跟踪存在分数与 mask 准确度，不能直接把其中任何一个视为 Dice。当前固定预算对照继续保持旧特征定义，以免把候选机制与选择证据同时改动。

## 当前解释

TP 前两名 anchor 的 b0 Oracle 为 0.860772，高于 TP 首选加 PC 备用的 0.816058；TP/PC 排序与实际传播质量相关性均弱。这个结果支持继续验证多参考图与检索适配性，尚不能替代 b0–b6 的固定预算对照。
