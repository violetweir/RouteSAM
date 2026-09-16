# 路线选择/路由器正式报告（checkpoint = base）

协议：Kvasir test 100 targets；路线 = 11 变体 × b0–b6（ViT-B @256，beam 32）；SAM3 @256；传播质量（正向 trace + 回传 q_return）由同一 checkpoint 计算；学生审核 q_model 复用 256-px X3/S2/S3 预测。校准/拟合只用 validation，test 一次性报告。

对照：固定 b6（anchor_conditioned_target_pooling__knn_cls）= 0.8573；P1 全池 oracle = 0.9291。

## 1. 方法对比（test selected Dice）

| 方法 | 拟合 | P1 (b0–b6) | gap | P2 (b3–b6) | gap |
|---|---|---:|---:|---:|---:|
| 固定 b6（不选择） | 无 | 0.8126 / 0.1165 | 0.8126 / 0.1083 |
| M1 固定 0.5/0.5（旧默认移植，未校准） | 无 | 0.8432 / 0.0860 | 0.8422 / 0.0788 |
| M2 两信号校准线性 | validation | 0.8417 / 0.0874 | 0.8415 / 0.0794 |
| M3 ridge 路由器 | validation | 0.8816 / 0.0475 | 0.8850 / 0.0360 |
| M4 纯 q_return | 无 | 0.8285 / 0.1006 | 0.8336 / 0.0873 |
| M4 纯 q_multi（T21 空掩码约定） | 无 | 0.8462 / 0.0829 | 0.8459 / 0.0750 |
| M4 纯 SAM score | 无 | 0.8540 / 0.0751 | 0.8663 / 0.0546 |
| 纯 q_multi（B7 空掩码约定） | 无 | 0.8309 / 0.0982 | 0.8267 / 0.0943 |
| 纯 q_model（X3） | 无 | 0.8892 / 0.0399 | 0.8870 / 0.0340 |
| 纯 q_model（三学生均值） | 无 | 0.8945 / 0.0346 | 0.8932 / 0.0278 |
| B7 几何（X3 审核） | 无 | 0.8889 / 0.0402 | 0.8880 / 0.0330 |
| B7 几何（三学生均值审核） | 无 | 0.8909 / 0.0382 | 0.8833 / 0.0376 |
| 三信号校准线性（X3） | validation | 0.8892 / 0.0399 | 0.8879 / 0.0331 |
| 三信号校准线性（三学生均值） | validation | 0.8899 / 0.0392 | 0.8679 / 0.0531 |

## 2. 最优配置

- **q_model_mean_only** @ P1_b0_b6：selected **0.8945**（oracle 0.9291，gap 0.0346）

## 3. 最优配置 vs 固定 b6（P1_b0_b6，test）

- 胜 72 / 平 1 / 负 27；平均增益 0.0819

### 最大提升（top 10）

| target | 选择 | 固定 b6 | oracle | 增益 |
|---|---:|---:|---:|---:|
| cju6uzxk0v83p0801rcwnexdu | 0.9513 | 0.0000 | 0.9722 | 0.9513 |
| cju85plp7lmkw0850rx42jdpf | 0.8868 | 0.0000 | 0.9680 | 0.8868 |
| cju2tjrog4jy30878pawyazqc | 0.9684 | 0.1017 | 0.9729 | 0.8668 |
| cju40jl7skiuo0817p0smlgg8 | 0.8179 | 0.1577 | 0.8300 | 0.6602 |
| cjyztzaqtrv430848l8xgcerw | 0.9248 | 0.2716 | 0.9248 | 0.6532 |
| cju5u6wf0kh1t0755bg1ssixv | 0.9120 | 0.2767 | 0.9694 | 0.6353 |
| cju43in5fm22c08175rxziqrk | 0.9678 | 0.3492 | 0.9813 | 0.6185 |
| cju8b4ja9r2s808509d45ma86 | 0.8928 | 0.3705 | 0.9681 | 0.5222 |
| cju30ajhw09sx0988qyahx9s8 | 0.8031 | 0.3043 | 0.9072 | 0.4988 |
| cju171py4qiha0835u8sl59ds | 0.9035 | 0.4269 | 0.9035 | 0.4766 |

### 最大回退（top 10）

| target | 选择 | 固定 b6 | oracle | 增益 |
|---|---:|---:|---:|---:|
| cju7ctvqn25dy08186g442m1r | 0.7136 | 0.9451 | 0.9554 | -0.2315 |
| cju1cyjb5qtie0993njqne9m3 | 0.7145 | 0.9428 | 0.9733 | -0.2283 |
| cju7d2q1k27nf08715zshsckt | 0.7686 | 0.9412 | 0.9633 | -0.1726 |
| cju7eea9b2m0z0801ynqv1fqu | 0.4537 | 0.5669 | 0.6024 | -0.1133 |
| cju8c6hnxsdvr0801wn0vrsa6 | 0.5044 | 0.6045 | 0.6833 | -0.1001 |
| cju5hjxaae3i40850h5z2laf5 | 0.8782 | 0.9754 | 0.9786 | -0.0973 |
| cju1c0qb4tzi308355wtsnp0y | 0.7557 | 0.8487 | 0.9156 | -0.0929 |
| cju8aj01yqeqm0850lhdz3xdw | 0.9289 | 0.9646 | 0.9721 | -0.0357 |
| cju85mpuglq8k0818d2it6hzb | 0.8456 | 0.8810 | 0.9209 | -0.0354 |
| cju2u4pymvc720988wsxrmi84 | 0.9030 | 0.9346 | 0.9564 | -0.0316 |

## 4. 最优配置选中的路线分布（P1_b0_b6）

| family | b0 | b1 | b2 | b3 | b4 | b5 | b6 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor_conditioned_patch_correspondence | 1 | 0 | 1 | 2 | 2 | 1 | 1 | 8 |
| anchor_conditioned_patch_correspondence__knn_cls | 0 | 0 | 2 | 0 | 1 | 1 | 1 | 5 |
| anchor_conditioned_patch_correspondence__knn_cond | 0 | 1 | 2 | 1 | 1 | 2 | 0 | 7 |
| anchor_conditioned_patch_correspondence__knn_pooled | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| anchor_conditioned_target_pooling | 0 | 1 | 2 | 0 | 3 | 1 | 0 | 7 |
| anchor_conditioned_target_pooling__knn_cls | 0 | 2 | 2 | 0 | 0 | 1 | 1 | 6 |
| anchor_conditioned_target_pooling__knn_cond | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 3 |
| anchor_conditioned_target_pooling__knn_pooled | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 4 |
| dino_global_pooling | 2 | 3 | 2 | 3 | 2 | 7 | 2 | 21 |
| dino_patch_average | 1 | 1 | 4 | 0 | 4 | 1 | 4 | 15 |
| t18_corrected | 2 | 4 | 5 | 3 | 5 | 2 | 2 | 23 |

## 5. 信号判别力（spearman vs test GT Dice）

| 信号 | P1 | P2 |
|---|---:|---:|
| q_cycle | 0.1283 | 0.0346 |
| q_multi | 0.6554 | 0.6451 |
| q_model_x3 | 0.7591 | 0.7411 |
| q_model_mean | 0.7393 | 0.7223 |

## 6. 固定公式方法在 validation 上的表现（稳定性参考）

| 方法 | P1 validation | P2 validation |
|---|---:|---:|
| B7_X3 | 0.8619 | 0.8641 |
| B7_mean | 0.8560 | 0.8526 |
| q_multi_only | 0.8140 | 0.8152 |
