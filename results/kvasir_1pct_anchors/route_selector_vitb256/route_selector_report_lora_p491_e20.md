# 路线选择/路由器正式报告（checkpoint = lora_p491_e20）

协议：Kvasir test 100 targets；路线 = 11 变体 × b0–b6（ViT-B @256，beam 32）；SAM3 @256；传播质量（正向 trace + 回传 q_return）由同一 checkpoint 计算；学生审核 q_model 复用 256-px X3/S2/S3 预测。校准/拟合只用 validation，test 一次性报告。

对照：固定 b6（anchor_conditioned_target_pooling__knn_cls）= 0.8978；P1 全池 oracle = 0.9357。

## 1. 方法对比（test selected Dice）

| 方法 | 拟合 | P1 (b0–b6) | gap | P2 (b3–b6) | gap |
|---|---|---:|---:|---:|---:|
| 固定 b6（不选择） | 无 | 0.8978 / 0.0380 | 0.8978 / 0.0369 |
| M1 固定 0.5/0.5（旧默认移植，未校准） | 无 | 0.8953 / 0.0405 | 0.8906 / 0.0441 |
| M2 两信号校准线性 | validation | 0.8665 / 0.0693 | 0.8704 / 0.0643 |
| M3 ridge 路由器 | validation | 0.8933 / 0.0424 | 0.8943 / 0.0404 |
| M4 纯 q_return | 无 | 0.8665 / 0.0693 | 0.8704 / 0.0643 |
| M4 纯 q_multi（T21 空掩码约定） | 无 | 0.8952 / 0.0405 | 0.8885 / 0.0463 |
| M4 纯 SAM score | 无 | 0.8803 / 0.0554 | 0.8897 / 0.0451 |
| 纯 q_multi（B7 空掩码约定） | 无 | 0.8886 / 0.0471 | 0.8885 / 0.0463 |
| 纯 q_model（X3） | 无 | 0.8927 / 0.0431 | 0.8940 / 0.0408 |
| 纯 q_model（三学生均值） | 无 | 0.8948 / 0.0410 | 0.8949 / 0.0398 |
| B7 几何（X3 审核） | 无 | 0.8983 / 0.0375 | 0.9022 / 0.0325 |
| B7 几何（三学生均值审核） | 无 | 0.9014 / 0.0344 | 0.8921 / 0.0426 |
| 三信号校准线性（X3） | validation | 0.8954 / 0.0403 | 0.8982 / 0.0365 |
| 三信号校准线性（三学生均值） | validation | 0.9035 / 0.0323 | 0.8704 / 0.0643 |

## 2. 最优配置

- **lin_cal_mean** @ P1_b0_b6：selected **0.9035**（oracle 0.9357，gap 0.0323）
- validation 校准权重：q_return 0.80 / q_multi 0.05 / q_model 0.15（validation selected 0.8711）

## 3. 最优配置 vs 固定 b6（P1_b0_b6，test）

- 胜 43 / 平 0 / 负 57；平均增益 0.0057

### 最大提升（top 10）

| target | 选择 | 固定 b6 | oracle | 增益 |
|---|---:|---:|---:|---:|
| cju183od81ff608017ekzif89 | 0.9578 | 0.4873 | 0.9653 | 0.4705 |
| cju7ctvqn25dy08186g442m1r | 0.9534 | 0.6619 | 0.9541 | 0.2915 |
| cju5kre09fhka0850h7b1898j | 0.9716 | 0.7861 | 0.9828 | 0.1855 |
| cju2zkpdl9h7t0799ix60teqg | 0.7809 | 0.6579 | 0.7834 | 0.1229 |
| cju7ebe962hr409872ovibahw | 0.4253 | 0.3282 | 0.4301 | 0.0972 |
| cju1c0qb4tzi308355wtsnp0y | 0.7879 | 0.7312 | 0.9152 | 0.0566 |
| cju83qd0yjyht0817ktkfl268 | 0.8375 | 0.7837 | 0.9744 | 0.0538 |
| cju8b4ja9r2s808509d45ma86 | 0.9593 | 0.9301 | 0.9657 | 0.0292 |
| cju8czvnztbf40871b4m7t78w | 0.9687 | 0.9464 | 0.9780 | 0.0224 |
| cju2yb31a8e8u0878wdashg7o | 0.9411 | 0.9211 | 0.9850 | 0.0200 |

### 最大回退（top 10）

| target | 选择 | 固定 b6 | oracle | 增益 |
|---|---:|---:|---:|---:|
| cju32phw2bv130801yj7bkouq | 0.5690 | 0.7571 | 0.7716 | -0.1880 |
| cjyzk8qieoboa0848ogj51wwm | 0.6755 | 0.7830 | 0.8176 | -0.1075 |
| ck2bxskgxxzfv08386xkqtqdy | 0.8706 | 0.9594 | 0.9654 | -0.0888 |
| cju1cyjb5qtie0993njqne9m3 | 0.8477 | 0.9233 | 0.9685 | -0.0756 |
| cju7efffp2ivf0817etg3jehl | 0.9064 | 0.9806 | 0.9822 | -0.0742 |
| cju171py4qiha0835u8sl59ds | 0.6988 | 0.7702 | 0.9327 | -0.0713 |
| cju77g99iyxc00817zqi2ppor | 0.8575 | 0.9066 | 0.9331 | -0.0490 |
| cju7d2q1k27nf08715zshsckt | 0.8575 | 0.8962 | 0.9522 | -0.0387 |
| cju5k7r0yf98c09878csbxb4d | 0.9245 | 0.9486 | 0.9614 | -0.0241 |
| cju77q10sz9ug0801449wu1nu | 0.6190 | 0.6399 | 0.7766 | -0.0209 |

## 4. 最优配置选中的路线分布（P1_b0_b6）

| family | b0 | b1 | b2 | b3 | b4 | b5 | b6 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor_conditioned_patch_correspondence | 1 | 0 | 3 | 0 | 1 | 0 | 2 | 7 |
| anchor_conditioned_patch_correspondence__knn_cls | 0 | 0 | 2 | 0 | 1 | 1 | 6 | 10 |
| anchor_conditioned_patch_correspondence__knn_cond | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 3 |
| anchor_conditioned_patch_correspondence__knn_pooled | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| anchor_conditioned_target_pooling | 0 | 1 | 0 | 1 | 1 | 0 | 2 | 5 |
| anchor_conditioned_target_pooling__knn_cond | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| anchor_conditioned_target_pooling__knn_pooled | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| dino_global_pooling | 1 | 0 | 0 | 3 | 1 | 4 | 3 | 12 |
| dino_patch_average | 0 | 3 | 4 | 4 | 3 | 1 | 0 | 15 |
| t18_corrected | 8 | 6 | 13 | 4 | 6 | 0 | 8 | 45 |

## 5. 信号判别力（spearman vs test GT Dice）

| 信号 | P1 | P2 |
|---|---:|---:|
| q_cycle | 0.0697 | 0.0409 |
| q_multi | 0.6800 | 0.6275 |
| q_model_x3 | 0.6866 | 0.6763 |
| q_model_mean | 0.6559 | 0.6437 |

## 6. 固定公式方法在 validation 上的表现（稳定性参考）

| 方法 | P1 validation | P2 validation |
|---|---:|---:|
| B7_X3 | 0.8611 | 0.8645 |
| B7_mean | 0.8531 | 0.8526 |
| q_multi_only | 0.8325 | 0.8437 |
