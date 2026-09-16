# KNN 实验汇总（sam3enc_s1008_base）

协议：SAM3 encoder KNN @1008 features, SAM3 base @256 eval，Kvasir test 100 targets，bridge b0–b6。

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_correspondence | patch_mean | 0.8202 | 0.8321 | 0.8302 | 0.8035 | 0.8173 | 0.8247 | 0.8388 |
| sam3enc_patch_correspondence | cond | 0.8202 | 0.8396 | 0.8018 | 0.7944 | 0.8028 | 0.8120 | 0.8084 |
| sam3enc_patch_correspondence | pooled | 0.8202 | 0.8499 | 0.8311 | 0.8033 | 0.8451 | 0.8154 | 0.8408 |
| sam3enc_target_pooling | patch_mean | 0.8205 | 0.8762 | 0.8729 | 0.8701 | 0.8655 | 0.8874 | 0.8771 |
| sam3enc_target_pooling | cond | 0.8205 | 0.8811 | 0.8823 | 0.8617 | 0.8615 | 0.8822 | 0.8639 |
| sam3enc_target_pooling | pooled | 0.8205 | 0.8735 | 0.8758 | 0.8714 | 0.8563 | 0.8643 | 0.8769 |
| sam3enc_patch_average | patch_mean | 0.7698 | 0.8020 | 0.8452 | 0.8672 | 0.8500 | 0.8546 | 0.8550 |
