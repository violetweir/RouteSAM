# KNN 实验汇总（sam3enc_s256_base）

协议：SAM3 encoder KNN @256 features, SAM3 base @256 eval，Kvasir test 100 targets，bridge b0–b6。

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_correspondence | patch_mean | 0.7066 | 0.7549 | 0.7885 | 0.8594 | 0.8576 | 0.8553 | 0.8536 |
| sam3enc_patch_correspondence | cond | 0.7066 | 0.7355 | 0.8357 | 0.8362 | 0.8499 | 0.8484 | 0.8318 |
| sam3enc_patch_correspondence | pooled | 0.7066 | 0.7458 | 0.8030 | 0.8652 | 0.8618 | 0.8505 | 0.8603 |
| sam3enc_target_pooling | patch_mean | 0.7981 | 0.8466 | 0.8700 | 0.8623 | 0.8739 | 0.8696 | 0.8740 |
| sam3enc_target_pooling | cond | 0.7981 | 0.8534 | 0.8647 | 0.8841 | 0.8461 | 0.8698 | 0.8462 |
| sam3enc_target_pooling | pooled | 0.7981 | 0.8391 | 0.8466 | 0.8493 | 0.8765 | 0.8657 | 0.8529 |
| sam3enc_patch_average | patch_mean | 0.7502 | 0.7889 | 0.8187 | 0.8204 | 0.8094 | 0.7795 | 0.7680 |
