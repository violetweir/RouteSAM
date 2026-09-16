# KNN 实验汇总（imr_s256_base）

协议：IMR KNN @256 (SAM3-enc pyramid+contrast+Qwen text), SAM3 base @256 eval，Kvasir test 100 targets，bridge b0–b6。

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_contrast | patch_mean | 0.7502 | 0.7889 | 0.8187 | 0.8204 | 0.8094 | 0.7796 | 0.7680 |
| sam3enc_imr | fused | 0.7981 | 0.8572 | 0.8589 | 0.8674 | 0.8751 | 0.8679 | 0.8624 |
| sam3enc_imr__tb025 | fused | 0.7981 | 0.8417 | 0.8431 | 0.8424 | 0.8508 | 0.8549 | 0.8457 |
| sam3enc_imr__tb050 | fused | 0.7981 | 0.8429 | 0.8342 | 0.8417 | 0.8237 | 0.8443 | 0.8661 |
| sam3enc_imr__tb075 | fused | 0.7981 | 0.8441 | 0.8297 | 0.8504 | 0.8211 | 0.8377 | 0.8570 |
| sam3enc_imr__tb100 | fused | 0.7981 | 0.8547 | 0.8398 | 0.8418 | 0.8552 | 0.8657 | 0.8464 |
| sam3enc_pyramid | patch_mean | 0.8113 | 0.8102 | 0.8374 | 0.8601 | 0.8714 | 0.8618 | 0.8743 |
| sam3enc_pyramid_contrast | patch_mean | 0.8014 | 0.7935 | 0.8233 | 0.8646 | 0.8692 | 0.8506 | 0.8464 |
