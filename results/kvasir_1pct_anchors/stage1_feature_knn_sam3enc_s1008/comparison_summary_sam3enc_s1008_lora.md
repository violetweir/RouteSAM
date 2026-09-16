# KNN 实验汇总（sam3enc_s1008_lora）

协议：SAM3 encoder KNN @1008 features, lora_p491_e20 @256 eval，Kvasir test 100 targets，bridge b0–b6。

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_correspondence | patch_mean | 0.8992 | 0.9129 | 0.9098 | 0.9029 | 0.9047 | 0.9149 | 0.9150 |
| sam3enc_patch_correspondence | cond | 0.8992 | 0.8808 | 0.8794 | 0.8830 | 0.8704 | 0.8735 | 0.8949 |
| sam3enc_patch_correspondence | pooled | 0.8992 | 0.9000 | 0.9135 | 0.8994 | 0.9061 | 0.9066 | 0.9021 |
| sam3enc_target_pooling | patch_mean | 0.8995 | 0.9108 | 0.8991 | 0.8927 | 0.8939 | 0.9082 | 0.8934 |
| sam3enc_target_pooling | cond | 0.8995 | 0.9154 | 0.9032 | 0.9131 | 0.9154 | 0.9136 | 0.9131 |
| sam3enc_target_pooling | pooled | 0.8995 | 0.9005 | 0.8921 | 0.8989 | 0.8970 | 0.9002 | 0.9026 |
| sam3enc_patch_average | patch_mean | 0.8261 | 0.8648 | 0.8728 | 0.8995 | 0.8981 | 0.8958 | 0.8993 |
