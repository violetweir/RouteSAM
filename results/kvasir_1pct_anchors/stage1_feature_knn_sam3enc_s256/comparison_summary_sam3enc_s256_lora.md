# KNN 实验汇总（sam3enc_s256_lora）

协议：SAM3 encoder KNN @256 features, lora_p491_e20 @256 eval，Kvasir test 100 targets，bridge b0–b6。

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_correspondence | patch_mean | 0.7647 | 0.8680 | 0.8722 | 0.8886 | 0.8997 | 0.9036 | 0.8994 |
| sam3enc_patch_correspondence | cond | 0.7647 | 0.8679 | 0.8739 | 0.8983 | 0.9082 | 0.8809 | 0.9012 |
| sam3enc_patch_correspondence | pooled | 0.7647 | 0.8683 | 0.8719 | 0.8849 | 0.8913 | 0.8985 | 0.8939 |
| sam3enc_target_pooling | patch_mean | 0.8482 | 0.8528 | 0.8659 | 0.8730 | 0.8927 | 0.8879 | 0.8960 |
| sam3enc_target_pooling | cond | 0.8482 | 0.8640 | 0.8277 | 0.8779 | 0.8871 | 0.8946 | 0.8988 |
| sam3enc_target_pooling | pooled | 0.8482 | 0.8597 | 0.8616 | 0.8620 | 0.8944 | 0.8901 | 0.8985 |
| sam3enc_patch_average | patch_mean | 0.8222 | 0.8505 | 0.8861 | 0.8689 | 0.8858 | 0.8930 | 0.8876 |
