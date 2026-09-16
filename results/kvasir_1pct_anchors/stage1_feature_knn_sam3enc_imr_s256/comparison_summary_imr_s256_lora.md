# KNN 实验汇总（imr_s256_lora）

协议：IMR KNN @256 (SAM3-enc pyramid+contrast+Qwen text), lora_p491_e20 @256 eval，Kvasir test 100 targets，bridge b0–b6。

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_contrast | patch_mean | 0.8222 | 0.8505 | 0.8861 | 0.8689 | 0.8858 | 0.8930 | 0.8876 |
| sam3enc_imr | fused | 0.8482 | 0.8752 | 0.8766 | 0.9040 | 0.9033 | 0.9019 | 0.9008 |
| sam3enc_imr__tb025 | fused | 0.8482 | 0.8705 | 0.8940 | 0.9000 | 0.8889 | 0.9030 | 0.9059 |
| sam3enc_imr__tb050 | fused | 0.8482 | 0.8746 | 0.8781 | 0.8968 | 0.9074 | 0.9060 | 0.9110 |
| sam3enc_imr__tb075 | fused | 0.8482 | 0.8689 | 0.8779 | 0.8961 | 0.8920 | 0.8922 | 0.9028 |
| sam3enc_imr__tb100 | fused | 0.8482 | 0.8699 | 0.8829 | 0.8702 | 0.8838 | 0.8877 | 0.8982 |
| sam3enc_pyramid | patch_mean | 0.8376 | 0.8597 | 0.8742 | 0.8974 | 0.8902 | 0.9027 | 0.8925 |
| sam3enc_pyramid_contrast | patch_mean | 0.8297 | 0.8640 | 0.8810 | 0.9061 | 0.9022 | 0.8996 | 0.8992 |
