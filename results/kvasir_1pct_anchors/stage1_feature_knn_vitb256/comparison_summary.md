# KNN 实验汇总（ViT-B @256 / SAM3 @256）

协议：Kvasir test 100 targets，冻结 SAM3 `sam3.pt`，forward-only Dice @ canvas 256，DINOv3 特征输入 256×256（grid 16），beam width 32，bridge b0–b6。

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | patch_mean | 0.7421 | 0.7817 | 0.7942 | 0.8340 | 0.8311 | 0.8357 | 0.8573 |
| dino_global_pooling | patch_mean | 0.7000 | 0.7849 | 0.8182 | 0.8059 | 0.7984 | 0.8124 | 0.8282 |
| dino_patch_average | patch_mean | 0.7325 | 0.7634 | 0.8350 | 0.8277 | 0.8001 | 0.8109 | 0.8121 |
| target_pooling | patch_mean | 0.6893 | 0.7691 | 0.7960 | 0.8052 | 0.7736 | 0.7734 | 0.7739 |
| target_pooling | cls | 0.6893 | 0.7777 | 0.8104 | 0.7909 | 0.7790 | 0.7978 | 0.8126 |
| target_pooling | pooled | 0.6893 | 0.7721 | 0.8090 | 0.7982 | 0.7927 | 0.7770 | 0.7833 |
| target_pooling | cond | 0.6893 | 0.7607 | 0.7831 | 0.7573 | 0.7493 | 0.7725 | 0.7608 |
| patch_correspondence | patch_mean | 0.7035 | 0.7862 | 0.8330 | 0.8103 | 0.7964 | 0.8097 | 0.8081 |
| patch_correspondence | cls | 0.7035 | 0.7960 | 0.8100 | 0.7986 | 0.8117 | 0.8117 | 0.8355 |
| patch_correspondence | pooled | 0.7035 | 0.7912 | 0.8185 | 0.7960 | 0.8049 | 0.8019 | 0.8090 |
| patch_correspondence | cond | 0.7035 | 0.7768 | 0.8123 | 0.7867 | 0.7862 | 0.8003 | 0.8230 |

## 旧参考（vits16 @224 + SAM3 @512，仅粗参考，分辨率不同不可直接对比）

| mode | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | 0.7479 | 0.7759 | 0.7565 | 0.7858 | 0.8217 | 0.8168 | 0.8192 |
| dino_global_pooling | 0.7408 | 0.7483 | 0.7583 | 0.7868 | 0.7775 | 0.8184 | 0.8186 |
| dino_patch_average | 0.7286 | 0.7269 | 0.7569 | 0.7468 | 0.7998 | 0.8292 | 0.8263 |
| anchor_conditioned_target_pooling | 0.7413 | 0.7602 | 0.8272 | 0.8408 | 0.8413 | 0.8486 | 0.8546 |
| anchor_conditioned_patch_correspondence | 0.7578 | 0.7735 | 0.8223 | 0.8328 | 0.8340 | 0.8336 | 0.8421 |
