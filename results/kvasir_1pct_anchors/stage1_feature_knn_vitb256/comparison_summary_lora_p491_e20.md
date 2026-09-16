# KNN 实验汇总（lora_p491_e20）

协议：Kvasir test, checkpoint=lora_p491_e20 (LoRA p491 e20), forward-only Dice @ canvas 256, DINOv3 @256，Kvasir test 100 targets，bridge b0–b6。

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | patch_mean | 0.8424 | 0.8859 | 0.8791 | 0.8815 | 0.8794 | 0.8831 | 0.8880 |
| dino_global_pooling | patch_mean | 0.8142 | 0.8498 | 0.8794 | 0.8786 | 0.8665 | 0.8775 | 0.8642 |
| dino_patch_average | patch_mean | 0.8197 | 0.8616 | 0.8650 | 0.8816 | 0.8666 | 0.8718 | 0.8768 |
| target_pooling | patch_mean | 0.8030 | 0.8531 | 0.8574 | 0.8467 | 0.8610 | 0.8724 | 0.8888 |
| target_pooling | cls | 0.8030 | 0.8324 | 0.8443 | 0.8283 | 0.8762 | 0.8819 | 0.8978 |
| target_pooling | pooled | 0.8030 | 0.8461 | 0.8383 | 0.8402 | 0.8632 | 0.8676 | 0.8866 |
| target_pooling | cond | 0.8030 | 0.8472 | 0.8497 | 0.8592 | 0.8716 | 0.8848 | 0.8861 |
| patch_correspondence | patch_mean | 0.7991 | 0.8577 | 0.8651 | 0.8529 | 0.8536 | 0.8863 | 0.8558 |
| patch_correspondence | cls | 0.7991 | 0.8403 | 0.8627 | 0.8544 | 0.8821 | 0.8768 | 0.8804 |
| patch_correspondence | pooled | 0.7991 | 0.8496 | 0.8685 | 0.8629 | 0.8742 | 0.8851 | 0.8834 |
| patch_correspondence | cond | 0.7991 | 0.8514 | 0.8624 | 0.8596 | 0.8826 | 0.8720 | 0.8626 |

## 旧参考（vits16 @224 + SAM3 @512，仅粗参考，分辨率不同不可直接对比）

| mode | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | 0.7479 | 0.7759 | 0.7565 | 0.7858 | 0.8217 | 0.8168 | 0.8192 |
| dino_global_pooling | 0.7408 | 0.7483 | 0.7583 | 0.7868 | 0.7775 | 0.8184 | 0.8186 |
| dino_patch_average | 0.7286 | 0.7269 | 0.7569 | 0.7468 | 0.7998 | 0.8292 | 0.8263 |
| anchor_conditioned_target_pooling | 0.7413 | 0.7602 | 0.8272 | 0.8408 | 0.8413 | 0.8486 | 0.8546 |
| anchor_conditioned_patch_correspondence | 0.7578 | 0.7735 | 0.8223 | 0.8328 | 0.8340 | 0.8336 | 0.8421 |
