# ft_1pct propagation-quality router results (Kvasir 1% full fine-tune)

Checkpoint: `work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt` (finetune_1pct_seed2026, epoch 20, 160 train steps).

## Ridge router (validation-trained scorer, no GT in selection)

| scheme | selected Dice | oracle | gap | acc | spearman |
|---|---:|---:|---:|---:|---:|
| b3-b6 target_pooling+patch_correspondence | 0.894648 | 0.919805 | 0.025157 | 0.28 | 0.352 |
| b3-b6 anchor_conditioned_target_pooling | 0.886241 | 0.903572 | 0.017331 | 0.37 | 0.376 |
| b3-b6 anchor_conditioned_patch_correspondence | 0.884695 | 0.901863 | 0.017168 | 0.31 | 0.311 |
| b0-b6 target_pooling+patch_correspondence | 0.877061 | 0.921800 | 0.044739 | 0.17 | 0.364 |
| b0-b6 anchor_conditioned_target_pooling | 0.868192 | 0.911798 | 0.043606 | 0.14 | 0.361 |
| b0-b6 anchor_conditioned_patch_correspondence | 0.890865 | 0.907062 | 0.016198 | 0.25 | 0.355 |

## Test per-bridge Dice (direct..b6)

| bridge | target_pooling | patch_correspondence |
|---|---:|---:|
| direct | 0.770790 | 0.797878 |
| b1 | 0.792198 | 0.816105 |
| b2 | 0.857394 | 0.857803 |
| b3 | 0.852228 | 0.859885 |
| b4 | 0.860925 | 0.884293 |
| b5 | 0.873081 | 0.865846 |
| b6 | 0.864322 | 0.879738 |

## Validation per-bridge Dice (scorer training split)

| bridge | target_pooling | patch_correspondence |
|---|---:|---:|
| direct | 0.791716 | 0.799931 |
| b1 | 0.823561 | 0.831825 |
| b2 | 0.827019 | 0.846262 |
| b3 | 0.818899 | 0.855962 |
| b4 | 0.848860 | 0.852945 |
| b5 | 0.841205 | 0.839182 |
| b6 | 0.862006 | 0.855331 |
