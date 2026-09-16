# E2 Validation Route Audit

## Route Hash Audit
| feature mode | routes 512 | routes 256 | same ordering | all route hashes equal | mismatches |
|---|---:|---:|---:|---:|---:|
| t18_corrected | 400 | 400 | True | True | 0 |
| dino_global_pooling | 400 | 400 | True | True | 0 |
| dino_patch_average | 400 | 400 | True | True | 0 |
| anchor_conditioned_target_pooling | 400 | 400 | True | True | 0 |
| anchor_conditioned_patch_correspondence | 400 | 400 | True | True | 0 |

## Canvas 512
| feature mode | weighted Dice | oracle Dice | selector regret | direct | bridge_1 | bridge_2 | bridge_3 | selected D/B1/B2/B3 | oracle D/B1/B2/B3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | 0.804241 | 0.842054 | 0.037813 | 0.770012 | 0.791919 | 0.784470 | 0.804409 | 25/26/14/35 | 33/22/21/24 |
| dino_global_pooling | 0.801193 | 0.848924 | 0.047731 | 0.776038 | 0.766501 | 0.767890 | 0.767032 | 26/29/22/23 | 26/24/20/30 |
| dino_patch_average | 0.777063 | 0.846125 | 0.069062 | 0.758151 | 0.739186 | 0.768787 | 0.782758 | 33/19/19/29 | 27/25/17/31 |
| anchor_conditioned_target_pooling | 0.766048 | 0.848740 | 0.082691 | 0.731848 | 0.764928 | 0.813977 | 0.770852 | 26/23/30/21 | 23/18/22/37 |
| anchor_conditioned_patch_correspondence | 0.793123 | 0.856683 | 0.063560 | 0.753812 | 0.783968 | 0.815171 | 0.807027 | 23/19/30/28 | 30/15/23/32 |

## Canvas 256
| feature mode | weighted Dice | oracle Dice | selector regret | direct | bridge_1 | bridge_2 | bridge_3 | selected D/B1/B2/B3 | oracle D/B1/B2/B3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | 0.830202 | 0.861486 | 0.031284 | 0.767406 | 0.803178 | 0.797865 | 0.804913 | 19/30/16/35 | 26/24/19/31 |
| dino_global_pooling | 0.805351 | 0.851277 | 0.045926 | 0.772543 | 0.800537 | 0.782218 | 0.791988 | 17/35/28/20 | 29/16/23/32 |
| dino_patch_average | 0.806499 | 0.848212 | 0.041713 | 0.753238 | 0.758628 | 0.778184 | 0.806683 | 26/20/26/28 | 26/25/20/29 |
| anchor_conditioned_target_pooling | 0.769089 | 0.858190 | 0.089101 | 0.733700 | 0.765756 | 0.822212 | 0.792900 | 23/27/21/29 | 25/18/25/32 |
| anchor_conditioned_patch_correspondence | 0.783248 | 0.866338 | 0.083090 | 0.755816 | 0.790581 | 0.813322 | 0.828202 | 22/16/29/33 | 28/14/26/32 |
