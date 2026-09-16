# Stage1 Base No-FT Route Audit 256 vs 512

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
| t18_corrected | 0.788488 | 0.873543 | 0.085055 | 0.747869 | 0.775949 | 0.756490 | 0.785808 | 27/24/23/26 | 30/17/16/37 |
| dino_global_pooling | 0.804751 | 0.856537 | 0.051786 | 0.740844 | 0.748290 | 0.758326 | 0.786777 | 30/25/22/23 | 24/19/26/31 |
| dino_patch_average | 0.795524 | 0.853908 | 0.058384 | 0.728634 | 0.726893 | 0.756876 | 0.746756 | 28/15/28/29 | 15/25/22/38 |
| anchor_conditioned_target_pooling | 0.806705 | 0.878697 | 0.071992 | 0.741306 | 0.760176 | 0.827209 | 0.840837 | 14/26/22/38 | 17/22/28/33 |
| anchor_conditioned_patch_correspondence | 0.810373 | 0.877550 | 0.067176 | 0.757814 | 0.773501 | 0.822305 | 0.832819 | 15/23/27/35 | 20/20/22/38 |

## Canvas 256
| feature mode | weighted Dice | oracle Dice | selector regret | direct | bridge_1 | bridge_2 | bridge_3 | selected D/B1/B2/B3 | oracle D/B1/B2/B3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | 0.813516 | 0.878715 | 0.065199 | 0.742137 | 0.781679 | 0.794190 | 0.834032 | 30/27/22/21 | 30/23/18/29 |
| dino_global_pooling | 0.793829 | 0.859586 | 0.065757 | 0.716809 | 0.776021 | 0.763548 | 0.785543 | 33/20/19/28 | 30/13/19/38 |
| dino_patch_average | 0.789006 | 0.862750 | 0.073744 | 0.719933 | 0.756324 | 0.763273 | 0.790535 | 26/15/29/30 | 19/23/20/38 |
| anchor_conditioned_target_pooling | 0.789439 | 0.880177 | 0.090739 | 0.722515 | 0.778659 | 0.830891 | 0.835595 | 18/21/22/39 | 16/23/22/39 |
| anchor_conditioned_patch_correspondence | 0.796623 | 0.876311 | 0.079688 | 0.734856 | 0.792785 | 0.824325 | 0.845590 | 15/26/26/33 | 19/19/22/40 |

## Key Equations
D_oracle = mean_q max(D_q_direct, D_q_b1, D_q_b2, D_q_b3)
R_selector = D_oracle - D_weighted
