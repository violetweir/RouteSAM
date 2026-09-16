### Canvas 512
| feature mode | weighted Dice | direct | bridge_1 | bridge_2 | bridge_3 | weighted IoU | selected direct/bridge_1/bridge_2/bridge_3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | 0.788488 | 0.747869 | 0.775949 | 0.756490 | 0.785808 | 0.719922 | 27/24/23/26 |
| dino_global_pooling | 0.804751 | 0.740844 | 0.748290 | 0.758326 | 0.786777 | 0.738917 | 30/25/22/23 |
| dino_patch_average | 0.795524 | 0.728634 | 0.726893 | 0.756876 | 0.746756 | 0.723077 | 28/15/28/29 |
| anchor_conditioned_target_pooling | 0.806705 | 0.741306 | 0.760176 | 0.827209 | 0.840837 | 0.742457 | 14/26/22/38 |
| anchor_conditioned_patch_correspondence | 0.810373 | 0.757814 | 0.773501 | 0.822305 | 0.832819 | 0.740419 | 15/23/27/35 |

### Canvas 256
| feature mode | weighted Dice | direct | bridge_1 | bridge_2 | bridge_3 | weighted IoU | selected direct/bridge_1/bridge_2/bridge_3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | 0.813516 | 0.742137 | 0.781679 | 0.794190 | 0.834032 | 0.753156 | 30/27/22/21 |
| dino_global_pooling | 0.793829 | 0.716809 | 0.776021 | 0.763548 | 0.785543 | 0.728351 | 33/20/19/28 |
| dino_patch_average | 0.789006 | 0.719933 | 0.756324 | 0.763273 | 0.790535 | 0.719049 | 26/15/29/30 |
| anchor_conditioned_target_pooling | 0.789439 | 0.722515 | 0.778659 | 0.830891 | 0.835595 | 0.726536 | 18/21/22/39 |
| anchor_conditioned_patch_correspondence | 0.796623 | 0.734856 | 0.792785 | 0.824325 | 0.845590 | 0.728791 | 15/26/26/33 |

### Delta 256 - 512
| feature mode | weighted Dice delta | direct delta | bridge_1 delta | bridge_2 delta | bridge_3 delta |
|---|---:|---:|---:|---:|---:|
| t18_corrected | +0.025028 | -0.005732 | +0.005730 | +0.037700 | +0.048224 |
| dino_global_pooling | -0.010922 | -0.024036 | +0.027731 | +0.005222 | -0.001234 |
| dino_patch_average | -0.006519 | -0.008701 | +0.029430 | +0.006397 | +0.043780 |
| anchor_conditioned_target_pooling | -0.017266 | -0.018791 | +0.018484 | +0.003682 | -0.005242 |
| anchor_conditioned_patch_correspondence | -0.013750 | -0.022958 | +0.019284 | +0.002020 | +0.012771 |
