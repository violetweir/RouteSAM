# E3 Selector Compare: fixed anchor-conditioned target pooling

Validation only. Pairwise logistic uses 5-fold query split inside validation.

## Canvas 512
| method | Dice | regret vs oracle | selected D/B1/B2/B3 |
|---|---:|---:|---:|
| current | 0.766048 | 0.082691 | 26/23/30/21 |
| route_wise_zscore | 0.794292 | 0.054448 | 20/21/35/24 |
| explicit_geometry | 0.784377 | 0.064362 | 1/2/31/66 |
| pairwise_logistic_ranker | 0.799513 | 0.049227 | 69/4/7/20 |
| always_bridge_3 | 0.770852 | 0.077888 | 0/0/0/100 |
| oracle | 0.848740 | 0.000000 | 23/18/22/37 |

## Canvas 256
| method | Dice | regret vs oracle | selected D/B1/B2/B3 |
|---|---:|---:|---:|
| current | 0.769089 | 0.089101 | 23/27/21/29 |
| route_wise_zscore | 0.800590 | 0.057599 | 9/30/30/31 |
| explicit_geometry | 0.798777 | 0.059413 | 1/2/31/66 |
| pairwise_logistic_ranker | 0.805845 | 0.052344 | 61/11/11/17 |
| always_bridge_3 | 0.792900 | 0.065290 | 0/0/0/100 |
| oracle | 0.858190 | 0.000000 | 25/18/25/32 |

