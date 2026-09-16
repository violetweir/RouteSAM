# Feature-Agnostic Union/Overlap Audit

Route identity hash excludes feature_mode: query_id + route_type + anchor_id + bridge_ids.

## Validation Union Oracle
| Canvas | Combo | union oracle | best single oracle | best single feature | delta | unique route-id fraction | duplicate-query count | decision |
|---|---|---:|---:|---|---:|---:|---:|---|
| 512 | T18+target | 0.887004 | 0.856683 | Patch correspondence | +0.030321 | 0.974 | 21 | multi-view route proposal |
| 512 | target+patchcorr | 0.869253 | 0.856683 | Patch correspondence | +0.012570 | 0.780 | 70 | multi-view route proposal |
| 512 | T18+target+patchcorr | 0.890528 | 0.856683 | Patch correspondence | +0.033845 | 0.833 | 78 | multi-view route proposal |
| 512 | all_features | 0.907047 | 0.856683 | Patch correspondence | +0.050364 | 0.782 | 99 | multi-view route proposal |
| 256 | T18+target | 0.892694 | 0.866338 | Patch correspondence | +0.026355 | 0.974 | 21 | multi-view route proposal |
| 256 | target+patchcorr | 0.880071 | 0.866338 | Patch correspondence | +0.013733 | 0.780 | 70 | multi-view route proposal |
| 256 | T18+target+patchcorr | 0.897057 | 0.866338 | Patch correspondence | +0.030718 | 0.833 | 78 | multi-view route proposal |
| 256 | all_features | 0.911628 | 0.866338 | Patch correspondence | +0.045289 | 0.782 | 99 | multi-view route proposal |

## Test Exploratory Union Oracle
| Canvas | Combo | union oracle | best single oracle | best single feature | delta | unique route-id fraction | duplicate-query count |
|---|---|---:|---:|---|---:|---:|---:|
| 512 | T18+target | 0.910500 | 0.878697 | Target pooling | +0.031803 | 0.974 | 20 |
| 512 | target+patchcorr | 0.890709 | 0.878697 | Target pooling | +0.012012 | 0.767 | 71 |
| 512 | T18+target+patchcorr | 0.913494 | 0.878697 | Target pooling | +0.034797 | 0.823 | 86 |
| 512 | all_features | 0.921158 | 0.878697 | Target pooling | +0.042460 | 0.763 | 99 |
| 256 | T18+target | 0.910225 | 0.880177 | Target pooling | +0.030047 | 0.974 | 20 |
| 256 | target+patchcorr | 0.891998 | 0.880177 | Target pooling | +0.011821 | 0.767 | 71 |
| 256 | T18+target+patchcorr | 0.912239 | 0.880177 | Target pooling | +0.032062 | 0.823 | 86 |
| 256 | all_features | 0.921385 | 0.880177 | Target pooling | +0.041208 | 0.763 | 99 |
