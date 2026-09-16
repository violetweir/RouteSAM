# E4-E7 Multi-view Candidate Pool Analysis

No SAM3 rerun was used. Inputs are existing validation route audit CSVs.

Important limitation: for cross-view path descriptors, this report uses scores available from existing proposed routes. A route not proposed by a view has no edge-level score in the current CSV, so its percentile for that view is treated as 0 until raw feature-edge matrices are exported.

## E4 Leave-one-out marginal contribution

### Canvas 256

| Feature view | Leave-one-out oracle drop | Unique oracle wins | Unique route ratio | Decision |
|---|---:|---:|---:|---|
| T18 | 0.012236 | 39 | 0.922 | keep |
| DINO-G | 0.011414 | 21 | 0.655 | keep |
| DINO-P | 0.005737 | 12 | 0.632 | keep |
| Target | 0.005903 | 25 | 0.487 | keep |
| Corr | 0.007158 | 25 | 0.497 | keep |

### Canvas 512

| Feature view | Leave-one-out oracle drop | Unique oracle wins | Unique route ratio | Decision |
|---|---:|---:|---:|---|
| T18 | 0.011475 | 29 | 0.922 | keep |
| DINO-G | 0.011263 | 30 | 0.655 | keep |
| DINO-P | 0.005480 | 20 | 0.632 | keep |
| Target | 0.009686 | 25 | 0.487 | keep |
| Corr | 0.008587 | 19 | 0.497 | keep |

## E7 strict unsupervised selectors

### Canvas 256

| Selector | Dice | Oracle | Regret | selected D/B1/B2/B3 | mean consensus |
|---|---:|---:|---:|---:|---:|
| S1_borda_mean_rank | 0.766311 | 0.911628 | 0.145316 | 5/15/26/54 | 0.348 |
| S2_median_rank | 0.767604 | 0.911628 | 0.144024 | 5/22/24/49 | 0.362 |
| S3_maximin | 0.810737 | 0.911628 | 0.100891 | 8/25/24/43 | 0.284 |
| S4_consensus_median | 0.753479 | 0.911628 | 0.158149 | 72/19/6/3 | 0.660 |
| S5_pareto | 0.750529 | 0.911628 | 0.161099 | 16/36/24/24 | 0.452 |
| S0_T18_current | 0.830202 | 0.861486 | 0.031284 | 20/30/16/34 |  |
| S0_Target_geometry | 0.810430 | 0.858190 | 0.047759 | 0/9/68/23 |  |
| S0_Corr_geometry | 0.817698 | 0.866338 | 0.048640 | 0/2/20/78 |  |

### Canvas 512

| Selector | Dice | Oracle | Regret | selected D/B1/B2/B3 | mean consensus |
|---|---:|---:|---:|---:|---:|
| S1_borda_mean_rank | 0.759924 | 0.907047 | 0.147124 | 7/14/30/49 | 0.352 |
| S2_median_rank | 0.751960 | 0.907047 | 0.155087 | 11/20/25/44 | 0.382 |
| S3_maximin | 0.783272 | 0.907047 | 0.123775 | 13/28/20/39 | 0.300 |
| S4_consensus_median | 0.754487 | 0.907047 | 0.152561 | 72/19/5/4 | 0.660 |
| S5_pareto | 0.732378 | 0.907047 | 0.174669 | 23/31/23/23 | 0.478 |
| S0_T18_current | 0.804241 | 0.842054 | 0.037813 | 27/25/14/34 |  |
| S0_Target_geometry | 0.800039 | 0.848740 | 0.048701 | 0/9/68/23 |  |
| S0_Corr_geometry | 0.792931 | 0.856683 | 0.063752 | 1/31/45/23 |  |
