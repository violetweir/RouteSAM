# Baseline-aware risk gate: dual-dataset frozen evaluation

Frozen thresholds: harm risk <= `0.10` or rescue probability >= `1.01`; labels: harm `delta < -0.05`, rescue `delta > +0.05`.
The gate and threshold use Kvasir validation only; no student predictions are loaded.

| Dataset | fixed b6 | raw ranker | gated ranker | raw delta | gated delta | catastrophic raw->gated | CVaR10 raw->gated | fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kvasir-SEG | 0.897780 | 0.906307 | 0.900460 | +0.008527 | +0.002680 | 4->1 | -0.123855->-0.022701 | 16 |
| CVC-ClinicDB | 0.815820 | 0.857032 | 0.856862 | +0.041213 | +0.041042 | 2->0 | -0.107111->-0.013229 | 12 |
