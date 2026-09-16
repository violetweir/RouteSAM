# `paper/` — paper-facing experiment packages

Consolidated, paper-ready extracts of the two largest experiment lines. These
packages were built from the raw run outputs under the original `work/` tree and
contain cleaned tables, protocol notes, source maps, and a few figures.

```text
paper/
  isic2018_experiment_package/   ISIC2018 pseudo-video study
  kvasir_experiment_package/     Kvasir-SEG 1% anchor study
  tools/                         scripts that (re)built these packages
```

## `isic2018_experiment_package/`

Read `isic2018_experiment_package/README.md` first. Highlights:

- `00_executive_summary.md` — claims and numbers for the Results section
- `01_experiment_protocol.md` — dataset, split, resolution, prompt protocol
- `02_main_results.md` — paper-ready main comparisons
- `03_round_progression_and_ablation.md` — S2/S3/X3/X4 progression, B7 notes
- `04_bridge_analysis.md` — target pooling and long-chain bridge analysis
- `05_sam3_lora_direct.md` — LoRA / direct SAM3 results (256 vs 1008)
- `06_runtime_and_engineering_notes.md` — loader, cache and speed notes
- `tables/`, `complete_experiment_tables/tables/` — CSV tables
- `figures/`, `raw_records/`, `source_map.csv`

## `kvasir_experiment_package/`

Read `kvasir_experiment_package/README.md` first. Highlights:

- `01_protocol.md` — `train=800 / validation=100 / test=100`, 8 anchors
- `02_results_overview.md` — overview of the Kvasir results
- `tables/top_aggregate_route_family_dice.csv` — start here for paper tables
- `tables/curated_metric_ledger_long.csv` — curated ledger across all runs
- `tables/all_json_scalar_metrics_long.csv` — every scalar metric from every JSON
- `tables/log_metric_lines.csv` — metric lines extracted from training logs

> **Size warning.** `tables/log_metric_lines.csv` (~63 MB) and
> `tables/all_kvasir_artifacts.csv` (~49 MB) are the largest files in this
> repository. They are under GitHub's 100 MB hard limit but above the 50 MB
> warning threshold — gzip them if you want a warning-free push.

## `tools/`

- `build_kvasir_paper_package.py` — rebuilds the Kvasir package from `work/`
- `update_isic_base_patch_correspondence_paper.py` — refreshes the ISIC
  base/patch-correspondence tables

Both scripts expect the original server output tree and are kept for
provenance; they are not part of the reproduction mainline.
