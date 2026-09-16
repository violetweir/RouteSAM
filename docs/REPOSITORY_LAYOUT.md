# Repository layout

This repository is a cleaned, publishable extraction of the original server
working tree `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`. Everything that is source
code, documentation, protocol, or a small result summary was kept; datasets,
checkpoints, mask dumps, feature caches, logs, and the old 7 GB git history
were dropped.

## Directory map

| Directory | Contents | Approx. size |
|---|---|---|
| `configs/` | run configuration; `reproduction.example.toml` is the template, the rest are study-specific | 40 KB |
| `docs/` | method, protocol and experiment documents; `docs/cross_dataset_1pct.md` is the **mainline** write-up, `docs/kvasir_program.md` the Kvasir-SEG inventory, `docs/s27_x3_b7_line.md` the historical line | 312 KB |
| `docs/v2/` | earlier document series (encoder-KNN, LoRA fusion, Qwen-text routing) | 64 KB |
| `envs/` | example conda manifests: `sam3.yml`, `student.yml` | 12 KB |
| `protocols/reproduction_v1/` | fixed path-free split and the 16 support anchors | 272 KB |
| `scripts/` | all reproduction stages and experiment drivers | 2.2 MB |
| `src/pvseg/` | shared utilities (`io`, `metrics`, `protocol`) | 24 KB |
| `tests/` | protocol asset, split-leakage, freeze-boundary and checkpoint tests | 28 KB |
| `third_party/` | vendored `SC-SAM` and `SynFoC-T20` student baselines | 956 KB |
| `medsam3/` | vendored SAM3 + LoRA training toolkit (238 files) | 18 MB |
| `paper/` | paper-facing experiment packages and the scripts that built them | 209 MB |
| `reports/` | per-experiment reproduction reports (Markdown) | 460 KB |
| `mainline/` | cross-dataset 1% studies: coverage anchors, score calibration, TP routes, diagnostics figures | 51 MB |
| `results/` | curated per-run summary metrics extracted from `work/` | 19 MB |

## Old path → new path

| Original server path | This repository |
|---|---|
| `README.md` | `README.md` (new landing page); original text kept at `docs/reproduction_guide.md` |
| `docs/mainline.md` | `docs/s27_x3_b7_line.md` (retitled: historical, not the mainline) |
| — | `docs/cross_dataset_1pct.md`, `docs/kvasir_program.md` (new mainline write-ups) |
| `docs/` | `docs/` |
| `docs_v2/` | `docs/v2/` |
| `scripts/` | `scripts/` |
| `src/`, `tests/`, `configs/`, `envs/`, `protocols/` | same names |
| `third_party/` | `third_party/` |
| `MedSAM3/` | `medsam3/` (dropped `outputs/`, `Machinery.zip`, demo screenshots, caches) |
| `paper_isic2018_experiment_package/` | `paper/isic2018_experiment_package/` |
| `paper_kvasir_experiment_package/` | `paper/kvasir_experiment_package/` |
| `build_kvasir_paper_package.py`, `update_isic_base_patch_correspondence_paper.py` | `paper/tools/` |
| `reproduction_reports/` | `reports/` |
| `new_project/` | `mainline/` |
| `work/<experiment>/` | `results/<experiment>/` (only `*.md`, `*.py`, and small `*.json` / `*.csv` / `*.txt` / `*.sh`) |
| `123.md` (scratch notes) | not included |
| `.git/` (7.24 GB of loose objects) | not included — start a fresh history |

Because of the `new_project/` → `mainline/` rename, the two scripts that
referenced it relatively (`scripts/prepare_busi_synfoc_protocol.py`,
`scripts/collect_synfoc_busi_result.py`) were updated accordingly. Code inside
`mainline/` still contains the original absolute server paths it was run with;
those are historical records, not live configuration.

## What was filtered out of every copy

- `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ipynb_checkpoints/`
- `.DS_Store`, `._*`
- backup copies: `*.bak`, `*.bak_*`, `*.pre_*`, `*.orig`, `*.rej`
  (e.g. `run_s27_student.py.bak_long_epoch_`,
  `train_sam3_lora_kvasir_e50.pre_direct_dice.py`)
- binaries and data: `*.pt`, `*.pth`, `*.ckpt`, `*.npz`, `*.npy`, `*.pkl`,
  `*.zip`, mask/image dumps, per-target prediction PNGs
- `configs/reproduction.toml` (machine-specific absolute paths; use the
  `reproduction.example.toml` template)

## Regenerating this extraction

The extraction was performed with `rsync` include/exclude filters:

```bash
# core trees, minus caches and backup copies
rsync -a --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.bak*' \
      --exclude='.DS_Store' --exclude='._*' "$SRC/src/" "$DST/src/"

# follow-up studies: text artifacts only, <= 200 KB each
rsync -a --prune-empty-dirs --max-size=200K \
      --include='*/' --include='*.md' --include='*.py' --include='*.json' \
      --include='*.yaml' --include='*.sh' --include='*.csv' \
      --exclude='*' "$SRC/new_project/" "$DST/mainline/"

# run outputs: reports, driver scripts and small summaries only
rsync -a --prune-empty-dirs --max-size=100K \
      --include='*/' --include='*.md' --include='*.py' --include='*.json' \
      --include='*.csv' --include='*.txt' --include='*.sh' --include='*.yaml' \
      --exclude='*' "$SRC/work/" "$DST/results/"
```

## Notes for publishing

- Start a fresh git history: the original `.git` is 7.24 GB of loose objects and
  cannot be pushed. `git init && git add . && git commit` in this folder.
- The largest tracked files are the paper table CSVs in
  `paper/kvasir_experiment_package/tables/` — `log_metric_lines.csv` (~63 MB) and
  `all_kvasir_artifacts.csv` (~49 MB). Both are below GitHub's 100 MB hard limit
  but above its 50 MB warning threshold. If you want to avoid the warning,
  gzip them and note the change in `paper/kvasir_experiment_package/README.md`.
- `.gitignore` already excludes `data/`, `work/`, checkpoints and `*.pt`/`*.npz`
  so a future local rerun will not dirty the tree. Those path rules are anchored
  to the repository root (`/data/`, `/work/`, `/runs/`) on purpose — the
  extracted experiment directories legitimately contain sub-folders with those
  names, and an unanchored rule would silently drop them.
- Vendored directories keep their upstream `.gitignore`
  (`medsam3/.gitignore`, `third_party/SC-SAM/.gitignore`). Neither currently
  ignores a file that is present, but check them if you add files there.
- Verified before hand-off: `git add -A` picks up all 5081 files and nothing is
  ignored.
