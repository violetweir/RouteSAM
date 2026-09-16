# `scripts/` — reproduction stages and experiment drivers

251 scripts (plus this index). The two entry points you normally need are at the
top; everything else is a single stage, a driver for one experiment line, or an
analysis helper.

All scripts read paths from `configs/reproduction.toml` (copied from
`configs/reproduction.example.toml`). Anything with an absolute
`/Data_8TB/lht/...` default is a historical server default kept for provenance.

> Note: two files (`tmp_eval_no_b7.py`, `_inspect_c3_feat.py`) are scratch
> utilities kept because they are referenced from reports.

## Entry points

| Script | Role |
|---|---|
| `run_pipeline.py` | clean final mainline: fixed pseudo-video protocol → `S27 X3 Final + B7` |
| `run_method_ladder.py` | full ladder B00 → T19/T20 → T18 → E1 → T21 → T24 → S27 X3 → B7 |
| `run_s27_student.py` | unified S27 student trainer (X0/X1/X3) |
| `run_t24_student.py` | T24 committee student trainer (S2/S3, legacy loss) |

## Reproduction ladder

| Stage | Scripts |
|---|---|
| B00 single-image SAM3 | `run_b00_sam3_single_image.py`, `summarize_b00_single_image.py` |
| T17 autonomous 1% anchors | `prepare_t17_autonomous_1pct.py`, `extract_t17_sam3_candidates.py`, `train_eval_t17_support_selector.py` |
| T18 two-frame pseudo-video | `prepare_t18_pseudovideo_pilot.py`, `prepare_t18_full_retrieval.py`, `eval_t18_pseudovideo_pilot.py` |
| T19 SC-SAM baseline | `run_t19_scsam_baseline.py`, `run_kvasir_1pct_scsam_synfoc_baselines.sh` |
| T20 SynFoC baseline | `run_t20_synfoc_baseline.py` |
| T21 frozen pseudo-video routes | `run_t21_dynamic_pseudovideo.py`, `prepare_s27_frozen_routes.py` |
| E1 multi-step propagation | `prepare_e1_multistep.py`, `eval_e1_multistep.py`, `summarize_e1_multistep.py` |
| T22/T23 tracker & adapters | `prepare_t22_training.py`, `train_t22_sam3_tracker.py`, `train_t23_memory_adapter.py`, `train_t23_single_image_decoder.py`, `eval_t23_single_image_decoder.py` |
| T24 committee students | `prepare_t24_pseudo_consensus.py`, `export_t24_student_predictions.py`, `run_phase1_post_t24.sh` |
| T25 route selection | `run_t25_validation_routes.py`, `run_t25_offline_analysis.py`, `calibrate_t25_linear_weights.py`, `export_t25_student_predictions.py` |
| T26 renewed selection | `run_t26_renewed_route_selection.py`, `export_t26_sam3_probabilities.py` |
| S27 X3 expansion | `build_s27_audit_and_tiers.py`, `build_s27_expansion_sets.py`, `prepare_s27_remaining_pool.py`, `select_and_eval_s27_student.py`, `verify_s27_training_weights.py` |
| Phase-1 orchestration | `run_phase1_orchestrator.sh`, `run_phase1_pre_t24.sh`, `run_phase1_stage2_4.sh`, `phase1_audit_tiers.py`, `phase1_b7_select.py`, `phase1_build_x3_manifest.py` |

## Route construction and propagation quality

| Script | Role |
|---|---|
| `prepare_kvasir_protocol.py`, `prepare_protocol.py`, `prepare_kvasir_baseline_data.py` | build the merged split and manifests |
| `stage1_feature_knn_routes.py`, `run_stage1_*.sh` | frozen-feature kNN route generation (all feature modes) |
| `stage1_eval_routes_forward_only.py`, `stage1_eval_feature_knn_routes.py`, `eval_route_forward_dice.py` | forward-only route Dice |
| `eval_route_propagation_quality.py` (+ `_det_gtmask*` variants) | propagation-quality feature extraction |
| `run_propagation_quality_main_modes.sh` | run the two main route generators |
| `eval_cycle_proxy.py`, `eval_transport_correlation.py`, `analyze_transport_proxies.py` | cycle-consistency and transport diagnostics |
| `build_sam3enc_cond_mask_routes.py`, `build_sam3enc_anchor_subsets.py`, `run_sam3enc_anchor_*.sh` | anchor-conditioned route variants |
| `filter_route_pool.py`, `seed_c0_256_round2d_e33_quality.py` | candidate-pool filtering |

## Selectors and routers

| Script | Selector |
|---|---|
| `summarize_candidate_invariant_v2.py` | v2 absolute (candidate-invariant) ridge router |
| `analyze_propagation_quality_router.py` | propagation-quality router analysis |
| `eval_bridge_range_quality_router.py` | bridge-range sweep (`--min-bridge 3 --max-bridge 6`) |
| `eval_ft1pct_pq_router.py`, `run_eval_ft1pct_pq_router.sh` | same router on the 1% fine-tuned SAM3 |
| `analyze_candidate_invariant_router.py`, `analyze_unified_candidate_pool_router.py` | unified candidate-pool routers |
| `run_route_selector_vitb256.sh`, `analyze_route_selector_vitb256.py`, `report_route_selector_vitb256.py` | ViT-B/256 route selector |
| `run_pairwise_route_ranker_vitb256.py`, `eval_frozen_pairwise_route_ranker_vitb256.py`, `run_pairwise_risk_gate_vitb256.py`, `run_pairwise_route_ranker_ablation_vitb256.py` | pairwise rankers and risk gating |
| `summarize_b7_threshold_grid.py`, `run_b7_x3_audit_vitb256.py` | B7 threshold grid and audit |
| `account_routeco_v1_fairness.py`, `train_routeco_sam3_v1.py`, `run_routeco_v1_*.sh` | RouteCo v1 |

## Kvasir 1% anchor line

`run_stage1_b7_routes.sh`, `run_stage1_b7_eval_forward.sh`,
`run_stage1_b7_validation_eval_forward.sh`, `run_stage1_eval_base.sh`,
`run_stage1_eval_ft20.sh`, `run_kvasir_max6_all_models.sh`,
`run_kvasir_multiframe_weighted_models.sh`, `run_kvasir_max6_pseudovideo.py`,
`summarize_kvasir_weighted_models.py`, `build_kvasir_1pct_plus_*_dataset.py`,
`run_kvasir_1pct_*_after_train.sh`, `prepare_clinicdb_external_kvasir8_protocol.py`.

## ISIC2018 / BUSI

`prepare_isic18_b7_medsam3_dataset.py`, `prepare_isic_resized_cache.py`,
`run_isic18_round1_round2a_from_pvf.sh`, `prepare_busi_synfoc_protocol.py`,
`run_busi_synfoc_1pct.sh`, `collect_synfoc_busi_result.py`.

## SAM3 LoRA, tracker and encoder transport

`train_sam3_lora_kvasir.py`, `train_sam3_lora_kvasir_e50.py`,
`prepare_c0_256_medsam3_dataset.py`, `prepare_c0_256_b7_medsam3_dataset.py`,
`build_c0_256_b7_lora_manifest.py`, `summarize_c0_256_b7_lora_checkpoints.py`,
`run_c0_256_round3_tracker.sh`, `train_round3_tracker.py`,
`run_c0_256_round3_lora.sh`, `summarize_round3_lora.py`,
`summarize_round3_tracker.py`, `inspect_round3_lora_checkpoint.py`,
`merge_sam3_lora_video_checkpoint.py`, `merge_sam3_video_checkpoints.py`,
`extract_sam3_encoder_patches.py`, `extract_sam3_mask_descriptors.py`,
`extract_sam3_fpn_transport_checkpoint.py`, `apply_sam3enc_base_fusion_train.py`,
`run_sam3enc_base_fusion_v1.py`, `sam3_lora.py`, `sam3_memory_adapter.py`,
`sam3_memory_write_modes.py`, `round3_lora.py`.

## Prompt and text-routing experiments

`compare_mask_vs_box_prompt.py`, `eval_base_sam3_direct_validation.py`,
`build_qwen_text_embeddings.py`, `build_qwen_text_knn_features.py`,
`build_qwen_text_routes.py`, `build_mask_visual_qwen_knn_routes.py`,
`qwen35_describe_images.py`, `qwen35_describe_masks.py`,
`run_mask_visual_lambda_scan.sh`, `run_original_visual_lambda_scan.sh`,
`run_triple_visual_qwen_scan.sh`, `run_msr_reroute_v1.py`.

## Supervisors and queues

The `supervise_*.sh` and `run_c0_256_*_queue.sh` scripts are the GPU queues that
chained long runs on the original server. They are kept to document the exact
execution order; they are not needed to reproduce results on a single machine.

## Utilities

`src/pvseg/` holds the shared library (`io`, `metrics`, `protocol`).
`sanitize_protocol_assets.py` cleans path leakage from protocol assets,
`analyze_consensus.py` measures route/label consensus, `merge_student_uncertainty.py`
and `export_student_uncertainty.py` handle uncertainty exports,
`analyze_feature_focus_paper.py` and `figA_examples.py` produce paper figures,
`visualize_round1_knn_features.py` renders kNN route examples,
`patch_remote_sam3_*.py` are server-side patches applied to the external SAM3
checkout, and `finalize_round3_*.py` / `archive_round3_full_ft_failure.py`
close out and archive the round-3 runs.

## Tests

```bash
python -m pytest tests -q
```

`tests/` checks protocol assets, split-leakage guards, freeze-boundary
invariants and checkpoint reload behaviour for the round-3 LoRA/tracker code.
