# `docs/` — method and protocol documents

## Start here

| Document | Contents |
|---|---|
| [`cross_dataset_1pct.md`](cross_dataset_1pct.md) | **the mainline** — Kvasir / ISIC2018 / BUSI (+TN3K) 1% anchor study: ACV pipeline, bias-ratio diagnostic, cross-dataset results |
| [`kvasir_versions.md`](kvasir_versions.md) | **Kvasir-SEG V1 → V7, version by version** — motivation, pipeline, pools, results, caveats, artifacts |
| [`kvasir_program.md`](kvasir_program.md) | **Kvasir-SEG, both lines** — line 1 (Kvasir-only `S27 X3 + B7` analogue: routes, selectors, SAM3 adaptation, students) and line 2 (automatic anchors + calibration + breadth/depth); complete experiment inventory |
| [`phase1_kvasir1pct_student.md`](phase1_kvasir1pct_student.md) | transferring the `S27 X3 + B7` student machinery onto the Kvasir 1% + propagation-quality-router baseline |
| [`s27_x3_b7_line.md`](s27_x3_b7_line.md) | **historical** — the original `S27 X3 + B7` merged CVC+Kvasir pipeline |
| [`protocol.md`](protocol.md) | the fixed reproduction protocol: splits, anchors, quality terms |
| [`reproduction_guide.md`](reproduction_guide.md) | full reproduction guide (original root README), including the Kvasir 1% anchor launch commands |
| [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) | directory map and old → new path mapping |

## Method

| Document | Contents |
|---|---|
| [`method_en.md`](method_en.md) | complete current method (student-audited pseudo-video SAM3, Kvasir 1%) |
| [`method_cn.md`](method_cn.md) | 方法说明（中文） |
| [`method_cn_v2.md`](method_cn_v2.md) | 方法说明 v2（中文） |
| [`method_ladder.md`](method_ladder.md) | the full B00 → B7 experimental ladder |
| [`b2_ft_method.md`](b2_ft_method.md) | B2 fine-tuning method |
| [`kvasir_1pct_anchor.md`](kvasir_1pct_anchor.md) | Kvasir 1% anchor protocol reference |
| [`phase1_kvasir1pct_student.md`](phase1_kvasir1pct_student.md) | phase-1 student pipeline for Kvasir 1% |

## Encoder kNN / routing experiments

| Document | Contents |
|---|---|
| [`knn_experiment_vitb256.md`](knn_experiment_vitb256.md) | ViT-B/256 frozen-feature kNN routes |
| [`knn_experiment_sam3enc_imr_s256.md`](knn_experiment_sam3enc_imr_s256.md) | SAM3 encoder, image-mask retrieval at 256 |
| [`knn_experiment_sam3enc_anchor_ablation.md`](knn_experiment_sam3enc_anchor_ablation.md) | anchor ablation for SAM3-encoder routes |
| [`knn_experiment_negcontrol_transport.md`](knn_experiment_negcontrol_transport.md) | negative control for the transport proxy |
| [`knn_experiment_lesion_mechanism.md`](knn_experiment_lesion_mechanism.md) | lesion-correspondence mechanism analysis |
| [`route_selector_experiment_vitb256.md`](route_selector_experiment_vitb256.md) | ViT-B/256 route selector |
| [`routeco_sam3_v1.md`](routeco_sam3_v1.md) | RouteCo SAM3 v1 |

## `docs/v2/` — earlier document series

The numbered series that preceded the current method documents. Kept for
provenance; some conclusions were superseded.

| Document | Contents |
|---|---|
| `01_sam3_encoder_knn_sam3_256_eval.md` | SAM3 encoder kNN routes + 256×256 forward Dice |
| `02_sam3_base_b0b6_fusion.md` | SAM3-base b0–b6 fusion |
| `03_sam3_lora_b0b6_fusion.md` | SAM3-LoRA b0–b6 fusion |
| `04_qwen_text_knn_first_eval.md` | Qwen text kNN, first evaluation |
| `05_hybrid_sam3_text_knn_lambda02.md` | hybrid SAM3 + text kNN (λ = 0.2) |
| `06_mask_visual_qwen_hybrid_lambda02.md` | mask-visual Qwen hybrid |
| `07_mask_visual_lambda_scan.md` | mask-visual λ scan |
| `08_triple_visual_qwen_direct_test.md` | triple visual Qwen direct test |
| `09_msr_sam3_two_stage_plan.md` | MSR + SAM3 two-stage plan |
| `10_lesion_knn_mechanism_handoff.md` | lesion kNN mechanism handoff |
| `11_c3_path_invariance_handoff.md` | C3 path-invariance handoff |
