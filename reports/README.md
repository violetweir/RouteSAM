# `reports/` — reproduction reports

Per-experiment Markdown reports. They are the narrative record behind the numbers
in `paper/` and the raw summaries in `results/`. Most are written in Chinese.

## C0 / C0-256 mainline reproductions

| Report | Title |
|---|---|
| `512_C0_reproduction.md` | 512_C0 Baseline 完整复现报告（详细版） |
| `C0_256_reproduction.md` | C0_256 Baseline 完整复现报告（详细版） |
| `C0_256_base_reproduction.md` | C0-256-base（未微调 SAM3 + 256×256）完整复现报告 |
| `C0_256_sam3knn_s256_b0_b6_run.md` | C0-256-base → SAM3-KNN@256, routes b0-b6 |
| `C0_256_sam3knn_s256_b0_b6_test_details.md` | C0-256 SAM3-KNN@256 b0-b6 current-route test details |
| `C0_256_base_B7_LoRA_e50_run.md` | C0-256-base → X3-best+B7 → SAM3 full-module LoRA (50 epochs) |
| `C0_256_full_experiment_history.md` | C0-256 系列完整实验记录：Base、全模块 LoRA、SAM3-KNN@256 与 b0-b6 |
| `C0_256_Round1_Round2A_complete_reproduction.md` | C0-256 全链路完整实验报告：Base、两轮全模块 LoRA、SAM3-KNN@256、Round-2A 与 Student-X4 |

## C0-256 routing and round 2/3 studies

| Report | Title |
|---|---|
| `C0_256_propagation_aware_routing.md` | 传播感知选路机制验证：KNN 图像相似性 vs SAM3 返回一致性 |
| `C0_256_propagation_risk_diagnosis.md` | 传播风险诊断：定位“高一致性但错误传播”的失败机制 |
| `C0_256_risk_gated_routing_validation.md` | 风险门控换路：validation target-disjoint OOF 反事实实验 |
| `C0_256_fpn_transport_e33_b4.md` | C0_256 SAM3-FPN foreground transport with e33 |
| `C0_256_round2a_fixed_knn_e33_x4.md` | Round-2A：固定 SAM3-base KNN topology，增强 propagation teacher，验证 X3→X4 反哺 |
| `C0_256_round2b_e33_knn_test_b0_b6.md` | Round-2B：SAM3-e33 KNN 与 SAM3-e33 teacher 的 Test b0-b6 对照 |
| `C0_256_round2b_topology_refresh_factorial.md` | Round-2B：SAM3-e33 KNN Topology Refresh 2×2 因子实验 |
| `C0_256_round2c_asymmetric_candidate_pool.md` | Round-2C：Asymmetric Correspondence Anchor Pool |
| `C0_256_round2c_lesion_correspondence_anchor_reranking.md` | Round-2C：双向病灶对应关系 Anchor Re-ranking Validation |
| `C0_256_round3_tracker_stage4.md` | Round3：Stage-IV Pseudo-Video Tracker Adaptation |
| `C0_256_round3_tracker_lora_t3_t4.md` | Round3 T3/T4：Constrained Pseudo-Temporal LoRA Adaptation |

## Kvasir SAM3-base and TP-only line

| Report | Title |
|---|---|
| `Kvasir_SAM3base_experiments_summary_20260906.md` | Kvasir-SEG：SAM3-base 当前实验汇总 |
| `Kvasir_SAM3base_20260906_direct_text_ablation.md` | SAM3-base 单图预测：空文本与 colon polyp |
| `Kvasir_SAM3base_20260906_no_student_final_masks.md` | SAM3-base：无学生网络的最终 mask 评测 |
| `Kvasir_single_TP_mainline_20260909.md` | Kvasir 单 TP 主线确认 |
| `Kvasir_TP_student_mainline_20260907.md` | Kvasir TP-only 学生主线完成报告 |
| `Kvasir_TP_only_students_pipeline_CN_20260909.md` | TP-only 伪标签 → S2/S3 → 委员会 → X3 → B7 的实际流程 |
| `Kvasir_TP_only_vs_dual_route_20260908.md` | 旧双路线与新 TP-only 学生主线对比 |
| `Kvasir_TP_filterfirst_students_20260909.md` | 先筛返回候选再 Router 的完整学生对照 |
| `Kvasir_TP_filterfirst_X3_diagnosis_20260909.md` | 580 张版本 X3 退步核查 |
| `Kvasir_TP580_S2_S3_test_checkpoints_20260909.md` | 当前 580 张版本：S2/S3 冻结 checkpoint 的 test 评估 |
| `Kvasir_TP_router_pseudo_filter_diagnosis_20260909.md` | Router 用途与首批伪标签覆盖率诊断 |
| `Kvasir_X3_pool_v2_plan_20260909.md` | X3 训练池重构及 S2/S3 监督错位修复 |

## Kvasir TP + patch-correspondence studies

| Report | Title |
|---|---|
| `Kvasir_TP_guided_PC_joint_20260907.md` | TP-guided Patch Correspondence 统一选路实验 |
| `Kvasir_TP_PC_candidate_rerank_20260907.md` | 固定 TP 结果 + 全 7 个 PC 候选的相对收益重排序 |
| `Kvasir_TP_PC_gain_gate_20260907.md` | 保留 TP 的 PC 增益门控实验 |
| `Kvasir_joint_diagnosis_and_next_steps_20260907.md` | Kvasir 联合路线失败诊断与下一轮建议 |

## Kvasir diagnostics (September 2026)

| Report | Title |
|---|---|
| `Kvasir_8anchor_b0_transfer_validation_20260908.md` | 8-anchor b0 直接传播诊断 |
| `Kvasir_anchor_retrieval_path_factorial_validation_20260908.md` | 辅助参考图检索与传播路径固定预算对照 |
| `Kvasir_B7_candidate_oracle_gap_20260908.md` | 候选上限下降与 B7 选择损失诊断 |
| `Kvasir_B_target_quality_validation_20260908.md` | 冻结 B 候选池的目标端质量证据实验 |
| `Kvasir_PC_evidence_and_auxiliary_validation_20260908.md` | PC 对应证据改进与替代辅助机制，第一阶段验证 |
| `Kvasir_PC_prompt_refinement_native1008_validation_20260908.md` | 对应图生成框提示，SAM3 重解码实验 |
| `Kvasir_PC_rethink_anchor_diversity_20260908.md` | 重新判断 PC 的作用与下一轮改进方案 |

## See also

- `docs/s27_x3_b7_line.md`, `docs/protocol.md` — the frozen protocol these reports follow
- `docs/kvasir_1pct_anchor.md` — the Kvasir 1% anchor protocol reference
- `results/` — the raw summary JSON/CSV behind each report
- `mainline/` — the September+ follow-up studies (TP-only pool, cross-dataset)
