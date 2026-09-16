# Kvasir SAM3-base：无学生网络的最终 mask 评测

使用本次重新传播的 test 候选，复现旧报告的传播质量 Router。未加载学生 checkpoint 或学生预测，未使用 B7。

Router 为 ridge=1.0 的岭回归评分器，只在原历史 validation 传播记录上拟合；使用回环、路径、mask 轨迹、SAM 分数等特征。测试选路函数不接收 target GT 指标或 GT 路径。先保存最终选路，再逐像素重新计算 Dice/IoU。

每个设置覆盖全部 100 张原 test 图。b3-b6 为旧报告无学生阶段的设置，b0-b6 为完整候选范围；二者均预先固定，没有根据 test 选择范围。单模式各自拟合 Router，联合模式额外使用模式特征。

| 候选范围 | 候选路线 | Test Dice | Test IoU | Oracle（仅分析） |
|---|---|---:|---:|---:|
| b3-b6 | target_pooling | 0.876869 | 0.815768 | 0.902689 |
| b3-b6 | patch_correspondence | 0.874294 | 0.810286 | 0.895989 |
| b3-b6 | combined | 0.881029 | 0.817542 | 0.919337 |
| b0-b6 | target_pooling | 0.885433 | 0.828274 | 0.907426 |
| b0-b6 | patch_correspondence | 0.861688 | 0.799622 | 0.911715 |
| b0-b6 | combined | 0.874083 | 0.813986 | 0.921471 |

最终 mask 是 Router 分数最高的一个 SAM3 候选 mask，直接保存为 256×256 二值 PNG，没有像素融合。每组 mask、逐图选路与逐图指标保存在对应 b*_b6/<scope>/ 子目录。

来源：README.md 的 Candidate-Invariant Propagation-Quality Router，以及 reproduction_reports/C0_256_base_reproduction.md 的 Step 2；代码为 scripts/eval_ft1pct_pq_router.py 和 scripts/analyze_propagation_quality_router.py。历史 DINO 与本次 SAM3-base@256 选路特征不同，不能将不同版本的报告数值视为同一配置。
