# 环节一：SAM3-base 候选生成与 Router——新老版本对比

更新日期：2026-09-12。数据集：Kvasir。本文只比较候选生成、正向/返回传播与无学生 Router；不包含伪标签入池、S2/S3、委员会、X3、B7 或 LoRA 微调。

## 1. 新老版本具体指什么

| 项目 | 老版 | 新版 |
|---|---|---|
| 路线 | Target Pooling（TP）＋Patch Correspondence（PC） | 仅 Target Pooling（TP） |
| 桥长 | 两条路线各 b0–b6 | TP b0–b6 |
| 每图候选 | 14 张 | 7 张 |
| 最终选择 | TP＋PC 联合 Router | TP 独立 Router |
| 最终 mask | 最高 Router 分数对应的一张 SAM3 二值 mask | 相同 |
| 是否融合 mask | 否 | 否 |
| 是否使用学生 | 否 | 否 |

这里的“老版”是原 SAM3 trunk@256 的双路线版本，不是更早 DINOv3 构路版本，也不是 Round2A 的 epoch33 微调教师。

新版沿用老版 TP 的候选生成方式和候选 mask。该环节的变化是移除 PC 并使用 TP 独立 Router，不是更新 SAM3 权重或改进 TP 本身。

`b0` 表示 anchor→target，没有中间桥；`b1–b6` 表示加入 1–6 张训练集桥接图。**b0 仍是有参考标注的传播，不是 SAM3 单图无参考直接预测。**

## 2. 统一测试标准

| 项目 | 两版共同设置 |
|---|---|
| 数据划分 | 原 train/validation/test = 800/100/100 |
| 标注身份 | 原 8 张人工标注；训练集中另有 792 张无标注图 |
| 测试集 | 同一批完整 100 张 test，质量较差或空预测也参与统计 |
| SAM3 checkpoint | 原始未微调 `sam3.pt` |
| 检索特征 | SAM3-base trunk，输入 256，patch_mean，beam width=32 |
| 传播画布 | 256×256；这不表示 SAM3 视频模型内部原生输入变为 256 |
| 传播提示 | anchor 人工标注导出的框；本环节没有 `colon polyp` 文本提示 |
| 桥接图范围 | 训练集；目标 test GT 不用于路径搜索或传播提示 |
| 指标尺寸 | 预测和 GT 均为 256×256；需要缩放时使用 NEAREST |
| 二值化 | 灰度 mask >127 |
| Dice | 每图 `2×交集/(预测面积+GT面积)`，再对 100 图宏平均 |
| IoU | 每图 `交集/并集`，再对 100 图宏平均 |
| 空分母处理 | 分母取 `max(分母,1)`；此数据集 GT 为非空病灶 |
| Router 拟合 | 原 validation；Ridge 正则系数 1.0 |
| Router 并列处理 | 最大化 `(score, -bridge_count, route_id)` |
| Oracle | 每张图用 GT 找出候选中 Dice 最高的一张后平均；仅事后分析 |

Router 拟合会使用 validation 候选的 GT Dice。测试选择不使用 test GT；本次复算先固定全部选择，再打开 test GT 计算指标。返回一致性使用的是已标注 anchor GT，不能把它与目标 test GT 混为一谈。

测试标准相同，但两个 Router 参数并不相同：联合版用 TP＋PC 的 1400 条 validation 候选并加入模式特征，TP 版用 700 条 TP 候选、不含模式特征。这是分别匹配 Router 的方案比较，不是固定同一个 Router、只删除 PC 的单因素消融。

## 3. b0–b6 完整 Test 结果

### 3.1 新老 Dice 直接对照

每条路线、每个桥长都是 100 张 test。旧 TP 与新 TP 逐像素一致，因此七行差值均为 0。

| 桥长 | 老版 TP Dice | 老版 PC Dice | 老版双路线均值 | 新版 TP Dice | 新 TP－旧 TP |
|---|---:|---:|---:|---:|---:|
| b0 | 0.798144 | 0.706645 | 0.752395 | 0.798144 | 0.000000 |
| b1 | 0.846623 | 0.754897 | 0.800760 | 0.846623 | 0.000000 |
| b2 | 0.870032 | 0.788470 | 0.829251 | 0.870032 | 0.000000 |
| b3 | 0.862253 | 0.859413 | 0.860833 | 0.862253 | 0.000000 |
| b4 | 0.873930 | 0.857625 | 0.865778 | 0.873930 | 0.000000 |
| b5 | 0.869552 | 0.855339 | 0.862446 | 0.869552 | 0.000000 |
| b6 | 0.874004 | 0.853581 | 0.863793 | 0.874004 | 0.000000 |

**“双路线均值”只是同一桥长下 200 张候选结果的平均，不是融合 mask、不经过 Router，也不是逐图在 TP 和 PC 中选最好。**后面的联合 Router 才是每图选一个候选的实际预测成绩。

### 3.2 TP 完整指标：新老相同

| 桥长 | 数量 | Dice | IoU | Dice 中位数 | 平均返回一致性 q_cycle |
|---|---:|---:|---:|---:|---:|
| b0 | 100 | 0.798144 | 0.739434 | 0.946492 | 0.856412 |
| b1 | 100 | 0.846623 | 0.783312 | 0.946073 | 0.868034 |
| b2 | 100 | 0.870032 | 0.806411 | 0.946816 | 0.899518 |
| b3 | 100 | 0.862253 | 0.800541 | 0.952737 | 0.916853 |
| b4 | 100 | 0.873930 | 0.813899 | 0.948239 | 0.924861 |
| b5 | 100 | 0.869552 | 0.808453 | 0.949360 | 0.939041 |
| b6 | 100 | 0.874004 | 0.811779 | 0.948556 | 0.937421 |

### 3.3 老版 PC 完整指标

| 桥长 | 数量 | Dice | IoU | Dice 中位数 | 平均返回一致性 q_cycle |
|---|---:|---:|---:|---:|---:|
| b0 | 100 | 0.706645 | 0.650366 | 0.935329 | 0.752706 |
| b1 | 100 | 0.754897 | 0.699311 | 0.941709 | 0.742701 |
| b2 | 100 | 0.788470 | 0.730725 | 0.942226 | 0.642437 |
| b3 | 100 | 0.859413 | 0.795500 | 0.943405 | 0.637434 |
| b4 | 100 | 0.857625 | 0.795786 | 0.944580 | 0.627593 |
| b5 | 100 | 0.855339 | 0.793748 | 0.941804 | 0.619069 |
| b6 | 100 | 0.853581 | 0.790866 | 0.946291 | 0.628343 |

### 3.4 老版双路线候选均值：描述性统计

| 桥长 | 候选数量 | Dice | IoU | Dice 中位数 | 平均返回一致性 |
|---|---:|---:|---:|---:|---:|
| b0 | 200 | 0.752395 | 0.694900 | 0.940924 | 0.804559 |
| b1 | 200 | 0.800760 | 0.741311 | 0.943610 | 0.805367 |
| b2 | 200 | 0.829251 | 0.768568 | 0.944863 | 0.770978 |
| b3 | 200 | 0.860833 | 0.798021 | 0.946810 | 0.777143 |
| b4 | 200 | 0.865778 | 0.804843 | 0.945365 | 0.776227 |
| b5 | 200 | 0.862446 | 0.801101 | 0.945890 | 0.779055 |
| b6 | 200 | 0.863793 | 0.801322 | 0.947342 | 0.782882 |

q_cycle 是返回 anchor 后与 anchor GT 的一致性，不是目标图 Dice，也不是候选被选中的概率。中位数对 200 个候选直接计算，不是两个路线中位数的平均。

## 4. Router 和 Oracle 完整结果

### 4.1 主对照：b0–b6

| 设置 | 每图候选 | Router Test Dice | Router Test IoU | Oracle Dice | Oracle－Router |
|---|---:|---:|---:|---:|---:|
| 老版：TP＋PC 联合 Router | 14 | 0.874083 | 0.813986 | 0.921471 | 0.047388 |
| 老版 PC 单独 Router，诊断对照 | 7 | 0.861688 | 0.799622 | 0.911715 | 0.050026 |
| 新版：TP 独立 Router；也等于旧池的 TP 子集对照 | 7 | **0.885433** | **0.828274** | 0.907426 | 0.021994 |

差值使用原始精度计算，而不是先把表中各项截断后相减：

| 新版相对老版联合 Router | 变化 |
|---|---:|
| Router Dice | +0.011349612483，约 **+1.1350 个百分点** |
| Router IoU | +0.014288395703，约 +1.4288 个百分点 |
| Oracle Dice | -0.014044355018，约 **-1.4044 个百分点** |
| Oracle－Router 差距 | 缩小 0.025393967502 |

此前口头比较用六位小数相减，出现过 gap=0.021993、Oracle 下降 1.4045 个百分点的末位差异。本表以原始数值计算：TP gap 四舍五入为 **0.021994**，Oracle 下降约 **1.4044** 个百分点。

### 4.2 历史候选范围补充：b3–b6

这组从同一批 b0–b6 结果取子集，并使用对应候选范围的历史 Router；不是另一次传播，也不能与 b0–b6 混作同配置。

| 设置 | 每图候选 | Router Test Dice | Router Test IoU | Oracle Dice | Oracle－Router |
|---|---:|---:|---:|---:|---:|
| TP＋PC | 8 | 0.881029 | 0.817542 | 0.919337 | 0.038307 |
| TP | 4 | 0.876869 | 0.815768 | 0.902689 | 0.025821 |
| PC | 4 | 0.874294 | 0.810286 | 0.895989 | 0.021695 |

## 5. 可复现性：已经完成哪些核查

1. 2026-09-06 曾对原固定 test routes 重新执行正向和返回传播，TP、PC 各 700 条，全部成功。
2. 2026-09-12 逐像素核验重新传播与更早候选：**1400/1400 张 mask 相同**。
3. 使用原 Python3.12 环境、原 validation 质量记录重新拟合 b0–b6 的 TP 与联合 Router：**参数最大绝对差为 0**。
4. 两版最终选路分别与存档 **100/100 一致**；代码、输入质量文件和最终 mask 哈希通过核查。
5. 本文附带的统一评估入口已经分别运行 old/new 两组，重算了全部固定桥 Dice、IoU、中位数、返回一致性，以及 b0–b6/b3–b6 Router 和 Oracle。保存的结果位于本文目录下 `核验数据/`。

今天新增的是离线复算与脚本核验，没有再次启动 GPU 传播。下文“重新执行传播”命令经过实际参数与历史进程记录核对，但本轮没有再次执行该部分。应使用原 `sam3` Python3.12 环境；另一 Python3.9 环境的严格参数相等检查曾未通过。

该 test 已用于多轮研究比较；结果可重现，不等于已经证明在独立新数据上稳定优于旧版。

## 6. 服务器路径与版本锁定

以下命令均在服务器 **Bash** 中运行，不直接粘贴到 Windows PowerShell。先登录并进入 Bash：

```bash
ssh violet@222.31.141.50
bash
```

所有后续命令共用这段设置，每个新终端需执行一次：

```bash
set -euo pipefail
PROJECT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY_SAM3=/home/violet/anaconda3/envs/sam3/bin/python
BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
SNAP="$PROJECT/work/rerun_kvasir_sam3base_test_20260906"
ARCHIVE="$PROJECT/new_project/stage1_old_new_comparison_20260912"
EVAL="$ARCHIVE/scripts/evaluate_stage1.py"
export PYTHONPATH="/Data_8TB/lht/sam3:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"
```

| 资产 | 位置或 SHA256 |
|---|---|
| 旧候选及 validation | `$PROJECT/work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6` |
| 9月6日重新传播结果 | `$SNAP/quality_root` |
| 冻结 Router 与原评估代码 | `$SNAP/final_masks_no_student` |
| 本次整理的统一评估脚本 | `$EVAL` |
| 本次已验证老版结果 | `$ARCHIVE/verified_old/results.json` |
| 本次已验证新版结果 | `$ARCHIVE/verified_new/results.json` |
| base checkpoint SHA256 | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| TP test routes SHA256 | `89db1347a13699f139cad501b5a6aa0aaee7b1cd6ddb621407dd6e31515e6e6f` |
| PC test routes SHA256 | `15708488c9eb1cb7155e3157de00939e048d50b3958447546556b9a63fbe4e54` |
| 原 Router 代码 SHA256 | `b77c8fb8347424a7f0216d3c0bb978a3f78f219ef31661728a02531e2032888d` |

## 7. 启动方式一：使用已有候选，重算环节一结果

这是最快的结果复现方式，不需要 GPU。统一入口是本次整理的脚本，读取**历史冻结的 Router 参数**，没有重新训练评分器或改变选路规则。`--variant old` 评估联合、TP 和 PC；`--variant new` 只读取 TP 候选并评估 TP。

### 7.1 老版：TP＋PC

```bash
OLD_REPLAY=$(mktemp -d "$ARCHIVE/old_eval_XXXXXX")
"$PY_SAM3" "$EVAL" \
  --variant old \
  --quality-root "$SNAP/quality_root" \
  --output-dir "$OLD_REPLAY/results"
```

预期：联合 b0–b6 Dice **0.874083**、Oracle **0.921471**。同时输出 TP、PC 固定 b0–b6 和单路线 Router 诊断。

### 7.2 新版：TP-only

```bash
NEW_REPLAY=$(mktemp -d "$ARCHIVE/new_eval_XXXXXX")
"$PY_SAM3" "$EVAL" \
  --variant new \
  --quality-root "$SNAP/quality_root" \
  --output-dir "$NEW_REPLAY/results"
```

预期：TP b0–b6 Dice **0.885433**、Oracle **0.907426**。

两版都会输出：`results.json`、`per_candidate_metrics.json`、`INPUTS_AND_CHOICES_FROZEN.json`、分组 Router 参数、最终选路清单、最终 mask、逐图指标及 `COMPLETE`。输出子目录必须不存在；使用 `mktemp` 每次创建新目录，历史结果保持不动。

## 8. 启动方式二：重新执行 test 正向和返回传播

本节使用原来固定的 routes，重新跑 SAM3。这样控制检索路径不变，直接复现本文候选。原始路径搜索如何启动另见第9节。

### 8.1 老版：两条路线都重新传播

准备独立目录，复制冻结路线；检查动态传播依赖仍为原版：

```bash
OLD_RUN=$(mktemp -d "$ARCHIVE/old_gpu_replay_XXXXXX")
mkdir -p "$OLD_RUN/logs"
for mode in sam3enc_anchor_conditioned_target_pooling sam3enc_anchor_conditioned_patch_correspondence; do
  mkdir -p "$OLD_RUN/quality_root/$mode/test_pool0_stage1"
  cp "$SNAP/quality_root/$mode/test_pool0_stage1/routes.jsonl" \
     "$OLD_RUN/quality_root/$mode/test_pool0_stage1/routes.jsonl"
done
cmp "$SNAP/code/run_t21_dynamic_pseudovideo.py" "$PROJECT/scripts/run_t21_dynamic_pseudovideo.py"
```

TP 用 GPU0、PC 用 GPU1。以下仅指定可见卡号；执行前确认卡可用，不停止其他任务：

```bash
CUDA_VISIBLE_DEVICES=0 "$PY_SAM3" -u \
  "$SNAP/code/eval_route_propagation_quality.py" \
  --checkpoint "$BASE" \
  --mode sam3enc_anchor_conditioned_target_pooling \
  --root "$OLD_RUN/quality_root" --split test --canvas 256 \
  > "$OLD_RUN/logs/tp.log" 2>&1 &
TP_PID=$!

CUDA_VISIBLE_DEVICES=1 "$PY_SAM3" -u \
  "$SNAP/code/eval_route_propagation_quality.py" \
  --checkpoint "$BASE" \
  --mode sam3enc_anchor_conditioned_patch_correspondence \
  --root "$OLD_RUN/quality_root" --split test --canvas 256 \
  > "$OLD_RUN/logs/pc.log" 2>&1 &
PC_PID=$!

wait "$TP_PID"
wait "$PC_PID"

"$PY_SAM3" "$EVAL" \
  --variant old \
  --quality-root "$OLD_RUN/quality_root" \
  --output-dir "$OLD_RUN/evaluation"
```

预期两路线各完成 700 条。没有 `--no-cycle`，因此包括返回传播；没有 `--resume`，因为这里使用全新目录。历史脚本会在候选生成后写目标 GT 的评估字段；这些字段不参与 Router 测试评分。

### 8.2 新版：只重新传播 TP

```bash
NEW_RUN=$(mktemp -d "$ARCHIVE/new_gpu_replay_XXXXXX")
MODE=sam3enc_anchor_conditioned_target_pooling
mkdir -p "$NEW_RUN/logs" "$NEW_RUN/quality_root/$MODE/test_pool0_stage1"
cp "$SNAP/quality_root/$MODE/test_pool0_stage1/routes.jsonl" \
   "$NEW_RUN/quality_root/$MODE/test_pool0_stage1/routes.jsonl"
cmp "$SNAP/code/run_t21_dynamic_pseudovideo.py" "$PROJECT/scripts/run_t21_dynamic_pseudovideo.py"

CUDA_VISIBLE_DEVICES=0 "$PY_SAM3" -u \
  "$SNAP/code/eval_route_propagation_quality.py" \
  --checkpoint "$BASE" --mode "$MODE" \
  --root "$NEW_RUN/quality_root" --split test --canvas 256 \
  > "$NEW_RUN/logs/tp.log" 2>&1

"$PY_SAM3" "$EVAL" \
  --variant new \
  --quality-root "$NEW_RUN/quality_root" \
  --output-dir "$NEW_RUN/evaluation"
```

预期完成 TP 700 条。新老版本的 TP 传播调用完全相同，仅输出目录不同；新版不执行 PC。

以上传播参数与历史 `*.process.json` 中实际命令一致，改动仅为输出目录和日志位置。依赖检查失败时，应先恢复存档版本到独立副本并调整依赖路径，不应直接覆盖旧项目代码。

## 9. 原始构路命令：从冻结 SAM3 特征重新搜索 test 路径

历史入口为：`$PROJECT/scripts/run_c0_256_sam3knn_s256_routes.sh`，实际调用 `scripts/stage1_feature_knn_routes.py`。历史入口会处理 train/validation/test，并使用固定历史输出目录，因此这里给出从其真实参数整理的**独立 test 输出版本**。

本步骤重新搜索路径，不重新提取特征。本文验证的主复现使用第8节已冻结 routes；本轮没有重新执行本节的搜索。

先准备独立目录及原特征缓存：

```bash
ROUTE_REBUILD=$(mktemp -d "$ARCHIVE/routes_rebuild_XXXXXX")
mkdir -p "$ROUTE_REBUILD/features"
cp "$PROJECT/work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features/sam3_base_s256_features.npz" \
   "$ROUTE_REBUILD/features/sam3_base_s256_features.npz"
```

老版构路：

```bash
for mode in sam3enc_anchor_conditioned_target_pooling sam3enc_anchor_conditioned_patch_correspondence; do
  "$PY_SAM3" scripts/stage1_feature_knn_routes.py \
    --mode "$mode" --feature-source sam3_base --feature-size 256 \
    --knn-feature patch_mean --split test \
    --min-bridge 0 --max-bridge 6 --beam-width 32 \
    --protocol-root "$PROJECT/work/kvasir_1pct_anchors/protocol" \
    --output-root "$ROUTE_REBUILD"
done
```

新版在另一个新建的 `ROUTE_REBUILD` 目录中只运行 TP：

```bash
"$PY_SAM3" scripts/stage1_feature_knn_routes.py \
  --mode sam3enc_anchor_conditioned_target_pooling \
  --feature-source sam3_base --feature-size 256 \
  --knn-feature patch_mean --split test \
  --min-bridge 0 --max-bridge 6 --beam-width 32 \
  --protocol-root "$PROJECT/work/kvasir_1pct_anchors/protocol" \
  --output-root "$ROUTE_REBUILD"
```

重新搜索后应先核对每模式 700 条、同一 100 张目标及 b0–b6 覆盖，并比较 source/bridge/target 序列，再使用第8节传播命令、将 `--root` 指向该目录。若生成记录含输出绝对路径，仅文件哈希不同不能直接推断实际检索路径不同。

## 10. 原历史脚本与本次入口的关系

| 工作 | 原历史文件 | 本文对应入口 |
|---|---|---|
| SAM3 特征构路 | `scripts/run_c0_256_sam3knn_s256_routes.sh` | 第9节，真实参数拆解到新目录 |
| 重跑两路线 test | `$SNAP/run.py` 的 `prepare` / `run` | 第8节，复制固定 routes 后调用原传播脚本 |
| 无学生 Router 拟合与评估 | `$SNAP/final_masks_no_student/evaluate.py` | 原代码在 validation 上拟合并存档六组 Router |
| 本次直接重算两版 | `$ARCHIVE/scripts/evaluate_stage1.py` | 第7节，加载冻结 Router、固定选择后重算指标 |

不要直接重新执行 `$SNAP/run.py prepare` 或原 `evaluate.py`：它们的历史输出目录已经存在，原脚本会拒绝覆盖。本文命令通过新目录运行。

`scripts/run_c0_256_sam3knn_s256_current_test_gpu1.sh` 还会导出 X3、计算 B7，因此不作为本文环节一的启动入口。

## 11. 本地交付与结论

本地目录：`F:\medsam3\新老环节对比`。

- `环节一.md`：本文。
- `scripts/evaluate_stage1.py`：已在服务器分别验证 old/new 的统一评估入口。
- `核验数据/old_results.json`：老版全部原始精度指标。
- `核验数据/new_results.json`：新版全部原始精度指标。

服务器整理目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/stage1_old_new_comparison_20260912`。

环节一的确定结论是：TP 候选本身未变；从双路线联合 Router 改为 TP 独立 Router 后，实际 Dice 提高约 1.1350 个百分点，候选数减少一半，但 Oracle 下降约 1.4044 个百分点。不能将 Oracle gap 缩小单独解释为 Router 算法变强，因为候选池和拟合范围也改变了。
