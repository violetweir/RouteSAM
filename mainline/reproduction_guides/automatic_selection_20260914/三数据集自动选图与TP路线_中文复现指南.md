# BUSI、ISIC2018、Kvasir 自动参考图选择与 SAM3-base 单 TP 路线复现指南

更新日期：2026-09-14。本文面向能够登录同一服务器、但没有前面对话上下文的 agent。所有结果均来自现存实验文件。本文同时提供方法定义、固定名单、输入位置、运行命令和验收值。

## 1. 先确定复现哪个版本

三个数据集共用的主流程是：

```text
原有 train RGB
  → SAM3-base 1008 特征
  → 全局描述子 + 局部视觉词直方图
  → 贪心覆盖选择固定预算的参考图
  → 名单冻结后，读取这些参考图的 GT
  → SAM3-base 256 特征 + patch_mean KNN + Target Pooling 路径评分
  → 每张目标图生成 b0–b6 共 7 个候选 mask
  → 分别评价 b0–b6；可进一步用独立 Router 选择一个候选
```

| 版本 | Kvasir（息肉） | ISIC2018 | BUSI |
|---|---|---|---|
| 自动参考集合 | 8 张 | 21 张 | 5 张 |
| 自动选图 + 原始 TP b0–b6 | 已完成 | 已完成 | 已完成 |
| validation 拟合/选择独立 Router | 已完成 | 已完成 | 已完成 |
| 训练均值校准 + top2 参考图 + 14 候选 Router | 尚未做此版 | 尚未做此版 | 已完成，test Dice 0.752198 |

**BUSI 的 top2 不是重新选择两张标注图。** 它仍保留原来自动选出的 5 张标注参考图，再为每张目标图从这 5 张中选择最合适的 2 张，各生成 7 个候选。它属于下游参考匹配改进，不属于标注参考集合选择算法的改动。

本文不涉及 PC、S2/S3、学生网络、B7 或 SAM3 LoRA。下述主实验提示是参考图 **GT tight box、无文本**；不要替换成 full-mask prompt 或文本提示，后两者属于另外的消融实验。

## 2. 服务器、环境与公共变量

SSH：`violet@222.31.141.50`。以下命令均在服务器 Bash 中运行；本地 PowerShell 可先执行 `ssh violet@222.31.141.50`，进入后执行 `bash`。

```bash
export P=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export E="$P/new_project/experiments"
export G="$P/new_project/reproduction_guides/automatic_selection_20260914"
export SAM_PY=/home/violet/anaconda3/envs/sam3/bin/python
export CPU_PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
export BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
export MODE=sam3enc_anchor_conditioned_target_pooling
export PYTHONPATH=/Data_8TB/lht/sam3:$P/src
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd "$P"
```

当前核验环境：

| 用途 | 环境 |
|---|---|
| SAM3 特征和传播 | Python 3.12.13、torch 2.10.0+cu128、torchvision 0.25.0+cu128、numpy 1.26.4、Pillow 12.2.0、timm 1.0.28 |
| sklearn 聚类选图 | Python 3.9.25、numpy 1.22.4、scikit-learn 1.6.1 |
| SAM3 源码位置 | `/Data_8TB/lht/sam3` |
| 当前源码 Git HEAD | `5dd401d1c5c1d5c3eedff06d41b77af824517619`；HEAD 本身不代表工作区没有改动 |
| SAM3 checkpoint SHA256 | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |

聚类应使用 CPU_PY，sam3 环境没有 sklearn。请优先使用实验目录内冻结的 `code/`，不要直接使用可能被后续修改的项目公共脚本。

复现辅助脚本：`$G/reproduce_automatic_selection.py`。本地副本位于 `F:\medsam3\新老环节对比\scripts\reproduce_automatic_selection.py`。支持 `extract / select / prepare / evaluate` 四个子命令，输出必须指向新目录。

`$G/source_hashes.json` 保存本次读取的 checkpoint、协议、256 特征缓存、实验脚本、辅助入口以及当前 SAM3 Python 源码 SHA256，可先执行：

```bash
"$SAM_PY" - <<'PY'
import os,json,hashlib
from pathlib import Path
for name,info in json.loads((Path(os.environ['G'])/'source_hashes.json').read_text()).items():
    h=hashlib.sha256()
    with open(name,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    assert h.hexdigest()==info['sha256'],name
print('所有存档输入哈希一致')
PY
```

## 3. 数据与划分：manifest 是唯一依据

| 数据集 | train | validation | test | 训练参考预算 |
|---|---:|---:|---:|---:|
| Kvasir-SEG | 800 | 100 | 100 | 8，即 1.0000% |
| ISIC2018 | 2075 | 259 | 260 | 21，即 1.0120% |
| BUSI | 517 | 64 | 66 | 5，即 0.9671% |

预算取 `round(train数量 × 0.01)`。本轮是固定 1% 预算，**不是已经找到了自动停止的最小标注数量**。Kvasir/ISIC 曾做覆盖收益停止规则和预算曲线探索，但主传播实验使用固定 8/21，BUSI 使用固定 5。

真实数据位置：

- Kvasir：`/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T02_fresh_polyp_hf_sources/raw_hf_snapshots/kvasir-seg/snapshot/train/{images,masks}`。这里物理路径包含 `train`，但不代表全部属于本实验 train；必须看 manifest 的 `split`。
- ISIC：`/Data_8TB/lht/MK-UNet/data/ISIC18`，具体文件路径以 manifest 为准；例如 train 图像为 JPG，mask 为 PNG。
- BUSI：`/Data_8TB/lht/MK-UNet/BUSI/BUSI_split/{train,val,test}/{images,masks}`；如图像 `benign (3).png` 对应 `benign (3)_mask.png`。

完整协议、自动 support 和完整 256 缓存都可从以下目录读取：

| 数据集 | 相对 `$E` 的目录（下称 BASEDIR） |
|---|---|
| kvasir | `automatic_anchor_tp_test_20260913/kvasir` |
| isic2018 | `automatic_anchor_tp_test_20260913/isic2018` |
| busi | `busi_auto5_tp_1pct_20260913` |

每个 BASEDIR 内：

```text
protocol/merged_manifest.jsonl     # 全量划分，顺序也必须保留
protocol/support_manifest.jsonl    # 自动参考名单、image_path、frozen_mask_path
quality_root/features/sam3_base_s256_features.npz
code/stage1_feature_knn_routes.py
code/eval_route_propagation_quality.py
code/run_t21_dynamic_pseudovideo.py
```

`split` 字段使用 `train / validation / test`，不是 `val`。`merged_id` 必须原样保留：Kvasir 带 `kvasir-seg::`，BUSI 带 `BUSI::`，ISIC 直接为 `ISIC_...`。

不重新划分、不从文件夹随机抽取替代协议。缓存的行与 manifest 的行对应；不能排序 manifest 后仍直接使用旧缓存。Kvasir/ISIC 的完整缓存保留旧 train+validation 原值，只追加了 test 行。

BUSI 当前协议是 647 张含病灶图，不含 normal。训练中 `benign (433)` 与 `malignant (145)` 文件 SHA256 相同，候选集合只允许其中第一个出现的文件被选；其余所有训练样本仍参与覆盖目标、词典拟合及训练均值计算。原始 BUSI 部分病例有额外 `_mask_1` 等文件，本轮沿用 split 的主 mask，未合并；更改这一点会改变实验定义。

## 4. 自动选图算法的精确定义

### 4.1 选图前允许读取什么

只用 train RGB。选图不使用类别、名称的语义、GT 像素、伪标签、anchor 条件描述子、validation/test 特征或成绩。ID 只用于顺序和并列处理。名单冻结后，才读取选中参考图的 GT 来构建下游原型和提示。

历史 Kvasir 1008 缓存含全数据独立图像编码，但选择器只取 train 行，且只在 train 上拟合局部词典与 IDF。不能因为缓存中存在 test 行，就把它们用于聚类或覆盖优化。

### 4.2 SAM3-base 1008 描述子

1. 原图转 RGB，bicubic、antialias resize 到 `1008×1008`，转 `[0,1]`，再按 `(x-0.5)/0.5` 归一化。
2. 冻结 SAM3-base，使用 `model.detector.backbone.vision_backbone.trunk` 的第一个输出；72×72 个 patch，每个 1024 维。
3. 每个 patch 先 L2 归一化，整图描述子为 **归一化 patch 的均值再 L2 归一化**，不是先平均未归一化输出。
4. 局部描述子每图取 64 个 patch。行列都使用 `np.linspace(4,67,8).round().astype(int)`；按先行后列生成 `row*72+col` 索引。

选图分辨率 1008，与后续 KNN/传播 canvas=256 是不同环节，不要相互替换。

### 4.3 局部词典与全局/局部相似度

训练所有图像的 64 个局部向量再次按 numpy L2 归一化，展开为 `(N*64,1024)`，拟合：

```python
MiniBatchKMeans(
    n_clusters=64,
    random_state=2026,
    n_init=3,
    batch_size=2048,
    max_iter=100,
)
```

其余参数沿用上述 sklearn 版本默认值。不要改成另一种 KMeans 初始化或全量 KMeans。

每图统计 64 个采样向量的聚类频次，除以 64 得到 `hist`。对每个视觉词 c：

```text
df[c] = train 中 hist[c] > 0 的图像数
idf[c] = log((N+1)/(df[c]+1)) + 1
local[i] = L2_normalize(sqrt(hist[i] * idf))
global[i] = L2_normalize(global_feature[i])
S = 0.5 * clip(global @ global.T, 0, 1)
  + 0.5 * clip(local @ local.T, 0, 1)
```

这里的局部视觉词反映局部外观，并不保证只编码病灶或临床形态。保留原 numpy dtype 运算，尤其 IDF 不要额外强制转 float32，以免改变并列附近的选择。

### 4.4 贪心覆盖选图

输入 train 按 ID 排序，SHA256 相同的候选只保留第一次出现者。初始化 `best=zeros(N,float32)`，每一步：

```text
gain[j] = mean_i(max(S[i,j] - best[i], 0))
j* = argmax gain[j]，排除已选和完全重复文件候选
best[i] = max(best[i], S[i,j*])
```

迭代固定 K 次，K 为 8/21/5。`np.argmax` 对并列取最早下标，所以 train 顺序会影响结果。该目标覆盖全部 train，不只是未选图像；完全重复文件虽然不能同时被选，仍保留在覆盖目标中。

主方案为 `global_local_facility`。Kvasir/ISIC 的 `global_facility`、随机 100 组和覆盖预算曲线是对照分析，不是本文 b0–b6 使用的名单。

## 5. 冻结的参考名单（按贪心选择顺序）

### Kvasir：8 张

```text
kvasir-seg::cju2wxv0hxs2f09884w48v8fi
kvasir-seg::cju1c4fcu40hl07992b8gj0c8
kvasir-seg::cju42xpi8lw4w0871ve317a1p
kvasir-seg::cju5vi4nxlc530817uoqm2m7a
kvasir-seg::cju3v664kh0px0818y4y7wolf
kvasir-seg::cju8dpa89u6l80818dj6lldh9
kvasir-seg::cju2t62nq45jl0799odpufwx6
kvasir-seg::cju7ajnbo1gvm098749rdouk0
```

冻结源：`$E/automatic_anchor_selection_pilot_20260911/SELECTIONS_FROZEN.json` 的 `selected.global_local_facility[:8]`。

### ISIC2018：21 张

```text
ISIC_0007788
ISIC_0009941
ISIC_0013410
ISIC_0014026
ISIC_0001126
ISIC_0010023
ISIC_0015614
ISIC_0014438
ISIC_0002453
ISIC_0011229
ISIC_0013458
ISIC_0015995
ISIC_0009252
ISIC_0010263
ISIC_0014829
ISIC_0013399
ISIC_0012453
ISIC_0011129
ISIC_0015206
ISIC_0016048
ISIC_0010360
```

冻结源：`$E/isic2018_auto21_tp_validation_20260911/selection/SELECTIONS_FROZEN.json` 的 `selected.global_local_facility[:21]`。

### BUSI：5 张

```text
BUSI::benign (3)
BUSI::benign (125)
BUSI::benign (305)
BUSI::malignant (195)
BUSI::malignant (187)
```

冻结源：`$E/busi_auto5_tp_1pct_20260913/selection/SELECTIONS_FROZEN.json` 的 `selected_ids`。良性/恶性名称没有参与选图，3+2 是选择后的统计结果。

## 6. 可执行复现：重新聚类并核验自动名单

以下 CPU 命令重新训练词典、计算 IDF、执行贪心，不直接抄名单；使用历史 1008 原始特征缓存，不打开 GT。首次核验建议按此方式排除 GPU 浮点差异。

```bash
export REPRO="$E/reproduce_auto_selection_$(date +%Y%m%d_%H%M%S)"
mkdir "$REPRO"
for DATASET in kvasir isic2018 busi; do
  "$CPU_PY" "$G/reproduce_automatic_selection.py" select \
    --dataset "$DATASET" --out "$REPRO/${DATASET}_selection" \
    > "$REPRO/${DATASET}_selection.log" 2>&1 || exit 1
done
```

验收文件：每个输出目录的 `selection_replay.json`。要求 `exact_match=true`，顺序也完全相同；`mask_pixels_read=0`。同时保存重算的词典、直方图和 greedy trace。原环境和缓存下，词典及直方图最大绝对误差应为 0。

缓存来源：

| 数据集 | 1008 原始特征来源 |
|---|---|
| Kvasir | `$P/work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/` 中 `sam3_base_s1008_features.npz`、`sam3_base_s1008_patches.npz`、`sam3_base_s1008_patches_raw.npy`；按 pilot 的 `train_images_only.json/cache_index` 取 train |
| ISIC | `$E/isic2018_auto21_tp_validation_20260911/selection_features.npz`，含 `ids/global_features/sampled_patches` |
| BUSI | `$E/busi_auto5_tp_1pct_20260913/selection_features.npz`，同上 |

### 可选：从 RGB 重新提取 1008 特征

这是更完整但更耗时的复现层级。先用 `nvidia-smi` 确认所选显卡有足够空闲显存，手动指定空闲卡；不要终止别人的任务。

```bash
export CUDA_VISIBLE_DEVICES=1  # 改为空闲卡
export DATASET=busi           # 可改 kvasir 或 isic2018
"$SAM_PY" "$G/reproduce_automatic_selection.py" extract \
  --dataset "$DATASET" --out "$REPRO/${DATASET}_fresh1008"
"$CPU_PY" "$G/reproduce_automatic_selection.py" select \
  --dataset "$DATASET" \
  --features "$REPRO/${DATASET}_fresh1008/selection_features.npz" \
  --out "$REPRO/${DATASET}_fresh_selection"
```

新入口按历史 ISIC/BUSI 的 batch1、FP32、无 autocast 提取方式实现。Kvasir 原始大缓存的生成历史与统一入口可能存在计算批次差异；因此，从 RGB 重提取不能预先承诺逐位一致。程序会核验最终名单，不一致时停止，不应悄悄换名单继续报“完全复现”。本次文档交付的执行核验范围在最后一节列明。

## 7. 从参考图到 TP b0–b6：具体做法

### 7.1 256 特征、前景原型与 Target Pooling

KNN/TP 使用输入 256，patch grid 为 `256//14=18`，每图 324 个 1024 维归一化向量。`patch_mean` 是归一化 patch 均值，再 L2 归一化。桥接图只来自 train。

对参考图 A，将其 GT nearest resize 到 18×18，取前景位置 patch 均值并归一化，得到 `p_A`。如果前景网格为空，历史脚本退回整图 patch 均值；复现时保留该规则。

```text
s_j = dot(p_A, x_j)
w_j = softmax(10 * s_j)
z(A,T) = L2_normalize(sum_j w_j * x_j)
TP(A,T) = dot(p_A, z(A,T))
```

缓存里同时可能有 `cond_correspondence`，但本实验模式只用 `cond_target`，不使用 PC 路线。

### 7.2 路径搜索

```text
b0：参考图 A → 目标 T
b1：参考图 A → train桥1 → T
...
b6：参考图 A → train桥1 → ... → train桥6 → T
```

每个 b 分别搜索，beam width=32。KNN 用整图 `patch_mean` 的余弦。对一个固定参考图，路径评分值包含：

- 每个桥和目标相对该参考图的 TP 分数；
- 相邻桥之间，以及最后桥到目标的 patch_mean 余弦。

按 `(这些值的最小值, 这些值的均值)` 字典序最大化。参考图与第一个桥的关系通过 TP 条件分数体现，不额外添加一条参考整图余弦边。邻居排序、反向扩展和并列处理保留冻结脚本实现。

原版对全部参考图比较后，每个 `(target,b)` 只留下最佳一条路径。因此每图是 7 个候选，不是 `参考数量×7` 个。

### 7.3 正向、返回与最终 mask

所选参考图 GT 经历史 `human_pool(support,512)` 处理后得到 tight box。正向将该框提示给 SAM3，沿构造的伪视频传播；canvas=256，SAM3 内部还会执行其自身输入处理。每帧使用历史脚本的最高分对象选择规则，目标帧的二值 mask 就是该路径的最终候选。

返回一致性：以目标预测 mask 初始化逆向传播，返回参考图后与参考 GT 算 Dice，记为 `q_cycle`。**返回时使用目标预测，不使用目标 GT**。返回过程用于质量特征，不替换前向目标 mask。

直接复用 `run_t21_dynamic_pseudovideo.py` 的初始化、对象选择和返回逻辑。不能擅自改成始终跟踪 object0；那是另一套 full-mask adapter 消融，会改变结果。

## 8. 可执行复现：路径、传播与 b0–b6 评分

先执行第 2 节公共变量和第 6 节，保留同一 shell 的 REPRO 变量。每次选择一个数据集，顺序跑 validation/test：

```bash
export DATASET=busi  # kvasir / isic2018 / busi
export OUT="$REPRO/${DATASET}_tp"
"$SAM_PY" "$G/reproduce_automatic_selection.py" prepare \
  --dataset "$DATASET" --selection "$REPRO/${DATASET}_selection" --out "$OUT"

export CUDA_VISIBLE_DEVICES=1  # 确认空闲后设置
for SPLIT in validation test; do
  "$SAM_PY" "$OUT/code/stage1_feature_knn_routes.py" \
    --mode "$MODE" --feature-source sam3_base --feature-size 256 \
    --knn-feature patch_mean --beam-width 32 --min-bridge 0 --max-bridge 6 \
    --split "$SPLIT" --protocol-root "$OUT/protocol" \
    --output-root "$OUT/quality_root" \
    > "$OUT/logs/${SPLIT}_routes.log" 2>&1 || exit 1

  "$SAM_PY" "$OUT/code/eval_route_propagation_quality.py" \
    --checkpoint "$BASE" --mode "$MODE" --root "$OUT/quality_root" \
    --split "$SPLIT" --canvas 256 --no-target-gt --resume \
    > "$OUT/logs/${SPLIT}_propagation.log" 2>&1 || exit 1

  "$SAM_PY" "$G/reproduce_automatic_selection.py" evaluate \
    --dataset "$DATASET" --out "$OUT" --split "$SPLIT" \
    > "$OUT/logs/${SPLIT}_metrics.log" 2>&1 || exit 1
done
```

`prepare` 复制协议和冻结脚本，默认复制完整 256 缓存；不会复制旧预测，因此传播命令会真正重新生成 mask。若要重新提取 256 特征，创建另一新 OUT，在 prepare 命令增加 `--fresh-features`；其余不变。使用全量独立图像缓存不表示把 validation/test 当桥或当聚类训练数据。

结果位于：

```text
$OUT/quality_root/$MODE/${SPLIT}_pool0_stage1/routes.jsonl
$OUT/quality_root/$MODE/propagation_quality_${SPLIT}/propagation_quality.jsonl
$OUT/quality_root/$MODE/propagation_quality_${SPLIT}/forward_masks/
$OUT/${SPLIT}_PREDICTIONS_FROZEN.json
$OUT/${SPLIT}_per_candidate_metrics.json
$OUT/${SPLIT}_results.json
```

应有候选数：Kvasir val700/test700，ISIC val1813/test1820，BUSI val448/test462。全部状态必须 success，每个 `(target,b)` 恰好一条。不应按 q_cycle 或 Dice 阈值丢图。

### 统一评价标准

GT 转灰度后 nearest resize 到 256×256，`>127` 为前景；预测使用保存的二值前向 mask。逐图算：

```text
Dice = 2 * intersection / (pred_area + gt_area)
IoU  = intersection / union
```

无额外平滑项；分母为零时设为 1。之后对全部目标图宏平均，不是将整套数据像素汇总后算一次。Oracle 为每张图候选 Dice 最大值再平均，只用于事后上界诊断，不能拿来选实际输出。评价入口先冻结预测记录 SHA256，后读取目标 GT。

## 9. 原始自动选图版 test 验收结果

本表均为 **自动参考集合 + 原始单 TP，每图 7 候选**。

| 路径 | Kvasir Dice，100 张 | ISIC2018 Dice，260 张 | BUSI Dice，66 张 |
|---|---:|---:|---:|
| b0 | 0.804710 | 0.837421 | 0.458291 |
| b1 | 0.863804 | 0.853724 | 0.514197 |
| b2 | 0.858386 | 0.856643 | 0.514843 |
| b3 | 0.862556 | 0.854948 | 0.533733 |
| b4 | 0.874749 | 0.852281 | 0.517095 |
| b5 | 0.865332 | 0.851927 | 0.514702 |
| b6 | 0.887069 | 0.852864 | 0.539444 |
| 7 候选 Oracle | 0.917615 | 0.884316 | 0.617141 |

数据源：Kvasir/ISIC 为各 BASEDIR 的 `results.json/results/automatic`；BUSI 为 BASEDIR 的 `test_results.json`。原始浮点精度以 JSON 为准。

重算同一批保存的 mask 时应在浮点舍入范围内一致；从头 GPU 传播可能存在后端/版本差异。遇到差异先比较名单、manifest 哈希、route ID、mask 哈希，再看 Dice，不能以“接近”掩盖协议不一致。

## 10. 独立 Router：方法与单独复现

Router 只在每张图既有候选中选一张，不融合 mask，不丢弃 test 图。它使用额外 validation GT 拟合候选 Dice，所以 **1% 仅指训练参考图标注预算，不是整个系统只使用了 1% 标注**。

legacy：原 28 项特征（桥长、路径瓶颈/均值、SAM 置信度/候选数、返回一致性和传播形态变化等），标准化后 Ridge(alpha=1)，回归候选 Dice。rank_peer：再加 8 个候选间一致性/面积特征，特征和 Dice 在同图候选内中心化，alpha 比较 1、10、100。

按图像分组 5 折，seed2026，同图 7 个候选始终同折。每个训练折单独估计标准化参数，按 validation OOF 所选 mask 的平均 Dice 选择配置。额外 outer5/inner4 嵌套验证估计完整选参过程。最后用全部 validation 拟合并冻结模型，冻结 test 的具体 route 选择后读取 test 指标。

| 数据集 | val 选择的配置 | test Dice | test IoU |
|---|---|---:|---:|
| Kvasir | legacy_a1 | 0.871374 | 0.809810 |
| ISIC2018 | rank_peer_a100 | 0.866363 | 0.786488 |
| BUSI 原版 | legacy_a1 | 0.566808 | 0.480026 |

Kvasir 此 Router 低于固定 b6；ISIC 旧式重拟合 Router 为 0.866573，与选定 rank_peer 几乎相同，不能因此声称新 Router 普遍更好。不要把 2026-09-11 Kvasir 直接搬旧 Router 的开发集诊断分数当成这里的 OOF。

结果/脚本：

```text
$E/automatic_anchor_tp_router_20260913/run.py
$E/automatic_anchor_tp_router_20260913/{kvasir,isic2018}/
$E/busi_auto5_tp_router_20260913/run.py
$E/busi_auto5_tp_router_20260913/busi/
```

每个数据集目录含 `folds_frozen.json / models_frozen.json / test_choices_frozen.json / results.json`，以及各方法的 `per_target.json / masks/`。

复现 Router 可以直接使用原候选，无需再运行 GPU。以下只改脚本输出 R，保留原数据路径，输出到新目录：

```bash
"$SAM_PY" - <<'PY'
import os
from pathlib import Path
e=Path(os.environ['E']); dest=Path(os.environ['REPRO'])
for source,newname,oldline in [
 ('automatic_anchor_tp_router_20260913','router_kvasir_isic',"R=E/'automatic_anchor_tp_router_20260913'"),
 ('busi_auto5_tp_router_20260913','router_busi',"R=E/'busi_auto5_tp_router_20260913'")]:
    text=(e/source/'run.py').read_text()
    assert text.count(oldline)==1,(source,'输出路径声明不同，先检查快照')
    out=dest/newname; assert not out.exists()
    text=text.replace(oldline,'R=Path('+repr(str(out))+')')
    (dest/(newname+'.py')).write_text(text)
PY
"$SAM_PY" "$REPRO/router_kvasir_isic.py" > "$REPRO/router_kvasir_isic.log" 2>&1
"$SAM_PY" "$REPRO/router_busi.py" > "$REPRO/router_busi.log" 2>&1
```

这段命令是**历史候选上的 Router 重拟合复现**。若接入第 8 节新生成的候选，需要同步修改脚本 SOURCES/T 输入，并保持对应 `per_candidate_metrics.json` 的 route ID 一致，不可混用两次运行的元数据。

## 11. BUSI 的额外改进：分数校准和 top2 多参考候选

实验目录：`$E/busi_calibrated_multi_anchor_20260913`，实际入口为其中 `pipeline.py`。

### 11.1 为什么增加这一版

原始 TP 不同参考图的分数均值差异很大。BUSI 原版 test 的 b0 有 65/66 张选择 malignant (187)，全部 b0–b6 中 455/462 条路径来自该参考。其原始高分不代表分割质量最好。

固定原 5 张参考图，用 517 张 train RGB 的现有 TP 特征估计每个参考的均值：

```text
mean_A = mean(TP(A,x))，x 遍历 517 张 train
centered(A,T) = TP(A,T) - mean_A
```

这里不读取其余训练图 GT。每个目标按 centered 排序参考图，比较 top1/top2/top3/all5。每个保留参考各构建 b0–b6，因此每图候选分别为 7/14/21/35。

**路径内部仍用原始 TP 与原始相邻余弦进行瓶颈/均值评分。** 不把减均值后的近零数值直接与原始约 0.8 的边分数混合，否则改变了路径目标。

### 11.2 val 选择与最终设置

预先固定 13 个配置：原版 7 候选+legacy alpha1；每个 K=1/2/3/5 各比较 legacy alpha1、calibrated_peer alpha10/100。calibrated_peer 在原28项和8项 peer 特征上，再加原始/校准目标分数、减参考训练均值的路径瓶颈/均值、参考校准排名，共41项，并进行同图中心化。

val64 先生成 2240 个候选，包含原448个可精确复用的候选。图像分组 OOF 最终选择 **K=2、legacy alpha=1**，OOF Dice=0.738958；完整配置选择的 nested OOF=0.709939。新增特征并未胜出。

选定方案在 test 只用每图 top2 的14候选；为对照和消融，实际推理记录还包含旧原版候选，共1183条，其中旧462条复用、新721条传播。**1183/66 不是最终 Router 的每图候选数量**；选定池为66×14=924条，Oracle也是在这924条内算。

### 11.3 BUSI test 结果与正确解释

| 路径 | 原版单参考 Dice | 校准 top1 单参考 Dice |
|---|---:|---:|
| b0 | 0.458291 | 0.705708 |
| b1 | 0.514197 | 0.709401 |
| b2 | 0.514843 | 0.679150 |
| b3 | 0.533733 | 0.689764 |
| b4 | 0.517095 | 0.662677 |
| b5 | 0.514702 | 0.698703 |
| b6 | 0.539444 | 0.718360 |
| 该7候选 Oracle | 0.617141 | 0.775461 |

| 最终方案 | test Dice | test IoU | 候选池 Oracle |
|---|---:|---:|---:|
| 原版7候选 Router | 0.566808 | 0.480026 | 0.617141 |
| 校准 top2、14候选 Router | **0.752198** | **0.668155** | **0.823649** |

新版对旧 Router 的逐图配对 Dice 增量为0.185389，10000次 bootstrap 95%区间为[0.104277,0.270905]。Oracle与实际输出仍相差0.071452。表中的 top1 b0–b6 不是“每个 b 的 top2 Router”结果；后者没有在这里单独训练7个 Router。

### 11.4 可执行复现命令

复制已冻结 pipeline，仅更改输出 R。prepare 会复制代码和协议，run 会重建所有参考各自路径、核验旧路径/预测并复用完全相同候选，然后完成 val→Router→test。旧目录不覆盖。

```bash
"$SAM_PY" - <<'PY'
import os
from pathlib import Path
e=Path(os.environ['E']); dest=Path(os.environ['REPRO'])
source=e/'busi_calibrated_multi_anchor_20260913/pipeline.py'
text=source.read_text(); old="R = E / 'busi_calibrated_multi_anchor_20260913'"
assert text.count(old)==1
out=dest/'busi_calibrated'; assert not out.exists()
(dest/'busi_calibrated_driver.py').write_text(text.replace(old,'R = Path('+repr(str(out))+')'))
PY
"$SAM_PY" "$REPRO/busi_calibrated_driver.py" prepare
"$SAM_PY" "$REPRO/busi_calibrated/pipeline.py" run \
  > "$REPRO/busi_calibrated/logs/pipeline.log" 2>&1
```

此 pipeline 自带 GPU 等待：空闲显存至少18000 MiB、没有超过1 GiB的其他计算进程，优先GPU1。该检查不能阻止别人随后启动任务；曾发生启动后其他进程进入导致 OOM。失败后先核查日志和显存，再运行同一 `pipeline.py run` 断点恢复，不重复 prepare，也不杀其他人的任务。

关键输出：`calibration_frozen.json`、`route_reuse_audit.json`、`validation_results.json`、`models_frozen.json`、`test_candidate_pool_frozen.json`、`test_choices_frozen.json`、`results.json`、`test_per_target.json`、`completion_audit.json`、`COMPLETE`。

## 12. 复现完成的判断与常见混淆

1. 先核验 checkpoint、协议、自动名单和顺序。自动选图不是人工选5/8/21张相似图代替。
2. 如果重跑选图名单不一致，先检查1008缓存、train顺序、numpy/sklearn版本、归一化顺序、IDF dtype和重复文件处理；不要用test成绩决定接受哪套名单。
3. 256缓存和merged_manifest必须一一对应；support数量及顺序也要一致。不要将1008选图缓存当作256 TP缓存。
4. 所有桥接节点来自train，目标不能出现在自身桥中；只有已选参考图GT可参与原型/框/返回评价。
5. b0没有中间桥，但依旧需要参考图提示，不等于无提示SAM3直接预测。
6. test全部图像参与宏平均；低质量或空预测不能删掉。q_cycle并不是真实target Dice。
7. Oracle不是可部署预测成绩；候选数量变化时分别报告。Router只选候选，因此不能改变同一固定候选池的Oracle。
8. 原数据划分用于沿用实验，不宣称已排除全部同病例/视频近重复。现有图像文件没有可靠视频链源ID可用于患者级去重；这里的“伪视频”是算法构造的路径。
9. validation GT是额外监督；此前test已被用于多轮诊断，本轮复现不是从未见过的盲测。没有充分随机参考集传播重复，不能仅据这三个单次实验证明自动选图普遍优于随机。
10. 不要重新运行旧目录的prepare；一些入口用 `mkdir(exist_ok=False)` 或日志 `open('x')`。必须创建独立输出，保留旧mask、权重和结果。

本文对应历史结果已经完成。复现文档本次另外执行了哪些核验、哪些命令仅做实现检查，见下方的交付核验记录，避免把文档核查误解成已经重跑全部GPU实验。

## 13. 本次文档交付的实际核验记录

2026-09-14 已执行：

| 检查 | Kvasir | ISIC2018 | BUSI |
|---|---|---|---|
| 从历史1008原始特征重新拟合词典、IDF、贪心选图 | 8张名单及顺序完全一致 | 21张名单及顺序完全一致 | 5张名单及顺序完全一致 |
| 词典/直方图最大绝对误差 | 0 / 0 | 0 / 0 | 0 / 0 |
| 文件去重后可选候选数量 | 800 | 2075 | 516 |
| 256缓存重建路径检查 | 全部test的700条route ID完全一致 | 首张test的b0–b6共7条完全一致 | 首张test的b0–b6共7条完全一致 |
| 使用本文评分入口重算全部存档test mask | 7个Dice差值均为0 | 7个Dice差值均为0 | 7个Dice差值均为0 |
| 使用文中复制脚本命令重拟合原版Router | 0.8713744761396023 | 0.8663633850059499 | 0.5668083894947364 |

7个 Bash 代码块通过 `bash -n`；内嵌 Python 块通过语法检查；辅助脚本通过编译检查。本次未重新执行1008 GPU编码、整套SAM3传播或BUSI校准版完整GPU流水线，相关命令依据已完成实验的冻结脚本提供。不要把本节CPU/存档评分核验写成一次新的从RGB到mask全流程重复实验。

服务器核验材料均在 `$G`：

```text
source_hashes.json
verify_{kvasir,isic2018,busi}/selection_replay.json
command_checks.json
verify_router_kvasir_isic/{kvasir,isic2018}/results.json
verify_router_busi/busi/results.json
reproduce_automatic_selection.py
三数据集自动选图与TP路线_中文复现指南.md
```

本地文档：`F:\medsam3\新老环节对比\三数据集自动选图与TP路线_中文复现指南.md`。建议接手agent先执行第6节，确认自动选图的词典与名单完全复现，再根据需要执行第8节传播、第10节Router或第11节BUSI扩展版。
