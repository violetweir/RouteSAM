# Pseudo-Video SAM3 — X3 + B7（中文说明）

基于**冻结 SAM3** 的类别无关伪视频分割：单图学生审计器 + 几何路线选择。

[English README](README.md) · [最新跨数据集实验](docs/cross_dataset_1pct.md) · [V1–V7 逐版本详解](docs/kvasir_versions.md) · [仓库结构说明](docs/REPOSITORY_LAYOUT.md) · [完整复现指南](docs/reproduction_guide.md) · [方法（中文）](docs/method_cn.md)

---

## 这是什么

**主线是 Kvasir-SEG、ISIC2018、BUSI 上的跨数据集 1% 锚点实验**（TN3K 作为第四
个数据集正在补）：**冻结 SAM3**，从 1% 标注预算的参考图构造**伪视频传播路线**，
把全部建模精力放在**路线/候选的选择**上。

| 线路 | 数据集 | 标注预算 | 定位 |
|---|---|---|---|
| **1% 锚点跨数据集实验** | Kvasir-SEG、ISIC2018、BUSI（+TN3K） | 1% 参考图，SAM3 冻结 | **主线** |
| ↳ Kvasir **V1–V4** | 仅 Kvasir-SEG | 8 张固定锚点，Router + S2/S3 学生 | 路线族与学生流程的演进 |
| ↳ Kvasir **V5–V7** | Kvasir-SEG | 自动覆盖选图，单 TP + Router | V7（校准 top-2）后来扩到跨数据集 |
| 伪视频 `S27 X3 + B7` | CVC-ClinicDB + Kvasir-SEG 合并 | 16 张固定锚点 | **历史线**，保留可复现 |

---

## 主线：跨数据集 1% 锚点实验

冻结 SAM3、同一套流程、同一份代码跑三个数据集：覆盖选图 → 免标注分数校准 →
宽度 × 深度候选 → ridge 验证器。完整说明见
**[docs/cross_dataset_1pct.md](docs/cross_dataset_1pct.md)**。

| 方法 | 每图候选 | Kvasir (100) | ISIC2018 (260) | BUSI (66) |
|---|---:|---:|---:|---:|
| 原始分数 top-1 | 7 | 0.870848 | 0.868228 | 0.565990 |
| 校准分数 top-1 | 7 | 0.860331 | 0.871717 | **0.719992** |
| 原始分数 top-2 | 14 | **0.891226** | 0.869276 | 0.656084 |
| 校准分数 top-2 | 14 | 0.862761 | **0.875864** | **0.752198** |
| 历史逐桥选择 + legacy Router | 7 | 0.871374 | 0.866573 | 0.566808 |

**结论。** 样板条件分数（TP）在参考图之间不可比。每个参考图训练均值 `μ_A` 的离散度
是一个**免标注**的诊断量，能预测分数校准的收益：

| 数据集 | 偏置比 | 校准增益（test） | 判定 |
|---|---:|---:|---|
| BUSI | 5.42 | +0.154（top-1）、+0.185（top-2） | 显著 |
| ISIC2018 | 1.10 | +0.004 / +0.007（val 为 +0.016 / +0.012，显著） | test 不显著 |
| Kvasir | 0.68 | −0.011 / −0.028 | 无效，**阴性对照** |

宽度 `k` 与深度 `m` 是真实杠杆，并且**与校准互相替代**：Kvasir 上 `k=1→2` 的天花板增益
在 `b0` 为 +0.069，全深度只剩 +0.015。只标 1 张图的全深度天花板已达 0.9142（test），
标到 8 张只再涨 +0.040。BUSI 上校准 top-2 + Router 达 **0.752198**，而同样 5 张标注训练
的 SynFoC 学生为 **0.653120**。

---

## Kvasir-SEG：V1 → V7 版本阶梯

Kvasir-SEG 是主线下最大的一支，前后重建过七次：**V1–V4** 沿用原 8 张固定锚点，演进路线族
与学生流程；**V5–V7** 改成自动覆盖选图，其中 **V7**（校准 top-2）后来扩展到了
ISIC2018 / BUSI / TN3K。

V1–V4 的共同形态就是 `S27 X3 + B7` 的 Kvasir 单数据集版本：8 张固定锚点、冻结 SAM3 的
`b0–b6` 路线（单 TP 版本每图 7 个候选）、独立 Router + 质量门限、S2/S3 学生、委员会审核、
X3、B7 路线选择。

| 版本 | 主线路 | 主要变化 |
|---|---|---|
| **V1** | 原 8 张参考图 → DINOv3 KNN → SAM3 **TP + PC** → S2/S3 → 委员会 → X3 → B7 | 最早的完整流程；后面也做了 SAM3 LoRA |
| **V2** | 原 8 张 → **SAM3-base 特征** KNN → TP + PC → S2/S3 → X3 → B7 | 用 SAM3 特征替换 DINOv3；包含 Round1、Round2A 等后续实验 |
| **V3** | 原 8 张 → **单 TP** `b0–b6` + 独立 Router → 448 张伪标签 → S2/S3 → 扩展池、X3、B7 | 去掉 PC，确定单 TP 基线 |
| **V4** | 原 8 张 → TP → **先筛合格候选再 Router** → 580 → 单学生软标签 → 全量重筛 → 新学生、B7 | 含 580 张旧 S2/S3 对照、单学生 hard/soft、620 张重筛池，以及后续新的 Round2 SAM3 微调 |
| **V5** | **自动选 8 张参考图** → SAM3-base KNN → 单 TP `b0–b6` → Router | 改变参考图来源 |
| **V6** | 自动 8 张 → **原始 TP 分数 top-2** 参考图 → 各走 `b0–b6` → Router | 每图候选 7 → 14 |
| **V7** | 自动 8 张 → **校准 TP 分数 top-2** 参考图 → 各走 `b0–b6` → Router | 免标注分数校准 |

各版本 test Dice（Kvasir，100 张）：

| 版本 | 头条结果 | Oracle | 支撑数字 |
|---|---|---:|---|
| V1 | X3-best + B7 **0.897146**（canvas 256）/ **0.899369**（原始 512） | 0.918109 / 0.919805 | Router 0.891288 / 0.894648；X3 单图 0.8633（512）；池 470 / 491；委员会 105-102 / 93-95-113；X3 池 677 / 679 |
| V2 | X3-best + B7 **0.895432**（base）/ **0.906380**（e33 LoRA 教师） | 0.921471 / 0.920206 | Router 0.874083（TP+PC 联合）或 0.885433（TP-only）；X3 单图 0.853920；首批 410 张，委员会 A 94 / B 128 / C 160，X3 池 632 |
| V3 | X3-best + B7 **0.892639** | 0.907426 | Router 0.885433；X3 单图 0.867161；448 张，委员会 A 94 / B 113，X3 池 655 |
| V4 | 620 重筛 final **0.869478**（best 0.858330）；Round2 SAM3 LoRA 50 轮 best **0.901950** | – | 580 池 S2/S3：S2 0.851530/0.851675、S3 0.841958/0.843769；单学生 hard 0.848592/0.854011、soft 0.851243/0.859530 |
| V5 | raw top-1 **0.870848** | 0.912017 | 独立 Router 0.871374；val OOF 0.848638 |
| V6 | raw top-2 **0.891226** | 0.932897 | val OOF 0.846724 |
| V7 | 校准 top-2 **0.862761** | **0.937542** | 校准 top-1 0.860331（val 0.852949）；val OOF 0.853773 |

**怎么读这条阶梯**

- **V1 → V2** 是路线特征替换：把 DINOv3 换成 SAM3 自己的编码器特征后，最好的固定路线从
  0.854627 升到 0.8874（`target pooling patch_mean` b5 @1008），B7 从 0.895432 升到 0.899369。
- **V2 → V3** 是候选集合变化：去掉 patch correspondence、每图只保留一个 TP 族，Router
  （0.874083 → 0.885433）和 X3（0.854101 → 0.867161）都变强，但 B7 略降
  （0.895432 → 0.892639），Oracle 与 B7 的差距从 0.026039 收窄到 0.014787。
- **V3 → V4** 是伪标签池变化：先筛合格候选再 Router，把首批从 448 扩到 580；后续用
  单学生等权训练替换多学生委员会，并对全部 792 张重筛得到 620 张池。随后用 LoRA 在学生
  审核池上做了 Round2 SAM3 微调。
- **V5 → V6 → V7** 改的是参考图来源与排序：自动选图 + 原始 top-1 得 0.870848；保留原始
  top-2 同时抬高上限（0.912017 → 0.932897）和实际成绩（0.891226）；校准把实际成绩压到
  0.862761，但上限继续升到 0.937542 —— 在 Kvasir 上瓶颈是**选择**而不是候选质量。

#### V4 内部的 Round2 SAM3 LoRA 微调

在学生审核池上做过全模块 LoRA（rank 16、alpha 32、dropout 0.1）。`20 epoch` 是最早的
LoRA 教师适配；Round2 用 10 epoch，另加一个 50 epoch 对照。

| 训练池（伪标签 + GT） | epoch | best epoch | Val Dice | Test Dice（`colon polyp`） | Test Dice（空文本） |
|---|---:|---:|---:|---:|---:|
| A0 · 428 + 8 | **50**（无裁剪） | 19 | 0.893717 | 0.901950 | 0.890174 |
| A0 · 428 + 8 | 50（final） | 50 | 0.873586 | 0.894245 | 0.884782 |
| A0 · 428 + 8 | 10 | 6 | 0.880909 | 0.901755 | 0.885264 |
| A1 · 596 + 8 | 10 | 4 | 0.887257 | 0.903294 | 0.870346 |
| A2 · 503 + 8 | 10 | 1 | 0.888680 | **0.912932** | 0.845776 |

10 epoch 双种子（2026 / 2027）：A0 0.901755 / 0.892288，A1 0.903294 / 0.886602，
A2 0.912932 / 0.884289 —— 排序在两个种子之间**反转**，没有稳定最优池，Round2 也没有超过
V1–V3 的 B7 结果。50 epoch 对照在第 19 轮见顶后回落；它同时关掉梯度裁剪、改了 cosine
周期，不是干净的时长消融。

#### V2–V4 使用的 LoRA 教师 checkpoint（20 epoch）

教师本身用同一套全模块 LoRA 配方训练 20 epoch（`configs/kvasir_1pct_lora.yaml`、
`configs/kvasir_1pct_plus_pseudo491_lora.yaml`）：

| checkpoint | LoRA 训练数据 |
|---|---|
| `lora_1pct_e20` | 8 张 GT 锚点（1%） |
| `lora_p491_e20` | 8 GT + **491 张伪标签**（第一轮 Router 扩展池） |

逐桥长 forward-only test Dice（100 targets，canvas 256）：

| checkpoint | 路线模式 | direct | b1 | b2 | b3 | b4 | b5 | b6 | 均值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora_1pct_e20` | target pooling | 0.771193 | 0.797666 | 0.861002 | 0.859552 | **0.886556** | 0.874425 | 0.880978 | 0.847339 |
| `lora_1pct_e20` | patch correspondence | 0.765991 | 0.800123 | 0.844173 | 0.852210 | **0.880581** | 0.869514 | 0.864183 | 0.839539 |
| `lora_p491_e20` | target pooling | 0.790054 | 0.824923 | 0.870163 | 0.877370 | **0.887164** | 0.890885 | 0.878756 | 0.859902 |
| `lora_p491_e20` | patch correspondence | 0.799188 | 0.830049 | 0.865747 | 0.884548 | 0.883273 | 0.889516 | **0.911171** | 0.866213 |

- 加入 491 张伪标签后，两种路线模式**每一档桥长都提升**（+0.009 ~ +0.047）；
  `patch correspondence b6 = 0.911171` 是 Kvasir 线里最好的固定单路线。
- 两个 LoRA checkpoint 在每一档上都优于冻结 base 和全量微调 `ft_1pct`。配上
  SAM3-encoder kNN @1008 特征后 `lora_p491_e20` 到 **0.9154**（target pooling cond，b1/b4）；
  @256 下到 **0.9082**（patch correspondence cond，b4）。
- 下游在 `lora_p491_e20` 路线上重跑：固定 b6 **0.8978**、B7 几何（三学生均值）
  **0.9014**、校准线性 **0.9035**、候选 Oracle 0.9357；pairwise ranker validation OOF
  **0.880443**；冻结迁移 ClinicDB test **0.857032**。
- 口径提醒：这些 B7 数字里的 S2/S3/X3 学生审计器是用 **base-SAM3** 伪标签训练的，
  没有在 LoRA 教师上重新训练。

#### V4 → V7 改了什么：自动选图与可校准分数

| V1–V4 | V5–V7 |
|---|---|
| 人工固定 8 张锚点 | **覆盖贪心自动选图**，只看 train RGB（`K = round(0.01·N_train)`） |
| 版本差异在路线族与学生配方 | 同一套单 TP `b0–b6` + Router 流程，差异在**参考图来源与排序** |
| 锚点分数直接比较（V5、V6） | V7 增加**免标注校准** `centered(A,T) = TP(A,T) − μ_A` |
| 每图一个参考图 | V6/V7 保留 **top-`k` 锚点 × `b0–b6`**，每图候选 7 → 14 |
| 人工 `q_multi`/`q_return` 阈值 | 显式 `Realized = Oracle − Gap` 分解；同一批四组消融随后跑到 **ISIC2018、BUSI**（TN3K 进行中） |

在 Kvasir 上这是校准的**阴性对照**（偏置比 0.68）：V7 校准 top-2 的候选池 Oracle 最高
（0.937542），但 Router 只兑现 0.862761，瓶颈从候选转移到了选择。它在 Kvasir 上真正确立的
是宽度杠杆（全深度天花板 vs `k`）和预算曲线——只标 1 张图的 test 天花板已达 0.9142。

Kvasir-SEG 的全部实验清单见 **[docs/kvasir_program.md](docs/kvasir_program.md)**。

## 历史线：S27 X3 + B7

> 这是项目最初的主链路，**不是当前主线**。它使用另一套协议
> （CVC-ClinicDB + Kvasir-SEG 合并、16 张锚点），保留是因为它端到端完全可复现。
> 完整说明见 [`docs/s27_x3_b7_line.md`](docs/s27_x3_b7_line.md)。

从 16 张固定的 CVC + Kvasir 训练锚点出发，用冻结的 SAM3 构造伪视频路线，把高置信伪标签
蒸馏成**单图学生审计器**，用学生审计扩展伪标签池，最后用学生在冻结路线之间做**路线选择**
（B7）。validation / test 的 mask 不参与路线构造、伪标签筛选或学生训练。

```text
16 张固定训练集 GT 参考图
  -> T21 冻结 SAM3 伪视频路线（direct / one_bridge / two_bridges）
  -> 568 张高置信伪标签
  -> T24 委员会学生（S2 val-best、S2 final、S3 final）
  -> 审计剩余训练图，划分 Tier A / B / C
  -> S27 X3 学生 = 原 568 + Tier A + Tier B（共 876 张伪标签）
  -> B7 学生辅助路线选择（在冻结 SAM3 路线上）
```

协议：CVC-ClinicDB 与 Kvasir-SEG 合并划分，**仅使用 16 张训练锚点**。

| 方法 | Test Dice | CVC Dice | Kvasir Dice |
|---|---:|---:|---:|
| 冻结 SAM3 多路线基线 | 0.8715 | – | – |
| S27 X3 单图学生 | 0.866738 | 0.886304 | 0.854802 |
| S27 X3 + 验证集选择线性选择器 | 0.885637 | 0.905077 | 0.873778 |
| **S27 X3 + B7 几何选择器** | **0.895835** | 0.904713 | 0.890420 |
| 三条冻结 SAM3 路线的 Oracle | 0.907835 | – | – |

B7 打分公式：

```text
score = (max(q_return, 1e-6) * max(q_multi, 1e-6)^2 * max(q_model, 1e-6)^2) ^ 0.2
```

- `q_return`：传播回参考图后的往返一致性
- `q_multi`：三条路线族之间的一致性
- `q_model`：学生对候选 mask 的置信度

## Kvasir 路线引擎：传播质量 router

更早的 Kvasir 1% 实验（同样冻结 SAM3，`train=800 / validation=100 / test=100`，
**8 张固定锚点**）用绝对分数替换了候选相对加权规则。它早于上面的跨数据集实验，
任务定义、router 与评价口径不同，保留作为过程记录。

| 选择器 | 候选范围 | Test Dice | Oracle | Oracle 差距 |
|---|---|---:|---:|---:|
| 固定最优路线（target pooling `b6`） | – | 0.854627 | – | – |
| v2 绝对分数 ridge router | target pooling `b0–b7` | 0.854437 | 0.898969 | 0.044531 |
| 传播质量 router | target pooling `b0–b7` | 0.872938 | 0.898969 | 0.026030 |
| 传播质量 router | target + patch `b0–b6` | 0.873626 | 0.903846 | 0.030220 |
| **传播质量 router** | **target + patch `b3–b6`** | **0.877299** | 0.896918 | 0.019619 |

在同一批 8 张锚点上微调 SAM3 后，同一 router 提升到 **0.894648**
（`ft_1pct`，`b3–b6` target + patch），Oracle 为 0.919805。

## 仓库结构

```text
configs/        运行配置（示例配置 + 各实验配置）
docs/           方法与协议文档（docs/v2/ 为早期系列）
docs/cross_dataset_1pct.md   最新 Kvasir / ISIC2018 / BUSI / TN3K 1% 实验
envs/           conda 环境示例（sam3、student）
protocols/      固定的路径无关划分与 16 锚点协议
scripts/        复现脚本与实验驱动（主链路：run_pipeline.py）
src/pvseg/      小型公共工具（io / metrics / protocol）
tests/          协议、泄漏与 checkpoint 不变量测试
third_party/    内置的 SC-SAM 与 SynFoC-T20 学生基线
medsam3/        内置的 SAM3 + LoRA 训练工具包
paper/          面向论文的实验包（表格、图、来源映射）
reports/        各实验的复现报告（Markdown）
mainline/       跨数据集 1% 研究：覆盖选图、分数校准、TP 路线（代码 + 报告）
results/        从 work/ 中提炼的逐次运行汇总指标（JSON/CSV/Markdown）
```

新目录与原服务器工作目录的对应关系见
[docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md)。

## 快速开始

```bash
cp configs/reproduction.example.toml configs/reproduction.toml
# 编辑 configs/reproduction.toml 中的路径

python scripts/run_method_ladder.py --config configs/reproduction.toml --dry-run
python scripts/run_pipeline.py     --config configs/reproduction.toml --dry-run
python scripts/run_pipeline.py     --config configs/reproduction.toml
```

可从任意阶段续跑：

```bash
python scripts/run_pipeline.py \
  --config configs/reproduction.toml \
  --from-stage s27_x3_train \
  --to-stage t25_b7_test
```

### 数据组织

```text
data/
  train/metadata.jsonl
  validation/metadata.jsonl
  test/metadata.jsonl
```

每行格式：

```json
{
  "file_name": "/abs/path/to/image.png",
  "mask_file_name": "/abs/path/to/mask.png",
  "merged_id": "CVC-ClinicDB::156",
  "source_dataset": "CVC-ClinicDB"
}
```

期望数量：`train=1290`、`validation=161`、`test=161`。固定的参考图 ID 见
`protocols/reproduction_v1/support_ids.txt`。

## 外部依赖

SAM3、数据集与模型权重**不在本仓库内**。SC-SAM 与 SynFoC 学生基线已内置在
`third_party/` 下，通过 `SC_SAM_ROOT` / `synfoc_root` 加载。

SAM3 相关命令与学生相关命令需要不同的环境，原服务器上为：

```text
SAM3:    /home/violet/anaconda3/envs/sam3/bin/python
Student: /home/violet/anaconda3/envs/mkunet_mamba/bin/python
```

## 收录与剔除

**收录**：源代码、各阶段脚本、配置、协议、方法与协议文档、实验报告、面向论文的
表格、跨数据集诊断图（7 张）、Kvasir 8 张锚点图像与 mask，以及 `results/` 下提炼后的
逐次运行汇总指标。

**剔除**（体积大或与本机绑定）：数据集、SAM3 / 学生权重（`*.pt`、`*.pth`）、
逐次运行的 mask 与特征缓存（少量图目录之外的 `*.png`、`*.npz`、`*.npy`）、
数 GB 的训练日志，以及原有 7 GB 的 `.git` 历史。`results/` 只保留小型汇总文件，
因此报告中的数字仍可追溯，但无需附带原始数据。

## 协议要点

- **跨数据集 1% 实验**（Kvasir-SEG / ISIC2018 / BUSI）是当前主线；`S27 X3 + B7` 是项目最初的主链路，保留且完全可复现，但不再是门面。
- 跨数据集实验中的「1%」**仅指参考图标注预算**；Router 使用 validation GT 拟合，
  且早期 test 已被多轮诊断，因此这些 test 数字不是盲测。
- 校准**不是普遍增益**：BUSI（偏置比 5.42）显著，ISIC2018（1.10）test 不显著，
  Kvasir（0.68）无效。不要把 BUSI 的增益当作通用结论。
- TN3K 目前是 **dry-run**（validation 23 / test 25 子集），完整的 576/614 仍在运行中，
  其数字为初步结果。
- 引用任何 S27/X3/B7 数字前先看 `docs/s27_x3_b7_line.md`；`S27 X0/X1/X3` 使用后期统一的 S27 学生训练器，并非旧 T24 监督损失实现的逐位复现。
- S27 训练器对 GT 使用逐样本前景 Dice；T24 使用两类别 batch Dice。
  公开结果来自被保留的 S27 实现。
- B7 作为固定的敏感性/主链路选择器汇报；验证集选择的线性选择器单独汇报。
- **禁止**使用 validation / test 的 mask 去改变参考锚点、伪标签、路线拓扑、
  训练数据或 checkpoint。

## 引用与许可

引用信息见 [CITATION.cff](CITATION.cff)；许可为 BSD-3-Clause，见 [LICENSE](LICENSE)。
内置第三方组件保留其自身许可，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
