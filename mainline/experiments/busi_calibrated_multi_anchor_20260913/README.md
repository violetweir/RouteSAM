# BUSI：TP 分数校准与多参考候选实验

日期：2026-09-13。

当前状态：实验已全部完成。validation 2240/2240、test 所需候选 1183/1183 均成功。验证集选定校准 top2 参考图（每图 14 个候选）+ legacy Ridge Router(alpha=1)。test Dice=0.752197576，IoU=0.668154514；原版 Router Dice=0.566808389。完整结果见 report.md 和 results.json。

## 1. 实验动机

原始 TP 不同参考图的相似度数值存在明显偏移。自动选择的 5 张参考图保持不变时，原版 test 的 b0 有 65/66 张选择 malignant (187)，全部 b0–b6 路径中有 455/462 条来自该参考图。

已有的 64 张 validation 逐参考图 b0 诊断结果：原始分数选图 Dice 为 0.494374；减去每个参考图在 517 张训练 RGB 上的平均分数后，Dice 为 0.700627；全部 5 个参考图 b0 的 Oracle 为 0.792824。这些是验证集诊断结果，不是本轮 test 成绩。

## 2. 固定条件

| 环节 | 设置 |
|---|---|
| 数据划分 | 原 train 517 / validation 64 / test 66 |
| 训练标注预算 | 自动选择的原 5 张，约占 train 的 0.967% |
| 参考图 | benign (3)、benign (125)、benign (305)、malignant (195)、malignant (187) |
| 模型 | SAM3-base 原始权重，不微调、不训练学生 |
| 提示 | 参考图 GT 的 tight box，无文本 |
| 路径 | SAM3-base patch_mean KNN，TP，beam=32，b0–b6 |
| 输入/评价 | 沿用旧版 256 流程，GT nearest resize，逐图 Dice 后平均 |
| 传播与返回 | 沿用旧版最高分对象前向传播；预测 mask 初始化反向返回 |
| Router 标注 | 使用额外 64 张 validation GT；不能理解为总共只用了 5 张标注 |

## 3. 本轮改动

### 3.1 校准跨参考图分数

对每个参考图 A，先计算：

```text
mean_A = mean(TP(A, x)), x 遍历原 517 张训练 RGB
calibrated_score(A, target) = TP(A, target) - mean_A
```

这里只读取未标注训练图的 RGB 特征。前景 prototype 沿用原 5 张参考图 GT，不读取其余训练图的隐藏 GT。

参考图内部的路径瓶颈分数、均值分数、KNN 搜索和 tie-break 保持原样。校准用于参考图之间的排序，不直接把中心化后的分数与原始边相似度混合计算路径瓶颈。

### 3.2 保留多个参考图

为每张目标图的每个参考图各生成 b0–b6 最佳路径，因此完整池有 35 个候选。按校准分数保留前 K 个参考图，比较：

| K | 每图候选数 |
|---:|---:|
| 1 | 7 |
| 2 | 14 |
| 3 | 21 |
| 5 | 35 |

另保留原始未校准的 7 候选作为对照。候选数量不同，Oracle 分开报告；Oracle 不参与 Router 选图或参数选择。

### 3.3 Router 与验证方式

预先固定 13 个方案：旧 7 候选 + legacy Ridge(alpha=1)；四个 K 各比较 legacy Ridge(alpha=1)、calibrated_peer Ridge(alpha=10/100)。

legacy 使用原 28 个特征。calibrated_peer 增加 8 个候选间 mask 一致性/面积特征，以及 5 个参考图校准特征：原始目标分数、中心化目标分数、减去参考图训练均值的路径瓶颈/均值、参考图校准排名。后者对特征和训练目标分别做每张图内部中心化。

沿用旧 BUSI 的图像级 5 折划分，整张图的所有候选必须在同一折。用 validation OOF 的选中 mask 平均 Dice 选择 K 和 Router 配置；同时做 outer 5-fold / inner 4-fold 的 nested CV，估计完整配置选择流程的表现。

最终在全部 validation 上拟合选定 Router，再冻结模型、K 和 test 候选池。test 的具体 route 选择也先冻结，之后才读取 test GT 计算结果。此前 test 已有诊断记录，本轮不能宣称它是从未看过的盲测集。

## 4. 已完成的核验

- validation 的 448 条旧路径、test 的 462 条旧路径，其 route ID 和原有全部字段逐条一致。
- 910 张旧候选 mask 的 SHA256 全部通过，直接复用，原实验不覆盖。
- validation 完整候选 2,240 条，复用 448 条，新增 1,792 条。
- test 完整路径已构建 2,310 条；真正新增传播数量由 validation 冻结后的 K 决定，并补齐旧对照和校准 top1 的 b0–b6。
- Router 使用旧预测构造的集成测试覆盖 7/14/21/35 候选、分组 CV、模型冻结、test 选择和评价，复现旧 Router validation OOF=0.6176338160、test Dice=0.5668083895。该检查数据是重复候选夹具，不能当作本轮改进结果。

## 5. 服务器位置与启动命令

实验目录：

```text
/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/busi_calibrated_multi_anchor_20260913
```

已完成的控制进程 PID：3868695。实际运行脚本是实验目录中冻结的 `pipeline.py`。以下命令供复现参考。

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PYTHONPATH=/Data_8TB/lht/sam3:/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/src \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
/home/violet/anaconda3/envs/sam3/bin/python \
new_project/experiments/busi_calibrated_multi_anchor_20260913/pipeline.py run
```

GPU 等待规则：每 30 秒检查一次；只有空闲显存至少 18,000 MiB 且该卡没有超过 1 GiB 的其他计算进程时才启动，优先 GPU1。后台控制进程仍在运行即可继续等待；不会杀掉现有训练。若推理失败，写入 `status.json` 并退出，保留已完成候选供核查后恢复。

主要文件（均相对上述实验目录）：

| 文件 | 用途 |
|---|---|
| `status.json` | 当前阶段 |
| `logs/pipeline.log` | CPU 路径、等待 GPU、流程进度 |
| `logs/propagation_validation.log` | validation 传播，实际启动后创建 |
| `logs/propagation_test.log` | test 传播，实际启动后创建 |
| `calibration_frozen.json` | 训练均值、每图参考排序与校准分数 |
| `route_reuse_audit.json` | 原路径和旧预测复用核验 |
| `router_integration_check.json` | Router 集成检查，非新实验成绩 |
| `b0_calibration_recheck.json` | 用已有 val b0 预测核查校准排序 |
| `validation_results.json` | 验证完成后输出各 K、OOF、nested CV、Oracle |
| `models_frozen.json` | validation 选定的模型、K、固定 b |
| `test_choices_frozen.json` | 读取 test GT 前冻结的逐图选择 |
| `results.json` / `report.md` | 全流程完成后的最终结果 |

完整流程已连接为：等待 GPU → validation 新候选传播 → 分组 CV 选择并冻结 → 等待 GPU → test 所需新候选传播 → 冻结选择 → test 评价与报告。
