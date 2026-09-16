# BUSI：分数校准 × 参考数量四组消融

日期：2026-09-14。目的：区分跨参考图分数校准与增加参考候选数量各自带来的收益。保持相同 Router 结构和超参数，不增加新的模型配置搜索。

## 实验分组

| 组别 | 每张目标图的参考排序 | 保留参考数 | 每图候选 |
|---|---|---:|---:|
| raw_top1 | 原始 TP(A,T) | 1 | 7 |
| centered_top1 | TP(A,T)减去参考A在517张train上的TP均值 | 1 | 7 |
| raw_top2 | 原始 TP(A,T) | 2 | 14 |
| centered_top2 | 减训练均值后的TP分数 | 2 | 14 |
| original_per_bridge（额外历史对照） | 每个b独立搜索后选择最优参考/路径 | 不固定一张参考用于全部b | 7 |

四个主要实验中，每张目标图先选定参考，然后这些参考分别生成全部b0–b6。raw_top1与历史original_per_bridge不是完全相同的选择规则，不可混用名称。

## 固定条件

- 原5张自动参考：benign (3)、benign (125)、benign (305)、malignant (195)、malignant (187)。没有重新选训练标注图。
- 原 train517 / validation64 / test66 划分。校准只使用train RGB特征，前景原型沿用5张已选参考GT。
- SAM3-base原权重，参考GT tight box，无文本，canvas256。参考内部的路径搜索保持原始TP瓶颈/均值评分、patch_mean KNN、beam32。
- 所有组均使用legacy的28项特征、Ridge(alpha=1)，不加peer或校准分数特征。标准化参数在每个训练折内估计。
- 沿用相同validation图像级5折；同图所有候选始终同折。每组在本组候选上分别拟合，超参数相同，权重不强行共用。
- 所有4组和历史对照都预先固定并报告，不根据test结果选择新的超参数。validation标注是训练1%标注预算之外的额外监督。
- 读取test GT前冻结全部模型和每组实际选择。test此前已被用于诊断，本轮是已有基准上的消融，不是新的盲测。

## 候选复用

原多参考实验已构建每张目标图5个参考各自的b0–b6最佳路径，沿用这些路径。比较route ID、参考、桥节点、参考框及mask SHA256后，才复用已有预测。

| 划分 | 五组候选并集 | 复用 | 需要新增传播 |
|---|---:|---:|---:|
| validation | 1400 | 1400 | 0 |
| test | 1351 | 1183 | 168 |

并集数不是单个Router的候选数；每个Router仍分别只看每图7或14个候选。所有test66张都参与评价，不按质量阈值丢图。

## 已完成的 validation 结果

OOF Dice 是该图未参与训练的Router选中mask的平均Dice。

| 组别 | 候选数/图 | OOF Dice | Oracle Dice | Oracle差距 |
|---|---:|---:|---:|---:|
| raw_top1 | 7 | 0.616268 | 0.675247 | 0.058978 |
| centered_top1 | 7 | 0.728772 | 0.754000 | 0.025228 |
| raw_top2 | 14 | 0.665683 | 0.783128 | 0.117445 |
| centered_top2 | 14 | 0.738958 | 0.823477 | 0.084518 |
| original_per_bridge | 7 | 0.617634 | 0.675247 | 0.057613 |

固定候选数时，校准分别提升OOF Dice：top1为0.112504，top2为0.073275。校准后top1→top2增加0.010186，但候选Oracle增加约0.069477，说明新增候选的潜力尚未完全转化成实际选择收益。

对应逐图配对bootstrap的95%区间：校准top1对原始top1为[0.044333,0.185253]；校准top2对原始top2为[0.011933,0.141118]；校准后top2对top1为[-0.023520,0.046310]。这些是探索性区间，未做多重比较校正，也不将OOF样本视为独立训练实验的重复。

## 已完成的 test 结果

test66张全部完成，新增168个候选成功；完整并集1351条预测无失败。以下各组均由相同设置、各自在validation上拟合的Router选择最终mask。

| 组别 | 候选数/图 | test Dice | test IoU | Oracle Dice | Oracle差距 |
|---|---:|---:|---:|---:|---:|
| 原始分数 top1 | 7 | 0.565990 | 0.479339 | 0.617141 | 0.051151 |
| 校准分数 top1 | 7 | 0.719992 | 0.635915 | 0.775461 | 0.055469 |
| 原始分数 top2 | 14 | 0.656084 | 0.564468 | 0.776347 | 0.120263 |
| 校准分数 top2 | 14 | **0.752198** | **0.668155** | **0.823649** | 0.071452 |
| 历史每b独立选参考对照 | 7 | 0.566808 | 0.480026 | 0.617141 | 0.050333 |

原始top1与历史对照在test恰好有相同候选，但validation候选并非完全相同，拟合权重和最终选择也略有差别，所以test Dice分别为0.565990与0.566808。这不是评分标准发生变化。

| 固定比较 | test Dice增量 | 逐图配对bootstrap 95%区间 |
|---|---:|---|
| top1：校准−原始 | +0.154001 | [0.076563, 0.236401] |
| top2：校准−原始 | +0.096114 | [0.021322, 0.173987] |
| 原始分数：top2−top1 | +0.090094 | [0.017658, 0.165548] |
| 校准分数：top2−top1 | +0.032206 | [-0.014657, 0.081100] |
| 校准top2−历史对照 | +0.185389 | [0.104277, 0.270905] |

这些区间使用固定预测的10000次图像级配对重采样，seed2026；没有额外训练种子重复，也未做多重比较校正。

### 当前结论

1. **校准有明确的本轮收益证据。** 相同7候选下提高15.40个百分点，相同14候选下提高9.61个百分点；不能把全部收益解释为增加候选数量。
2. **多参考候选也有潜力，但校准后的额外收益还不稳定。** 校准top2是本轮最高均值，比校准top1高3.22个百分点；该差值95%区间包含0，validation的差值区间也包含0。尚不能宣称top2在其他数据或重复实验中必然更优。
3. **不校准直接扩候选会增加选择困难。** 原始top2的Oracle为0.776347，已接近校准top1的0.775461，但实际Dice仅0.656084，低于校准top1的0.719992。Oracle与实际输出差距达到0.120263。
4. 本轮最高均值配置仍为校准top2 + legacy Router，0.752198，精确复现此前结果；它与Oracle仍差0.071452。校准top1保留7候选，0.719992，可作为候选传播开销更低的对照。候选数量7→14使传播任务数翻倍，本文未进行受控端到端测速。

### Router最终参考使用数量

| 参考图 | 原始top1 | 校准top1 | 原始top2 | 校准top2 |
|---|---:|---:|---:|---:|
| benign (3) | 0 | 23 | 9 | 29 |
| benign (125) | 0 | 9 | 0 | 7 |
| benign (305) | 0 | 18 | 5 | 13 |
| malignant (195) | 1 | 6 | 31 | 17 |
| malignant (187) | 65 | 10 | 21 | 0 |

这一表是Router最终选中mask的来源，每列总计66；不是选图算法最初选出的参考集合比例，也不是全部候选路径占比。

## 启动命令及服务器文件

实验目录：

```text
/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/busi_calibration_factorial_20260914
```

本轮已全部完成，控制进程历史PID为102305。日志、`completion_audit.json`和`COMPLETE`保存完成核验。

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export PYTHONPATH=/Data_8TB/lht/sam3:/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

# 已执行，不要在同一目录重复prepare。
/home/violet/anaconda3/envs/sam3/bin/python new_project/busi_calibration_factorial.py prepare

# 实际执行入口；只有确认原控制进程已退出后才用它断点恢复。
/home/violet/anaconda3/envs/sam3/bin/python \
  new_project/experiments/busi_calibration_factorial_20260914/pipeline.py run
```

GPU等待规则沿用前一轮：空闲至少18000 MiB且无超过1 GiB的其他计算进程，优先GPU1，其次GPU0。本轮GPU0空闲后开始运行。不会停止已有的其他任务。

| 文件 | 用途 |
|---|---|
| `predefined_policy.json` | 实验条件与比较项目 |
| `pool_membership_frozen.json` | val/test五组的逐图候选route ID，先于评价冻结 |
| `reuse_audit.json` | 候选复用与补跑数量 |
| `folds_frozen.json` | 沿用的validation分折 |
| `validation_results.json` | OOF、Oracle、参考占比、逐图结果及配对比较 |
| `models_frozen.json` | 五组各自的Ridge权重，test评价前冻结 |
| `status.json` | 当前阶段或错误 |
| `logs/pipeline.log` | 全流程日志 |
| `logs/propagation_test.log` | 新增168候选的传播进度 |
| `test_choices_frozen.json` | 读取test GT前冻结的实际选择 |
| `results.json`、`test_per_target.json` | 完成后的五组test指标、Oracle及逐图结果 |
| `report.md`、`completion_audit.json`、`COMPLETE` | 最终汇总及完整性核验 |

评价仍为256二值mask与nearest缩放GT的逐图Dice/IoU后宏平均。报告配对差值与参考图使用占比；计时仅汇总候选记录的历史seconds，不将不同GPU负载下的混合记录当作严格的端到端速度对比。
