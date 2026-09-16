# Kvasir、ISIC2018：分数校准 × 参考数量四组消融

日期：2026-09-14。将BUSI已完成的四组消融迁移到息肉和皮肤病灶数据，检验跨参考图分数校准是否具有通用性。

## 固定实验条件

| 数据集 | train | validation | test | 原有自动参考数量 |
|---|---:|---:|---:|---:|
| Kvasir-SEG | 800 | 100 | 100 | 8 |
| ISIC2018 | 2075 | 259 | 260 | 21 |

沿用原数据划分和自动参考名单，不重新选标注样本。SAM3-base原始权重、GT tight box、无文本、canvas256，patch_mean KNN、beam32和原始TP路径瓶颈/均值评分不变。

对于每个参考A，从本数据集全部train RGB的现有TP特征计算：

```text
mean_A = mean_train(TP(A,x))
centered(A,T) = TP(A,T) - mean_A
```

不使用其余训练图GT。参考图GT只限于已选的8/21张，用于原型、初始框和返回一致性。均值仅用于目标与参考之间的排序，不改变参考内部路径的原始TP/余弦评分尺度。

## 四组与额外对照

| 组别 | 排序分数 | 每张目标保留参考 | 每图候选 |
|---|---|---:|---:|
| raw_top1 | 原始TP(A,T) | 1 | 7 |
| centered_top1 | 减训练均值的TP(A,T) | 1 | 7 |
| raw_top2 | 原始TP(A,T) | 2 | 14 |
| centered_top2 | 减训练均值的TP(A,T) | 2 | 14 |
| original_per_bridge | 历史每个b独立搜索并选择参考/路径 | 按b变化 | 7 |

四组中先对每张目标选定参考，再让这些参考分别生成b0–b6。分数相同时按anchor ID升序破除并列。Router候选顺序为参考排名、桥长，预测分数并列取最早候选。

原始top1和历史每b独立选择不属于同一种规则，额外保留历史对照以便准确解释变化。

## 统一Router

所有组单独在各自候选池上拟合相同的legacy28 Ridge(alpha=1)，不增加peer/校准特征，不改超参数。沿用各数据集之前冻结的validation图像级5折，同图所有候选始终同折；标准化在每个训练折内计算。

完成validation预测后计算每组OOF和Oracle，再用全validation拟合五个模型并冻结。完成test候选后，先冻结全部实际选择，再读取test GT计算五组结果。不根据本轮test选择新的参数。

历史同结构Router对照应复现：

| 数据集 | 原版legacy validation OOF Dice | 原版legacy test Dice |
|---|---:|---:|
| Kvasir | 0.8513170057291387 | 0.8713744761396023 |
| ISIC2018 | 0.8623961438691973 | 0.8665732155247388 |

ISIC此前按validation选出的rank_peer_a100为test0.866363；本轮统一legacy结构，因此主要历史对照使用0.866573，不把不同Router结构混入四组比较。

validation标注是训练参考1%预算之外的额外监督。test此前已用于诊断，本轮是已有基准上的消融，不是未见过的盲测。

## 路径与预测复用

优先使用原始自动参考实验的mask，再读取已有P1/P2实验的稳定快照。复用需同时满足传播脚本SHA256、route ID、参考、桥节点、提示框、参考mask SHA256和前向mask SHA256一致。

Kvasir已有P2全参考路径，因此直接筛出四组所需路径。ISIC只构建原始top2和校准top2参考并集所需的路径，额外加入历史每b对照，无需为21张参考全部生成传播候选。

正在运行的P2质量记录不会作为不可变文件来做最终哈希断言；本轮保存读取到的独立快照及其哈希。执行每个split前重新检查新增的可复用mask，以减少与既有任务重复传播。

对于历史route ID，沿用其原有浮点字段，避免完整train+val+test矩阵与历史train+val矩阵末位浮点差异改变历史Router输入。重建中与历史重叠的路径须保持route ID一致。

## 执行顺序与状态

CPU分别准备两个数据集的路径。GPU流程串行执行：ISIC validation→Router冻结→ISIC test→评价，然后Kvasir validation→Router冻结→Kvasir test→评价。

启动时GPU0空闲；GPU1已有其他训练与Kvasir P2传播。本轮不停止其他任务。每个传播阶段检查至少18000 MiB空闲显存且无超过1 GiB的其他计算进程，再选择显卡。GPU1满足条件则优先GPU1，否则使用可用的GPU0。

当前控制进程历史PID为126614；CPU路径进程历史PID为ISIC126612、Kvasir126613。是否仍运行请看状态和日志，不以PID记录本身判断。

## 服务器位置与启动命令

```text
/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/cross_dataset_calibration_factorial_20260914
```

总入口为根目录的`pipeline.py`。每个数据集目录保存独立代码快照、协议、缓存、模型、mask和报告。

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export PYTHONPATH=/Data_8TB/lht/sam3:/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export F=new_project/experiments/cross_dataset_calibration_factorial_20260914/pipeline.py
export PY=/home/violet/anaconda3/envs/sam3/bin/python

# prepare已执行，同一输出位置不可重复prepare。
"$PY" "$F" prepare isic2018
"$PY" "$F" prepare kvasir

# CPU路径构建；完成标志存在时跳过。
"$PY" "$F" routes isic2018
"$PY" "$F" routes kvasir

# 仅在确认原控制进程已退出后，用此命令恢复；不会重做已完成传播。
"$PY" "$F" run-all
```

完整重新复现应复制入口并将ROOT改为新的输出目录，保留原实验结果。当前已提交上述流程，不要重复启动另一个控制器争用显卡。

主要文件相对于实验根目录：

| 文件 | 内容 |
|---|---|
| `status.json`、`pipeline.log` | 总阶段与控制日志 |
| `processes.json` | 已提交进程及命令 |
| `{dataset}/status.json` | 当前数据集阶段/错误 |
| `{dataset}/logs/routes.log` | CPU路径进度 |
| `{dataset}/logs/propagation_validation.log` | validation传播 |
| `{dataset}/logs/propagation_test.log` | test传播 |
| `{dataset}/route_audit.json` | 路径并集及历史重叠核验 |
| `{dataset}/{split}_reuse_audit.json` | 复用、缺失数量和来源快照 |
| `{dataset}/calibration_frozen.json` | 训练均值及每图参考排序 |
| `{dataset}/pool_membership_frozen.json` | 五组逐图候选成员 |
| `{dataset}/validation_results.json` | OOF、Oracle、配对比较 |
| `{dataset}/models_frozen.json` | 五组冻结模型 |
| `{dataset}/test_choices_frozen.json` | test GT读取前冻结的选择 |
| `{dataset}/results.json`、`test_per_target.json` | test汇总及逐图结果 |
| `{dataset}/report.md`、`completion_audit.json`、`COMPLETE` | 最终报告和验收 |

## 评价与解读

GT nearest resize到256，二值阈值>127，逐图Dice/IoU后对完整test宏平均，不删低质量图。Oracle为逐图候选最好Dice再平均，7候选和14候选分别报告。

报告校准在固定候选数下的收益、增加参考数量的收益、Oracle差距和参考使用分布。逐图配对bootstrap10000次、seed2026，区间为探索性结果，不做多重比较校正，不将跨折OOF视作独立训练重复。

候选历史seconds的汇总只作为传播工作量参考，不是受控端到端测速。7→14意味着候选传播任务数翻倍。

本文件是实验方案与运行说明。尚无新的跨数据集四组结果时，不应将历史Router或BUSI结果填入Kvasir/ISIC新配置成绩。

## 启动后的核验记录

两个数据集路径均已构建完成：

| 数据集 | validation五组候选并集 | test五组候选并集 |
|---|---:|---:|
| Kvasir | 1942 | 1862 |
| ISIC2018 | 5547 | 5437 |

这些是多个实验组共享后的候选并集，每个Router仍只看其指定的7或14个候选。

ISIC首次validation复用2326个，新增3221个，已在GPU0开始传播。Kvasir首次validation快照可复用975个，还缺967个；另一项P2实验继续生成候选，本轮正式执行Kvasir时会再次读取快照，实际补跑量可能减少。

新入口已完成原始对照的CPU集成核验：Kvasir validation OOF=0.8513170057291387、ISIC=0.8623961438691973，均与存档完全一致。记录位于每个数据集的`control_integration_check.json`；这不是四组新配置的结果，也没有为该检查读取test GT。

## 2026-09-15：ISIC完整validation结果

259张验证图、5547个并集候选已全部完成。以下是完整验证集按图像分组5折的OOF选中mask Dice，不是test，也不是此前113张子集的Oracle。

| 配置 | 每图候选 | Router OOF Dice | Oracle Dice | 差距 |
|---|---:|---:|---:|---:|
| 原始top1 | 7 | 0.861163 | 0.886672 | 0.025509 |
| 校准top1 | 7 | 0.876904 | 0.901635 | 0.024731 |
| 原始top2 | 14 | 0.868854 | 0.903707 | 0.034853 |
| 校准top2 | 14 | **0.880950** | **0.917948** | 0.036998 |
| 历史每b独立选择对照 | 7 | 0.862396 | 0.887467 | 0.025071 |

相同候选数时，校准top1相对原始top1提升0.015742，配对bootstrap95%区间[0.000417,0.031808]；校准top2相对原始top2提升0.012096，区间[0.000883,0.024890]。校准后top2相对top1仅增加0.004046，区间[-0.002022,0.011855]包含0。

完整验证集支持校准带来收益，但校准后的top2额外收益仍不稳定。校准top2相对历史对照提高1.8554个百分点；最终效果继续等待完整test。五组模型已冻结，test共5437候选，复用2314个，需要补跑3123个。Kvasir按原计划随后执行。

## 最终结果：两套数据均已完成

以下是最终完整test结果，取代上面的运行中状态；此前各节保留为实验过程记录。Kvasir覆盖全部100张test，ISIC覆盖全部260张test。两个数据集均通过历史legacy Router精确复现、输入不变、选择先于test GT读取等完整性核验；根目录及数据集目录均已生成`COMPLETE`。

### 完整test Dice

| 方法 | 每图候选 | Kvasir Dice | ISIC2018 Dice |
|---|---:|---:|---:|
| 原始分数 top1 | 7 | 0.870848 | 0.868228 |
| 校准分数 top1 | 7 | 0.860331 | 0.871717 |
| 原始分数 top2 | 14 | **0.891226** | 0.869276 |
| 校准分数 top2 | 14 | 0.862761 | **0.875864** |
| 历史每b独立选择 + legacy Router | 7 | 0.871374 | 0.866573 |

加粗仅标识本表test均值最高的配置，不能解释为独立验证后重新确定的统一基线。本轮四组是预先固定的消融，不根据test改动超参数或更换配置。

### 候选Oracle与实际选择差距

| 方法 | Kvasir Oracle | Kvasir Oracle−Dice | ISIC Oracle | ISIC Oracle−Dice |
|---|---:|---:|---:|---:|
| 原始top1 | 0.912017 | 0.041169 | 0.883608 | 0.015380 |
| 校准top1 | 0.906667 | 0.046337 | 0.890596 | 0.018879 |
| 原始top2 | 0.932897 | 0.041671 | 0.903881 | 0.034605 |
| 校准top2 | **0.937542** | **0.074782** | **0.911909** | **0.036045** |
| 历史每b独立对照 | 0.917615 | 0.046241 | 0.884316 | 0.017743 |

### 逐图配对差值

差值单位为Dice的0–1尺度；例如0.01表示1个百分点。区间为10000次图像配对bootstrap的探索性95%区间，未做多重比较校正。

| 比较 | Kvasir差值 [95%区间] | ISIC差值 [95%区间] |
|---|---|---|
| 校准top1−原始top1 | -0.010517 [-0.031494, 0.004702] | +0.003489 [-0.007685, 0.014676] |
| 校准top2−原始top2 | -0.028465 [-0.065831, 0.004531] | +0.006588 [-0.003581, 0.018563] |
| 原始top2−原始top1 | +0.020378 [-0.001077, 0.047059] | +0.001048 [-0.012279, 0.013093] |
| 校准top2−校准top1 | +0.002430 [-0.024604, 0.029778] | +0.004147 [-0.006658, 0.014842] |
| 校准top2−历史Router对照 | -0.008614 [-0.040066, 0.020608] | +0.009291 [-0.001039, 0.019631] |

所有上述test差值区间都包含0，不能将本轮均值变化写成稳定的跨重复实验提升或下降。

### 当前结论

1. **不能将训练均值校准直接宣布为通用增益。** BUSI已有明显收益；ISIC本轮校准top2均值最高，但相对历史Router仅提高0.9291个百分点，区间包含0；Kvasir本轮校准的实际Dice均值反而下降。
2. **Kvasir校准top2的问题主要表现在选择。** 14候选Oracle为0.937542，比原始top2的0.932897更高，但Router实际Dice为0.862761，低于原始top2的0.891226，留下0.074782差距。候选池确有好的mask，当前legacy打分未有效兑现潜力。与此同时，校准top1的Oracle本身下降，说明校准不是只影响Router，也会改变保留候选的质量。
3. **候选数量增加不保证实际性能增加。** ISIC原始top2把Oracle从0.883608抬到0.903881，实际Dice却只增加0.001048。评价必须同时报告候选上限与实际选择结果。
4. **Kvasir的0.891226不能只看test就直接升级为基线。** 原始top2的validation OOF为0.846724，低于历史Router的0.851317和校准top2的0.853773。它在test上的均值最高但验证集并未支持选它。它也只比此前自动参考固定b6的test0.887069高约0.4156个百分点。
5. 现阶段可以保留各数据集已有基线，把这轮作为跨数据集消融证据。后续若改Router，应在validation上研究候选分数的可比性、错误选择类型和可靠性，再冻结方案评价test；不能通过test Oracle直接构造实际选图器。

最终结果文件为服务器实验根目录的`results.json`，内含两个数据集；本地同步副本为`核验数据/Kvasir_ISIC2018_四组消融_完整结果.json`。逐图选择与mask分别位于数据集目录的`test_per_target.json`及各组`masks/`。
