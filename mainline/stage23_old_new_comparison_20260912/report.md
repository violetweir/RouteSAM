# 环节二＋环节三：首批伪标签筛选与学生训练——新老版本对比

更新日期：2026-09-12。数据集：Kvasir。本文接续《环节一.md》，联合比较“训练池如何筛选、监督标签如何构造、学生如何训练”。截至本文更新，所列训练和best/final测试均已完成。

**主要结论：新版单学生简化了训练流程，但目前按validation-best对应Test比较，尚未超过旧448张TP-only S3。新版soft用448张池的best好于用580张池，final则相反，不能宣称缩小池带来了稳定收益。**

## 1. 比较范围与版本定义

### 1.1 为什么把环节二和三合起来看

环节二决定哪些图像与哪些mask进入训练，环节三决定如何学习这些标签。扩大池后继续使用旧训练配方的结果，只能说明该组合的效果，不能代表“新池＋新软目标＋新采样”整套配方。

同时保留中间对照，才能区分池变化和训练变化：

```text
环节一：SAM3-base生成TP b0–b6候选
             ↓
环节二：从792张无标注训练图筛选首批伪标签
             ↓
环节三：生成监督标签、训练学生、按Val选best
             ↓
学生单图直接预测test，报告best/final
```

本文到学生单图测试为止，不纳入委员会扩展池、620张重筛学生、X3、B7、SAM3微调或自动anchor选择。学生test没有参考图、文本提示、Router或B7选mask。

### 1.2 五个版本与主对照

|编号|版本|候选与筛选|伪标签数|训练配方|用途|
|---|---|---|---:|---|---|
|H|最早SAM3特征双路线|TP＋PC，先Router再检查|410|旧S2/S3|历史参考，包含环节一变化|
|A|旧TP-only|TP，先Router再检查|448|旧S2/S3|本环节主基线|
|B|只改变筛选顺序|TP，先返回筛选再Router|580|旧S2/S3|观察扩大池的影响|
|C|新版单学生|同B|580|等权epoch训练；hard/soft对照|新配方主实验|
|D|新版配方回到原池|直接取A的原448张|448|等权epoch训练；soft|观察池大小的影响|

所有版本另加原8张GT，表中伪标签数不含GT。**410→448包含路线变化；448→580才是TP候选固定后的筛选顺序变化。**

## 2. 环节二：筛选规则具体差别

### 2.1 两个固定质量条件

- 图像整体一致性 `q_multi ≥ 0.90`：同一目标七张TP候选，两两计算Dice，对21对取平均。
- 候选返回一致性 `q_return ≥ 0.95`：沿对应路径返回anchor，与anchor人工GT比较；质量文件字段为`q_cycle`。

q_multi在这里是图像级分数，不是后续B7中单候选与其他候选的平均一致性。返回分数使用已标注anchor GT，不使用目标无标注训练图GT。

|步骤|A：旧448张|B/C：新版580张|
|---|---|---|
|1|Router对七个候选打分，选最高分|检查图像整体q_multi≥0.90|
|2|检查整体q_multi≥0.90|保留返回q_cycle≥0.95的候选|
|3|检查选中候选q_cycle≥0.95|Router只在合格候选中选最高分|
|4|两项通过才入池，否则排除整图|合格候选非空才入池|

使用相同TP候选、冻结Router和阈值。没有重新生成候选、放宽阈值或重新划分数据。

### 2.2 数量变化

|指标|A：旧规则|B/C：新规则|变化|
|---|---:|---:|---:|
|待筛训练图|792|792|0|
|接纳伪标签|448|580|+132|
|接纳率|56.57%|73.23%|+16.67个百分点|
|暂不接纳|344|212|-132|
|加8GT后的训练图数|456|588|+132|

原规则的排除情况：仅整体一致性失败76张，仅返回一致性失败160张，两项失败108张。160张中有132张存在其他返回合格的候选，因此新规则能接纳它们；剩余28张仍没有合格返回候选。

原448张的选中路线、mask及该轮旧训练图像权重逐项保持不变。新增132张并不代表最终扩展学生额外增加132张；扩展池属于后续环节，不能直接相加。

### 2.3 质量与覆盖率不能混为一谈

原validation五折、按目标图分组的折外Router诊断如下。这里评价的是接纳子集，不是完整val/test成绩：

|接纳子集|图数|平均Dice|Dice<0.5图数|
|---|---:|---:|---:|
|原规则|51|0.942856|0|
|新规则|71|0.913386|3|
|其中新增|20|0.838237|3|

新规则覆盖率更高，但接纳子集平均质量下降约2.95个百分点。新增20张原Router首选均值0.843000，改选返回合格候选后0.838237，未显示目标mask平均质量提高。

上述51张来自折外Router。历史完整validation拟合Router后筛出52张、子集均值0.944029，是另一统计口径，不能混用。

原580张主mask曾在冻结后进行训练GT事后审计，平均Dice=0.910379；它不是学生test分数，也不能与没有同口径审计数据的448张直接比较质量。训练GT未用于这些池的准入选择。

## 3. 环节三：标签、采样与训练策略

### 3.1 配方总览

|项目|A/B：旧S2/S3|C/D：新版单学生|
|---|---|---|
|学生结构|两次独立训练的U-Net，S2和S3|一个U-Net；hard是配方对照，不是额外委员会成员|
|S2／hard监督|Router选中的硬mask|相同主mask作为hard对照|
|S3／soft监督|全部七张TP候选平均|75%主mask＋25%返回合格TP候选平均|
|GT采样|每批固定6张GT，反复抽取8张GT|GT与伪标签统一打乱，每轮每图一次|
|伪标签采样|每批固定6张|无固定配额；多数batch没有GT|
|图像质量权重|q_multi归一化并截断到[0.2,1]|全部1|
|像素权重|S3按候选方差加权|全部1|
|伪标签总体系数|0.5|取消|
|伪标签权重ramp|前2000iter从0增至1|取消|
|损失|GT/S2为CE＋Dice；S3为加权BCE＋软Dice，并叠加旧权重|每图BCE＋Soft Dice，最后对batch平均|
|训练长度|40,000iter|816epoch|
|Batch size|12|12|
|优化器|SGD，lr=.01，momentum=.9，weight decay=1e-4|相同基础优化器设置|
|学习率衰减|随迭代线性下降|随epoch线性下降，轮内固定|
|验证频率|每200iter|每4epoch|
|模型选择|完整validation均值Dice最高|相同原则|

“学生”为代码中的`SamUnet`，实际包含U-Net；这里没有训练SAM3编码器。旧S3已经是软标签训练，因此本次变化不能简单称为“硬标签换软标签”。

### 3.2 旧S3监督

```text
P_all = (M0 + M1 + … + M6) / 7
W_pixel = max(0.1, exp(-4 × pixel_variance))
```

旧S3平均全部七个候选，包含返回一致性不合格的候选；再使用像素权重、图像权重、0.5伪标签系数和ramp。B版本虽然改了入池顺序，S3共识仍平均全部七张。

TP-only主线已修复uint16软标签/权重读取，按65535归一化。最早双路线历史代码曾存在16位图读取问题，因此H与TP-only版本还包含实现修复差异，不能视作单纯的候选数消融。

### 3.3 新版soft监督

```text
C = {返回一致性 ≥ 0.95 的TP候选}
M = 冻结Router在C中选中的主mask
P = mean(Mk，Mk属于C)
Y = 0.75 × M + 0.25 × P
```

0.75/0.25是监督目标混合比例，不是损失权重。只有一个合格候选时Y=M；Y按0.5阈值化后仍等于M。因此该方法软化监督，不直接修正主mask的二值形状。0.75/0.25是固定实验配方，未证明为最优比例。

统一损失：每张图计算前景BCE和Soft Dice，再对batch平均；Soft Dice平滑常数1。不再按照GT/伪标签分别求均值后加权。

### 3.4 448张新版与580张新版的控制变量

D直接复用C中对应448张图像和软标签，逐项核对原448张主mask；使用相同保存的初始模型权重。共有图像保留其580张清单中的索引，用于逐图增强随机种子，避免删除图像导致共有样本增强改变。

|项目|C：580张新配方|D：448张新soft|
|---|---:|---:|
|训练图总数|588|456|
|每epoch步数|49|38|
|epoch|816|816|
|总iter|39,984|31,008|
|验证间隔iter|196|152|
|初始化seed|2026|2026|

共有标签、初始化、逐图增强保持，但shuffle后的batch组合和总步数不同。这是固定epoch的池大小配方对照，不是固定计算量实验。D没有另外训练hard对照。

## 4. 测试与指标口径

|项目|本次TP-only各组共同设置|
|---|---|
|划分|原800/100/100，原8GT与792无标注身份|
|test覆盖|完整100张，每个checkpoint都计算，不按训练入池阈值筛test|
|输入|RGB单图，256×256；ImageNet均值方差归一化|
|图像/标签缩放|对应存档入口的NEAREST处理|
|预测|学生前景概率≥0.5，直接输出二值mask|
|Dice/IoU|每图计算，再对100张宏平均|
|空GT|该test的病灶GT均非空，空预测按0分计|
|best|每个训练run内部按validation选择；不按test选择epoch|
|final|预先固定训练终点，单列诊断|
|文本、参考图、Router、B7|学生test均不使用|

旧S2/S3的validation列来自训练入口，test由原导出入口计算；新学生会重新加载权重复现validation后再测test。本文没有把旧validation重新包装为统一入口重算的数值。H的test来自历史训练评估入口，未在本轮统一重新导出，列为历史参考。

每个模型的best按各自validation选择，不意味着可以根据本表test在S2、S3、soft之间再选择“正式最佳方法”。test已用于多轮研究比较，不属于未用于开发决策的新盲测。

## 5. 全部结果

### 5.1 主表：TP-only各组best/final

数量均指伪标签数，另加8GT。旧组节点以iter表示，新组节点以epoch表示，并提供对应iter。

|组|学生|权重|节点|Iter|Val Dice|Test Dice|Test IoU|
|---|---|---|---|---:|---:|---:|---:|
|A：448旧配方|S2|best|iter32800|32800|0.816123|0.846007|0.760456|
|A：448旧配方|S2|final|iter40000|40000|0.810172|0.838730|0.751171|
|A：448旧配方|S3|best|iter29800|29800|0.814231|**0.860283**|0.779007|
|A：448旧配方|S3|final|iter40000|40000|0.806084|0.853202|0.771334|
|B：580旧配方|S2|best|iter26000|26000|0.824722|0.851530|0.771841|
|B：580旧配方|S2|final|iter40000|40000|0.817908|0.851675|0.774503|
|B：580旧配方|S3|best|iter22800|22800|0.824242|0.841958|0.760371|
|B：580旧配方|S3|final|iter40000|40000|0.803916|0.843769|0.765014|
|C：580新配方|hard|best|epoch436|21364|0.819562|0.848592|0.769222|
|C：580新配方|hard|final|epoch816|39984|0.812377|0.854011|0.775123|
|C：580新配方|soft|best|epoch556|27244|0.827330|0.851243|0.773038|
|C：580新配方|soft|final|epoch816|39984|0.814561|**0.859530**|0.783362|
|D：448新配方|soft|best|epoch476|18088|**0.828689**|**0.855191**|0.770759|
|D：448新配方|soft|final|epoch816|31008|0.806991|0.851452|0.767965|

### 5.2 最早双路线410张：历史参考

|学生|权重|Iter|Val Dice|Test Dice|Test IoU|
|---|---|---:|---:|---:|---:|
|S2|best|32400|0.820152|0.855042|0.771485|
|S2|final|40000|0.809828|0.852245|0.767905|
|S3|best|20200|0.816272|0.845245|0.763529|
|S3|final|40000|0.812164|0.862539|0.786166|

这组不是“448张旧版本”。它包含TP＋PC候选和历史实现差别，不用于单独推断筛选顺序或新训练损失的收益。

### 5.3 关键差值：按原始精度计算

|比较|Test Dice变化|百分点变化|解释|
|---|---:|---:|---|
|B580旧S2 best－A448旧S2 best|+0.005522905|+0.5523|扩大池对旧S2有正向记录|
|B580旧S3 best－A448旧S3 best|-0.018324606|-1.8325|同样扩大池对旧S3明显不利|
|C580新soft best－A448旧S3 best|-0.009039296|-0.9039|整套新配方未超过旧S3|
|D448新soft best－A448旧S3 best|-0.005091890|-0.5092|同448张池的新配方仍未超过旧S3|
|D448新soft best－C580新soft best|+0.003947406|+0.3947|新版回到448张，best更高|
|D448新soft final－C580新soft final|-0.008077584|-0.8078|但final更低|

C组soft相对hard的best Test差为+0.002650989，配对bootstrap 95%区间为[-0.013394,+0.017837]，跨0；单seed下不能声称软化标签稳定提高性能。

D组best到final的Val从0.828689降至0.806991，Test从0.855191降至0.851452。说明继续训练到终点没有改善本次结果，但仅凭这一条曲线不能区分过拟合、优化波动等原因。

## 6. 哪些已经验证，哪些不能下结论

1. 筛选顺序变化能找回132张存在返回合格候选的训练图；不是模型突然多预测出132张图。
2. 合格候选不一定等于目标mask准确；折外诊断已经显示新增部分平均质量更低。
3. 旧S2与旧S3对扩大池的反应不同，因此不能只用池数量判断训练收益。
4. 新版用一个学生替代S2/S3双学生流程，但计算更简单不等于精度更高。
5. 当前TP-only主表中，旧448张S3的val-best对应Test=0.860283；新版448soft为0.855191，580soft为0.851243。尚无证据支持新配方在该口径超过旧S3。
6. final与best必须分别比较。不能因580soft final=0.859530较高就替换其已按validation选出的best。

## 7. 可复现性与本轮核查

|对象|已经完成的核查|
|---|---|
|448→580池变化|580张mask哈希核验；重新计算候选选择；原448路线/mask/权重保持|
|448旧S2/S3|找到完整40000iter权重；2026-09-12在GPU0两个任务并行补测best/final，400张test输出；权重前后哈希不变|
|580新hard/soft|相同初始化、epoch顺序及逐图增强；全部软标签量化与阈值形状检查；best/final测试完成|
|448新soft|全部448张软标签复算；原主mask一致；初始化及共有图像增强一致；816轮各456图不重复，31008步；best/final的validation复现与test完成|
|本文统一测试入口|分别用旧448 S3和新448soft验证legacy/single两条代码路径，各自best/final精确复现原始记录|

本文整理期间没有重训学生。只为验证附带的统一评估入口，重新导出了上述四个checkpoint的预测。不能将这种冻结权重重测称为多seed训练复现。

原始精度结果、checkpoint路径及SHA256存于同目录：`核验数据/环节二三_原始结果合集.json`。结果差值从原始数值计算；正文保留六位小数。

## 8. 服务器路径与公共环境

以下命令均在服务器Bash运行。先连接：

```bash
ssh violet@222.31.141.50
bash
```

每个新终端执行公共设置：

```bash
set -euo pipefail
PROJECT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY_STUDENT=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
OLD448="$PROJECT/work/kvasir_tp_student_mainline_20260907"
OLD580="$PROJECT/work/kvasir_tp_filterfirst_students_20260909"
NEW580="$PROJECT/new_project/experiments/single_student_hard_soft_20260909"
NEW448="$PROJECT/new_project/experiments/tp448_single_student_soft_20260912"
ARCHIVE="$PROJECT/new_project/stage23_old_new_comparison_20260912"
EVAL="$ARCHIVE/scripts/evaluate_student_checkpoints.py"
DATA="$PROJECT/work/kvasir_1pct_anchors/baseline_data"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export SC_SAM_ROOT="$PROJECT/third_party/SC-SAM"
unset ISIC_LIGHT_STRONG_AUG ISIC_RESIZED_CACHE_ROOT
cd "$PROJECT"
```

学生使用`mkunet_mamba`环境，Python3.9、PyTorch2.1.2+cu118。不要与《环节一》的SAM3 Python3.12环境混用。

|资产|位置|
|---|---|
|旧448首批池|`$OLD448/pseudo_manifest_original.jsonl`|
|旧448 S3共识|`$OLD448/S3_consensus/pseudo_consensus.jsonl`|
|新筛选580首批池|`$OLD580/pseudo_manifest_original.jsonl`|
|580旧S3共识|`$OLD580/S3_consensus/pseudo_consensus.jsonl`|
|580新标签清单|`$NEW580/data/train_manifest.jsonl`|
|448新标签清单|`$NEW448/data/train_manifest.jsonl`|
|统一测试入口|`$EVAL`|
|旧组权重|对应根目录`students/S2或S3/student_best.pth`、`student_final.pth`|
|新组权重|对应根目录`runs/hard或soft/student_best.pth`、`student_final.pth`|

## 9. 启动方式一：用已训练权重重测best/final

该入口只做预测与评估，不训练、不修改历史权重，输出目录必须不存在。legacy先冻结权重，再调用原导出程序；single沿用存档`train_student.py`的评估函数，并验证validation一致。

### 9.1 旧448 S2/S3

```bash
REPLAY=$(mktemp -d "$ARCHIVE/eval_old448_XXXXXX")
"$PY_STUDENT" "$EVAL" \
  --family legacy --run-root "$OLD448" --students S2 S3 \
  --output-dir "$REPLAY/results" --gpu 0
```

预期S2 best/final=0.846007/0.838730；S3=0.860283/0.853202。

### 9.2 580池＋旧S2/S3配方

```bash
REPLAY=$(mktemp -d "$ARCHIVE/eval_old580_XXXXXX")
"$PY_STUDENT" "$EVAL" \
  --family legacy --run-root "$OLD580" --students S2 S3 \
  --output-dir "$REPLAY/results" --gpu 0
```

预期S2 best/final=0.851530/0.851675；S3=0.841958/0.843769。

### 9.3 580新hard/soft

```bash
REPLAY=$(mktemp -d "$ARCHIVE/eval_new580_XXXXXX")
"$PY_STUDENT" "$EVAL" \
  --family single --run-root "$NEW580" --students hard soft \
  --output-dir "$REPLAY/results" --gpu 0
```

预期hard best/final=0.848592/0.854011；soft=0.851243/0.859530。

### 9.4 448新soft

```bash
REPLAY=$(mktemp -d "$ARCHIVE/eval_new448_XXXXXX")
"$PY_STUDENT" "$EVAL" \
  --family single --run-root "$NEW448" --students soft \
  --output-dir "$REPLAY/results" --gpu 0
```

预期soft best/final=0.855191/0.851452。所有评估输出包括权重冻结清单、mask、逐图指标、汇总和COMPLETE。

## 10. 启动方式二：从初始化重新训练旧S2/S3

本节复用冻结池及标签，运行已存档的旧训练程序，输出到全新目录。不是继续训练旧checkpoint，也不启动委员会、X3或B7。本轮只核对命令与历史`stages/train_S*.command.json`，没有重新执行这部分训练。

### 10.1 旧448配方：GPU0两个任务并行

```bash
TRAIN_SRC="$OLD448"
TRAIN_OUT=$(mktemp -d "$ARCHIVE/train_legacy448_XXXXXX")
mkdir -p "$TRAIN_OUT/logs"

"$PY_STUDENT" -u "$OLD580/code/cuda_runner.py" \
  "$TRAIN_SRC/code/run_t24_student.py" \
  --data-path "$DATA" \
  --labeled-list "$TRAIN_SRC/protocol/frozen_labeled_images.txt" \
  --seed 2026 --max-iterations 40000 --val-interval 200 --num-workers 4 \
  --pseudo-manifest "$TRAIN_SRC/pseudo_manifest_original.jsonl" \
  --output-dir "$TRAIN_OUT/students/S2" --experiment S2 --defer-test \
  > "$TRAIN_OUT/logs/S2.log" 2>&1 &
S2_PID=$!

"$PY_STUDENT" -u "$OLD580/code/cuda_runner.py" \
  "$TRAIN_SRC/code/run_t24_student.py" \
  --data-path "$DATA" \
  --labeled-list "$TRAIN_SRC/protocol/frozen_labeled_images.txt" \
  --seed 2026 --max-iterations 40000 --val-interval 200 --num-workers 4 \
  --pseudo-manifest "$TRAIN_SRC/S3_consensus/pseudo_consensus.jsonl" \
  --output-dir "$TRAIN_OUT/students/S3" --experiment S3 --defer-test \
  > "$TRAIN_OUT/logs/S3.log" 2>&1 &
S3_PID=$!

wait "$S2_PID"
wait "$S3_PID"
cp -a "$TRAIN_SRC/code" "$TRAIN_OUT/code"
"$PY_STUDENT" "$EVAL" \
  --family legacy --run-root "$TRAIN_OUT" --students S2 S3 \
  --output-dir "$TRAIN_OUT/evaluation" --gpu 0
```

`cuda_runner.py`只设置线程数及每进程25%显存分配上限，两个进程都使用公共环境指定的GPU0。执行前确认GPU0资源可用，不停止其他任务。原448训练记录是顺序训练；这里并行是运行调度变化，不作为已经验证的逐比特训练重现声明。

### 10.2 580池＋旧配方

重新执行10.1同一段训练命令，仅把开头两项换为：

```bash
TRAIN_SRC="$OLD580"
TRAIN_OUT=$(mktemp -d "$ARCHIVE/train_legacy580_XXXXXX")
```

其余参数保持不变。程序从各自manifest读取448或580张；不手工把两组清单混合。

## 11. 启动方式三：重新训练新版单学生

### 11.1 复制准备好的数据清单与存档代码到新目录

先选择来源，只选一个：

```bash
# 580张新版：来源已有hard/soft数据
SOURCE="$NEW580"
# 若要复现448张新版soft，将上一行改为 SOURCE="$NEW448"

NEW_RUN=$(mktemp -d "$ARCHIVE/train_single_XXXXXX")
mkdir -p "$NEW_RUN/data" "$NEW_RUN/logs"
cp -a "$SOURCE/code" "$NEW_RUN/code"
cp "$SOURCE/config.json" "$SOURCE/initial_model.pth" "$NEW_RUN/"
cp "$SOURCE/data/train_manifest.jsonl" \
   "$SOURCE/data/validation_manifest.jsonl" \
   "$SOURCE/data/test_manifest.jsonl" "$NEW_RUN/data/"
"$PY_STUDENT" - "$NEW_RUN" "$SOURCE" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
config = json.loads((root / 'config.json').read_text())
config.update(gpu=0, replay_source=sys.argv[2])
(root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
PY
```

manifest引用原图像和标签绝对路径，直接只读复用。580与448的存档trainer分别具有588/456样本数、39984/31008步的断言，必须使用对应SOURCE，不能只改清单却混用trainer。

### 11.2 新版soft训练与测试

```bash
"$PY_STUDENT" -u "$NEW_RUN/code/train_student.py" \
  --root "$NEW_RUN" --variant soft \
  > "$NEW_RUN/logs/train_soft.log" 2>&1

"$PY_STUDENT" "$EVAL" \
  --family single --run-root "$NEW_RUN" --students soft \
  --output-dir "$NEW_RUN/evaluation_soft" --gpu 0
```

训练入口从保存的`initial_model.pth`重新开始。580对应816epoch/39984iter，448对应816epoch/31008iter；评估自动读取best/final，不按test选epoch。

580张hard对照可在新目录下使用相同命令，把`--variant soft`换为`--variant hard`，评估改为`--students hard`、新输出目录。448张hard尚未实验，不能把它写入已有结果。

## 12. 标签构造脚本、日志与避免误启动完整流程

|工作|实际实现/存档|
|---|---|
|旧448筛选|`$OLD448/code/mainline.py`的`pseudo_prepare()`|
|580筛选顺序诊断|`$PROJECT/work/kvasir_tp_pseudo_filter_20260909`；580预览经复算后正式用于训练|
|580旧配方准备|`$OLD580/code/run_students.py`的`setup()`|
|旧S3共识|对应根目录`code/prepare_phase1_s3_consensus.py`|
|580新版软标签准备|`$NEW580/code/prepare_experiment.py`|
|448新版子集与标签核查|`$NEW448/code/setup_tp448_soft.py`|
|448新版自动训练＋测试|`$NEW448/code/run_tp448_soft.py`|

如果只需复现旧S3标签，可以输出到独立目录：

```bash
# SOURCE_LEGACY可设为OLD448或OLD580
SOURCE_LEGACY="$OLD448"
LABEL_RUN=$(mktemp -d "$ARCHIVE/legacy_consensus_XXXXXX")
"$PY_STUDENT" "$SOURCE_LEGACY/code/prepare_phase1_s3_consensus.py" \
  --original-manifest "$SOURCE_LEGACY/pseudo_manifest_original.jsonl" \
  --quality-root "$SOURCE_LEGACY/quality" \
  --output-root "$LABEL_RUN/S3_consensus" \
  --min-bridge 0 --max-bridge 6 --beta 4 --pixel-min-weight 0.1
```

不能直接执行历史`mainline.py`/`run_students.py`的完整pipeline来代替本文命令：它们还会继续委员会、X3、B7。历史prepare/run脚本还可能使用固定输出目录；本文使用独立目录或冻结标签复用，避免覆盖原记录。

|版本|训练日志|已完成结果|
|---|---|---|
|448旧S2/S3|`$OLD448/students/S2或S3/train.log`|`$PROJECT/new_project/experiments/tp448_s2_s3_test_20260912/results.json`|
|580旧S2/S3|`$OLD580/logs/train_S2.log`、`train_S3.log`|`$OLD580/s2_s3_test_audit/results.json`|
|580新hard/soft|`$NEW580/logs/train_hard.log`、`train_soft.log`|`$NEW580/results.json`及`final_checkpoint_test/results.json`|
|448新soft|`$NEW448/logs/train_soft.log`|`$NEW448/results.json`|

新版每轮的`runs/<variant>/train_epochs.jsonl`包含epoch、iter、lr、样本数和顺序哈希；`validation.jsonl`保存验证历史。日志位置与启动命令均已按实际文件核对。

## 13. 本地交付与当前判断

文件夹：`F:\medsam3\新老环节对比`。

- `环节二与环节三.md`：本文。
- `scripts/evaluate_student_checkpoints.py`：旧/新学生统一测试启动入口。
- `核验数据/环节二三_原始结果合集.json`：八类来源文件的原始数据、绝对路径和SHA256。
- `448张_S2_S3补测.md`、`448张_新版软标签结果.md`：本轮补测和新训练详细结果。

服务器归档：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/stage23_old_new_comparison_20260912`。

当前应保留“旧448 S3”和“新版448/580 soft”三组作为后续研究对照。结果支持把筛选与训练放在一起评价，但还不足以把新版整套配方定为精度更高的新基线。
