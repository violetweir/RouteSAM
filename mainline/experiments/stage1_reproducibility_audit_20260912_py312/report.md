# 环节1：TP+PC 与 TP-only 可复现性核查

核查日期：2026-09-12。结果：PASS。

本次没有重新进行 GPU 传播，而是用冻结候选、原 validation 记录和原 Router 代码重新拟合、选路，并逐像素重算全部 1400 张候选的指标。9月6日保存的重新正向/返回传播结果，与更早的旧候选逐像素比较，1400/1400 相同。

| 指标 | TP+PC b0–b6 | TP-only b0–b6 |
|---|---:|---:|
| Test Dice | 0.8740830338579912 | 0.8854326463411537 |
| Test IoU | 0.8139856609477036 | 0.8282740566505135 |
| Oracle Dice | 0.9214706098083678 | 0.9074262547898724 |
| 最终选择与存档一致 | 100/100 | 100/100 |
| 重新拟合 Router 参数最大绝对差 | 0 | 0 |

共同标准：原 train/validation/test=800/100/100，原8张标注；SAM3-base，SAM3特征256、传播画布256；同一100张test及对应GT；b0–b6；输出与GT以NEAREST处理至256，uint8>127二值化；每图Dice=2交集/(预测面积+GT面积)，IoU=交集/并集，最后对100图取宏平均。全部test参与，不按质量删图。最终mask为Router选中的一张候选，不做融合。

Router均为ridge=1，在同一100张validation上拟合，拟合会使用validation GT Dice。测试选路不读取test GT，选择先冻结再计算指标。联合Router使用两路线的1400条validation候选及模式特征；TP Router使用700条候选，不含模式特征。故测试标准一致，但这是各自匹配Router的两套方案比较，不是固定同一个Router的单因素消融。

本次检查了4份输入质量文件的原始SHA256、原Router代码SHA256、每张最终mask的存档SHA256，以及新旧候选对应GT的一致性。100张test身份和train/validation/test互斥关系均通过。

复现环境使用服务器sam3环境Python3.12。初次使用另一Python3.9环境在严格参数相等断言处停止，未生成评测结果；原环境重新运行后参数差为0、选路全部一致。复现应固定原环境。

边界：上述证明冻结协议及其结果可重现，不证明TP-only在新数据上必然更优。本test已用于多轮研究对照，不能再视为从未用于开发决策的全新盲测。

服务器归档：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/stage1_reproducibility_audit_20260912_py312`，包含audit.json、冻结选路清单、核查程序与COMPLETE。
