# Kvasir：冻结 B 候选池，改进替换质量判断

本轮已于 2026-09-08 17:44 完成。主实验 validation Dice 为 0.854187，低于原 B 选择器的 0.857937，未通过预设后续 test 门槛。完整结果见同目录 report.md 与 results.json。

候选保持为原 TP b0–b6 七候选与 B 的备用 TP anchor 七候选，去重后共 1393 条路径。默认结果仍为原 TP 独立 Router。沿用原 validation 100 张与训练 8/792 身份，不使用学生网络，本轮不评估 test。

## 预先固定的三个对照

| 方案 | 增益选择器特征 | 当前 validation 折外 Dice |
|---|---|---:|
| legacy | 原 15 个特征 | 0.857937，已逐图复现 |
| remove_object_scores | 删除旧对象分数及其差值，保留前 13 个特征 | 0.856882，已完成 |
| target_quality（主实验） | 前 13 个特征 + 目标帧预测 IoU、对象存在概率、mask logit 稳定性，各取 TP 值及辅助相对差 | 待全量提取与验证 |

删除旧对象分数的结果相对 legacy 为 -0.001055，配对 bootstrap 95% 区间 [-0.007164, 0.003657]。这不能证明旧分数无用；它可能保留参考对象相关信息，只是不应被解释为当前目标 mask 的质量。

目标端分数通过只读包装 tracker 方法记录，返回张量不做修改。前 14 条路径覆盖 b0–b6，均与冻结 mask 逐像素一致。完整提取也逐条核对，发生不一致立即停止，不混用不同预测。

选择器结构、专家拟合、Ridge 正则 10/100、阈值 0.02/0.05/0.10 与 TP 回退均保持。仍按目标图做 5 外层、4 内层、3 层专家交叉拟合。主实验预先定为 target_quality，不根据外层成绩挑选控制组作为赢家。

预测 IoU、对象存在概率和稳定性都是待验证证据，不等于真实 Dice。目标 GT 只用于选择器训练折与评估，不参与新评分提取或传播。

全量任务已在远端运行，结束后自动评估、校验并生成 report.md 与 results.json。完成报告将归档到 reproduction_reports/Kvasir_B_target_quality_validation_20260908.md。

远端目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_rethink_20260908/b_target_quality`。
