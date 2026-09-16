# Kvasir SAM3-base 单图预测：空文本与 colon polyp

原 test 100 张；未微调 SAM3-base；输入与评测均为 1008×1008；无 anchor、桥接、点框、学生或 Router。两组仅改变 query_text。

类别概率 >=0.5 的 queries 合并为前景，mask 概率 >=0.5 二值化。空文本使用 query_text=""，接口仍保留 query 机制，不能解释成完全没有 query 的自动分割。

| 提示 | Test Dice | Test IoU | Dice 中位数 | 非空预测数 |
|---|---:|---:|---:|---:|
| 空文本 | 0.361394 | 0.304767 | 0.197243 | 90/100 |
| colon polyp | 0.461427 | 0.432159 | 0.263805 | 61/100 |

文本提示 Dice 提升 10.0033 个百分点。提示词在测试前按历史协议固定，未搜索 test 上最优提示。

两组共 200 张最终 mask 均已保存，并从保存 PNG 和 COCO GT 重新计算逐图 Dice/IoU，误差为零。输出目录 empty/masks 与 category/masks；逐图结果 per_image.jsonl；汇总 summary.json。

colon polyp 的 0.461427 与旧报告 Direct 1008+text 一致。此前路线传播实验 canvas=256，并带 anchor 信息；这里是单图1008，比较时需要保留协议差异。

来源脚本：scripts/eval_sam3_lora_direct_split.py；本次副本添加输入审计、逐图记录和 mask 导出，未改模型和阈值。
