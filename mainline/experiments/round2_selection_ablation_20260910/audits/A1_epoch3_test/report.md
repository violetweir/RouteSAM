# A1 epoch 3：用户指定的 test 补测

在原定两组训练结束前，按用户明确要求固定 A1 epoch 3 做一次 test 补测。两组继续原定 10 epoch，最终权重仍按 validation 选择。

| 权重 | Val Dice | Test Dice | Test IoU | Test 数量 |
|---|---:|---:|---:|---:|
| A1 epoch 3 | 0.880858535 | 0.889508084 | 0.829941043 | 100 |

单张图像加固定 colon polyp 文本；图像先缩至256再输入1008模型，输出与GT在256上评估，逐图宏平均。没有点/框提示、TP传播、Router或B7。所有预测落盘并冻结后才读取GT。

本次属于训练期间的指定 checkpoint 诊断，不替代训练结束后的 validation-best 评测，也不依据此 test 结果调整训练。旧 e33 的1008评测与本次口径不同。

权重 SHA256：`b430a51edcc924a7c436db6f88b51472329d1b771103ac56cebdf0fca2524a4d`
