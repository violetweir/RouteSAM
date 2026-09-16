# 当前 580 张版本：S2/S3 冻结 checkpoint 的 test 评估

按用户要求评估既有 validation-best 与 final-40000，不训练、不按 test 选择模型。使用与 X3 单图结果相同的导出和 Dice 口径：256×256、前景概率阈值 0.5、逐图 Dice 后对完整 100 张求平均。

| 模型 | checkpoint | 步数 | validation Dice（训练入口） | test Dice | test IoU |
|---|---|---:|---:|---:|---:|
| S2 | valbest | 26000 | 0.824722 | 0.851530 | 0.771841 |
| S2 | final | 40000 | 0.817908 | 0.851675 | 0.774503 |
| S3 | valbest | 22800 | 0.824242 | 0.841958 | 0.760371 |
| S3 | final | 40000 | 0.803916 | 0.843769 | 0.765014 |

validation 列来自训练日志；test 列统一采用已有导出入口。四个 checkpoint 与全部 400 张输出 mask 均记录 SHA256。既有 X3-best test Dice 为 0.853667。
