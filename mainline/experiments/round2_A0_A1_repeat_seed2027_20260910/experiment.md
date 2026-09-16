# A0/A1 seed2027重复训练

分别对照已有新设置seed2026的A0和A1，只改变随机种子。原始数据划分、池成员、标签、每轮顺序生成算法及训练策略不变。

| 组 | 伪标签+GT | GPU | Epoch | 更新次数 |
|---|---:|---:|---:|---:|
| A0 | 428+8 | 0 | 10 | 4360 |
| A1 | 596+8 | 1 | 10 | 6040 |

原图直接1008；全模块LoRA rank16/alpha32/dropout0.1，batch1、样本等权。cosine学习率5e-5至5e-7，每次更新前梯度总范数裁剪1.0。无warmup、无梯度累积。

固定colon polyp训练和validation，每轮完整100张、Dice256选择best。训练结束后对各自best执行test100有文本主评估及空文本诊断，不按test改选权重。

数据仅复用读取，旧数据/权重/结果哈希存于OLD_ARTIFACTS_HASHES.json。所有新训练输出在本目录，seed2027的初始LoRA必须与A2 seed2027完全一致。

日志：train_A0_seed2027.log、train_A1_seed2027.log。
