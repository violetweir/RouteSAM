# A2降低学习率：双种子对照

与同种子旧A2对照，仅将初始学习率5e-5改为1e-5、cosine终点5e-7改为1e-7。

- seed2026/GPU0，seed2027/GPU1，各10轮、5110更新。
- 同一503伪标签+8GT池、原图1008、等权batch1、全模块LoRA、clip1.0，无warmup或梯度累积。
- 每轮记录原生各项loss、梯度范数、完整validation100 Dice。
- 固定训练监测集为64张伪标签图+8张GT，按target_id哈希选取，与标签质量无关；监测预测与训练监督标签的Dice，不能当作真实训练GT Dice。监测保存恢复随机数状态，不改变后续dropout随机序列。
- 两组按colon polyp的validation最佳权重冻结后，统一test100有文本主评估及空文本诊断。
- 原数据只读复用、旧结果不覆盖。
