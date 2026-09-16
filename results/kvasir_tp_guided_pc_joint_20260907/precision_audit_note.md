# 数值核验修正说明

初次参考实现误差约0.00028771。后续代码检查确认，除model_builder启用TF32，sam3_tracking_predictor.py:51还进入持续bfloat16 autocast。因此初次误差归因应表述为混合精度，而非只归因TF32。新局部相似度与attention使用float64，核验误差1.30343e-8；旧TP/Top8精确复现。仅修改数值执行精度，未改变公式、参数、划分或checkpoint；修改发生于任何新路线传播之前。
