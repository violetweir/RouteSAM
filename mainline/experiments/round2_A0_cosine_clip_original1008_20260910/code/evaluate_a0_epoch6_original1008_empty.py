"""User-requested, fixed A0 epoch-6 direct test audit; no training mutation."""
import json
import shutil
import traceback
from pathlib import Path
import direct_eval_original1008_empty as d

def main():
    dest = d.R / 'audits/A0_epoch6_original1008_empty_test'
    dest.mkdir(parents=True, exist_ok=False)
    d.D = dest
    source = d.R / 'runs/A0/epoch6.pt'
    frozen = dest / 'epoch6.pt'
    shutil.copy2(source, frozen)
    digest = d.sha(source)
    assert d.sha(frozen) == digest
    val = json.loads((d.R / 'runs/A0/validation_epoch6/summary.json').read_text())
    d.save(dest / 'AUDIT_FROZEN.json', dict(
        arm='A0', epoch=6, source=str(source), checkpoint_sha256=digest,
        validation_original1008_text_reference=val, reason='User requested original1008 empty-text test for the same A0 epoch6 checkpoint',
        role='diagnostic; does not change validation-based final checkpoint selection',
        protocol='single image, empty text string, original image directly resized to 1008; metrics256, no points/boxes/TP/Router/B7',
        training_policy='Independent diagnostic only; does not alter currently running trainings'))
    d.save(dest / 'status.json', dict(stage='initializing'))
    d.torch.set_num_threads(4)
    trainer = d.t.SAM3TrainerNative(str(d.R / 'A0.yaml'))
    model = trainer.model
    model.num_interactive_steps_val = 0
    d.torch.set_autocast_cache_enabled(False)
    d.torch.clear_autocast_cache()
    d.load_adapter(model, d.torch.load(frozen, map_location='cpu', weights_only=True))
    d.save(dest / 'status.json', dict(stage='test', epoch=6))
    result = d.evaluate(model, 'test', 'test')
    assert result['count'] == 100
    assert d.sha(source) == digest and d.sha(frozen) == digest
    d.save(dest / 'results.json', dict(epoch=6, checkpoint_sha256=digest, validation_original1008_text_reference=val, test=result, diagnostic=True))
    report = '\n'.join([
        '# A0 epoch 6：原图1008、空文本 test 补测', '',
        '在训练结束后，按用户明确要求固定 A0 epoch 6 做一次 test 补测。本补测不修改当前任何训练，最终权重仍按 validation 选择。', '',
        '| 权重 | 参考 Val Dice（原图1008、有文本） | Test Dice | Test IoU | Test 数量 |',
        '|---|---:|---:|---:|---:|',
        f"| A0 epoch 6 | {val['dice']:.9f} | {result['dice']:.9f} | {result['iou']:.9f} | {result['count']} |", '',
        '单张图像加空字符串文本（无语义文本提示）；原图直接缩放至1008输入模型，不经过256中间缩放，输出与GT在256上评估，逐图宏平均。没有点/框提示、TP传播、Router或B7。所有预测落盘并冻结后才读取GT。', '',
        '本次属于训练结束后的文本提示消融，不替代训练结束后的 validation-best 评测，也不依据此 test 结果调整训练。本次仅修改推理图像的输入缩放；训练权重来自原图直接1008的第二轮A0，Dice仍在256上计算。表中Val使用colon polyp文本，本次未重测Val。', '',
        f'权重 SHA256：`{digest}`', ''])
    (dest / 'report.md').write_text(report, encoding='utf-8')
    d.save(dest / 'status.json', dict(stage='complete', test=result))
    (dest / 'COMPLETE').write_text('complete\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        dest = d.R / 'audits/A0_epoch6_original1008_empty_test'
        if dest.exists():
            (dest / 'FAILED.txt').write_text(traceback.format_exc())
        raise
