import json,shutil,traceback
import train_direct as d
d.D=d.R/'audits/epoch19_test_text_20260912'

def main():
    d.D.mkdir(parents=True,exist_ok=False)
    src=d.R/'runs/seed2026/epoch19.pt';dst=d.D/'epoch19.pt'
    shutil.copy2(src,dst);digest=d.sha(dst);assert digest==d.sha(src)
    val=json.loads((d.R/'runs/seed2026/validation_epoch19/summary.json').read_text())
    d.save(d.D/'FROZEN.json',dict(epoch=19,seed=2026,checkpoint_sha256=digest,validation=val,prompt='colon polyp',input='original RGB directly1008',metric='macroDice256',reason='User-requested epoch19 test audit during 50epoch training; no change to training or best-checkpoint selection'))
    d.torch.set_num_threads(4)
    d.torch.cuda.set_per_process_memory_fraction(.40)
    trainer=d.t.SAM3TrainerNative(str(d.R/'seed2026.yaml'))
    model=trainer.model;model.num_interactive_steps_val=0
    d.torch.set_autocast_cache_enabled(False)
    d.load_adapter(model,d.torch.load(dst,map_location='cpu',weights_only=True))
    d.TEXT_PROMPT='colon polyp'
    result=d.evaluate(model,'test','test_text')
    assert result['count']==100 and d.sha(src)==digest
    rows=json.loads((d.R/'data/shared/test.json').read_text())
    masks=json.loads((d.D/'test_text/PREDICTIONS_FROZEN.json').read_text())
    assert len(masks)==100 and {p['target_id'] for p in masks}=={r['merged_id'] for r in rows}
    byid={r['merged_id']:r for r in rows};vals=[]
    for p in masks:
        assert d.sha(d.Path(p['mask_path']))==p['mask_sha256']
        m=d.np.array(d.PIL.open(p['mask_path']).convert('L'))>127
        g=d.np.array(d.PIL.open(byid[p['target_id']]['mask_file_name']).convert('L').resize((256,256),d.PIL.Resampling.NEAREST))>127
        inter=int((m&g).sum());total=int(m.sum())+int(g.sum());union=total-inter
        vals.append((2*inter/total if total else 1.,inter/union if union else 1.))
    avg=d.np.mean(vals,axis=0)
    assert abs(avg[0]-result['dice'])<1e-12 and abs(avg[1]-result['iou'])<1e-12
    d.save(d.D/'results.json',dict(epoch=19,seed=2026,validation=val,test=result,checkpoint_sha256=digest,verified_masks=100,original_weight_unchanged=True))
    report=f'''# A0 50轮训练：epoch19 Test审计

- 权重：epoch19；seed2026；cosine 5e-5→5e-7覆盖50轮，关闭梯度裁剪。
- 输入：原图直接1008；文本提示colon polyp；直接预测，无图像框/点提示。
- 指标：完整test100，输出与GT在256分辨率逐图宏平均。
- Val Dice：{val['dice']:.9f}
- Test Dice：{result['dice']:.9f}
- Test IoU：{result['iou']:.9f}

用户指定epoch19的中途诊断测试；50轮训练继续，正式最佳权重仍按Val Dice选择。全部100张预测哈希与重算指标核验，原权重未修改。
'''
    (d.D/'report.md').write_text(report,encoding='utf-8')
    (d.D/'COMPLETE').write_text('complete\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        if d.D.exists():(d.D/'FAILED.txt').write_text(traceback.format_exc())
        raise
