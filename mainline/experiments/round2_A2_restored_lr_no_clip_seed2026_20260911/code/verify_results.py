from pathlib import Path
import json, math, hashlib
import numpy as np
from PIL import Image

B=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments')
R=B/'round2_A2_restored_lr_no_clip_seed2026_20260911'
D=R/'runs/seed2026';A=R/'audits/epoch10_test'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (D/'COMPLETE').exists() and (A/'COMPLETE').exists()
best=read(D/'results.json');final=read(A/'results.json')
epochs=[json.loads(s) for s in (D/'epochs.jsonl').read_text().splitlines()]
u=[json.loads(s) for s in (D/'updates.jsonl').read_text().splitlines()]
assert len(epochs)==10 and len(u)==5110
assert all(not x['clipped'] and not x['gradient_clipping_enabled'] and x['grad_norm_before']==x['grad_norm_after'] and math.isfinite(x['loss']) and math.isfinite(x['grad_norm_before']) for x in u)
assert all(math.isclose(x['learning_rate'],5e-7+(5e-5-5e-7)*(1+math.cos(math.pi*i/5110))/2,rel_tol=1e-9) for i,x in enumerate(u))
assert sha(D/'best.pt')==read(D/'WEIGHTS_FROZEN.json')['best_sha256']
assert sha(D/'epoch10.pt')==final['checkpoint_sha256']==sha(A/'epoch10.pt')
for p,h in read(R/'BASELINE_HASHES.json').items():assert sha(Path(p))==h
gt={r['merged_id']:r for r in read(R/'data/shared/test.json')}
for dest,res in [(D/'test_best',best['test']),(D/'test_best_empty',best['test_empty']),(A/'test_text',final['test']),(A/'test_empty',final['test_empty'])]:
    masks=read(dest/'PREDICTIONS_FROZEN.json')
    assert len(masks)==100 and {p['target_id'] for p in masks}==set(gt)
    vals=[]
    for p in masks:
        assert sha(Path(p['mask_path']))==p['mask_sha256']
        m=np.array(Image.open(p['mask_path']))>127
        g=np.array(Image.open(gt[p['target_id']]['mask_file_name']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
        inter=int((m&g).sum());total=int(m.sum())+int(g.sum());union=total-inter
        vals.append((2*inter/total if total else 1.,inter/union if union else 1.))
    avg=np.mean(vals,axis=0)
    assert abs(avg[0]-res['dice'])<1e-12 and abs(avg[1]-res['iou'])<1e-12
baseline=read(B/'round2_A1_A2_cosine_clip_original1008_20260910/runs/A2/results.json')
out=dict(best=best,final=final,clipped_baseline=baseline,updates_verified=5110,masks_verified=400,old_artifacts_unchanged=True,epochs=[dict(epoch=x['epoch'],val=x['validation']['dice'],train_fit=x['train_label_fit']['dice'],loss=x['loss']) for x in epochs])
(R/'verified_summary.json').write_text(json.dumps(out,indent=2))
report='''# A2 seed2026：恢复学习率并关闭裁剪，最佳与最后权重测试

配置：cosine 5e-5→5e-7，关闭梯度裁剪，A2 503伪标签+8GT，10epochs，原图直接1008。主评估colon polyp，指标为256逐图宏平均，test100。最佳指按Val Dice选择，最后权重为用户要求的epoch10诊断测试。

|设置与权重|epoch|Val Dice|Test Dice|Test IoU|空文本Test Dice|
|---|---:|---:|---:|---:|---:|
'''
for name,r,ep in [('同学习率、裁剪1.0最佳',baseline,baseline['best_epoch']),('关闭裁剪、Val最佳',best,best['best_epoch']),('关闭裁剪、最后',final,10)]:
    report+=f"|{name}|{ep}|{r['validation']['dice']:.6f}|{r['test']['dice']:.6f}|{r['test']['iou']:.6f}|{r['test_empty']['dice']:.6f}|\n"
report+='\n核验5110次更新均未裁剪且数值有限，cosine符合配置；400张预测的哈希与重算指标一致，最佳及最终权重哈希一致，旧对照文件未修改。\n'
report+='\n|epoch|Val Dice|训练监测标签拟合Dice|\n|---|---:|---:|\n'
for x in epochs:report+=f"|{x['epoch']}|{x['validation']['dice']:.6f}|{x['train_label_fit']['dice']:.6f}|\n"
(R/'training_report.md').write_text(report,encoding='utf-8')
print(json.dumps(out))
