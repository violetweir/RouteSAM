from pathlib import Path
import json,hashlib,collections,shutil
import numpy as np
from PIL import Image
import router
R=Path(__file__).resolve().parents[1];P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
read=lambda p:json.loads(p.read_text())
lines=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
new=lines(R/'quality_root/sam3enc_anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl');old=lines(R/'baseline/propagation_quality.jsonl')
sets={'original21':old,'automatic21':new};N=259
assert len(new)==len(old)==N*7 and all(r['status']=='success' for r in new)
assert all(not any(k.startswith('gt_') for k in r) for r in new)
for rows in sets.values():
 for r in rows:
  p=Path(r['forward_mask_path']);p=p if p.is_absolute() else P/p;r['forward_mask_path']=str(p)
  assert sha(p)==r['forward_mask_sha256'];r['feature_mode']='anchor_conditioned_target_pooling'
save(R/'PREDICTIONS_FROZEN.json',{name:[dict(route_id=r['route_id'],mask_sha256=r['forward_mask_sha256']) for r in rows] for name,rows in sets.items()})
folds=read(R/'validation_folds_frozen.json')['target_to_fold']
results={};per={}
for name,rows in sets.items():
 for r in rows:
  m=np.array(Image.open(r['forward_mask_path']).convert('L'))>127
  gp=Path(r['target_mask_path_evaluation_only']);gp=gp if gp.is_absolute() else P/gp
  g=np.array(Image.open(gp).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
  inter=int((m&g).sum());total=int(m.sum())+int(g.sum());d=2*inter/total if total else 1.
  if 'gt_dice_evaluation_only' in r:assert abs(d-r['gt_dice_evaluation_only'])<1e-6
  r['gt_dice_evaluation_only']=d
 groups=router.grouped(rows);assert len(groups)==N and set(groups)==set(folds)
 selected={};models={}
 for fold in range(5):
  train=[r for r in rows if folds[r['target_id']]!=fold]
  model=router.Ridge.fit(train,ridge=1.,include_mode=False)
  models[str(fold)]=vars(model)
  for target,rr in groups.items():
   if folds[target]!=fold:continue
   clean=[{k:v for k,v in r.items() if not k.startswith('gt_') and k!='target_mask_path_evaluation_only'} for r in rr]
   c=max(clean,key=lambda r:(model.score(r),-r['bridge_count'],r['route_id']))
   chosen=next(r for r in rr if r['route_id']==c['route_id'])
   selected[target]=dict(target_id=target,fold=fold,route_id=c['route_id'],anchor_id=c['anchor_id'],score=model.score(c),dice=chosen['gt_dice_evaluation_only'],oracle=max(r['gt_dice_evaluation_only'] for r in rr))
 dest=R/name;dest.mkdir();save(dest/'fold_models.json',models);save(dest/'oof_predictions.json',selected)
 result=dict(n=N,router_oof_dice=float(np.mean([r['dice'] for r in selected.values()])),oracle_dice=float(np.mean([r['oracle'] for r in selected.values()])),fixed_bridge={str(b):float(np.mean([r['gt_dice_evaluation_only'] for r in rows if r['bridge_count']==b])) for b in range(7)},candidate_sources=dict(collections.Counter(r['anchor_id'] for r in rows)),selected_sources=dict(collections.Counter(r['anchor_id'] for r in selected.values())))
 result['oracle_gap']=result['oracle_dice']-result['router_oof_dice'];results[name]=result;per[name]=selected
comparison={};rng=np.random.default_rng(2026);targets=sorted(folds)
for metric in ['dice','oracle']:
 d=np.array([per['automatic21'][t][metric]-per['original21'][t][metric] for t in targets]);bs=d[rng.integers(0,N,size=(10000,N))].mean(1)
 comparison[metric]=dict(delta=float(d.mean()),paired_bootstrap_95ci=np.quantile(bs,[.025,.975]).tolist())
out=dict(results=results,comparison=comparison,router_protocol='5-fold target-grouped OOF within original val259; fixed ridge1, same frozen folds, separate models for original/automatic anchor candidates; no test, no Kvasir router transfer',verified_masks=N*14)
save(R/'results.json',out)
report='''# ISIC2018自动选21张：TP验证结果

固定原train2075/val259/test260划分，自动选21张与原21张比较。选图仅用train RGB的SAM3-base1008整图+局部特征；TP检索与传播均256，beam32、b0-b6不变。

没有可复用的已确认ISIC独立Router文件，因此使用同一预冻结的val目标5折、固定ridge1进行原/新候选各自OOF Router评分。不转用Kvasir Router，不在训练折上报分。所有候选mask冻结后计算GT指标。

|参考集|Val OOF Router Dice|Val Oracle Dice|Oracle差距|
|---|---:|---:|---:|
'''
for name,r in results.items():report+=f"|{name}|{r['router_oof_dice']:.6f}|{r['oracle_dice']:.6f}|{r['oracle_gap']:.6f}|\n"
report+='\n|路线|原21张|自动21张|\n|---|---:|---:|\n'
for b in range(7):report+=f"|b{b}|{results['original21']['fixed_bridge'][str(b)]:.6f}|{results['automatic21']['fixed_bridge'][str(b)]:.6f}|\n"
report+='\n这是一组自动参考集的开发验证，不证明跨选图种子稳定性。自动选图停止数只是特征覆盖启发式。预算和特征公式沿用Kvasir方法，未根据ISIC GT手工改样本；test尚未评估。\n'
(R/'report.md').write_text(report,encoding='utf-8');print(json.dumps(out),flush=True)
