from pathlib import Path
import json,hashlib,collections,shutil
import numpy as np
from PIL import Image
import router
R=Path(__file__).resolve().parents[1]
MODE='sam3enc_anchor_conditioned_target_pooling'
read=lambda p:json.loads(p.read_text())
lines=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
scorer=router.Ridge(**read(R/'router.json'))
new=lines(R/f'quality_root/{MODE}/propagation_quality_validation/propagation_quality.jsonl')
old=lines(R/'baseline/propagation_quality.jsonl')
assert len(new)==len(old)==700 and all(r['status']=='success' for r in new)
assert all(not any(k.startswith('gt_') for k in r) for r in new)
allsets={'original8':old,'automatic8':new};choices={};groups={}
for name,rows in allsets.items():
 groups[name]=router.grouped(rows);assert len(groups[name])==100
 choices[name]={}
 for t,rs in groups[name].items():
  assert len(rs)==7
  clean=[{k:v for k,v in r.items() if not k.startswith('gt_') and k!='target_mask_path_evaluation_only'} for r in rs]
  for r in clean:r['feature_mode']='anchor_conditioned_target_pooling'
  chosen=max(clean,key=lambda r:(scorer.score(r),-int(r['bridge_count']),r['route_id']))
  choices[name][t]=dict(route_id=chosen['route_id'],score=scorer.score(chosen),anchor_id=chosen['anchor_id'],bridge_count=chosen['bridge_count'])
  for r in rs:assert sha(Path(r['forward_mask_path']))==r['forward_mask_sha256']
assert set(groups['original8'])==set(groups['automatic8'])
save(R/'PREDICTIONS_AND_SELECTIONS_FROZEN.json',dict(selections=choices,new_quality_sha256=sha(R/f'quality_root/{MODE}/propagation_quality_validation/propagation_quality.jsonl'),router_sha256=sha(R/'router.json')))
results={};per_target={}
for name,rows in allsets.items():
 for r in rows:
  m=np.array(Image.open(r['forward_mask_path']).convert('L'))>127
  g=np.array(Image.open(r['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
  inter=int((m&g).sum());total=int(m.sum())+int(g.sum());union=total-inter
  val=2*inter/total if total else 1.
  if 'gt_dice_evaluation_only' in r:assert abs(val-r['gt_dice_evaluation_only'])<1e-6
  r['gt_dice_evaluation_only']=val;r['gt_iou_evaluation_only']=inter/union if union else 1.
 dest=R/name;(dest/'selected_masks').mkdir(parents=True,exist_ok=False)
 scored=[]
 for t,rs in groups[name].items():
  chosen=next(r for r in rs if r['route_id']==choices[name][t]['route_id'])
  oracle=max(rs,key=lambda r:r['gt_dice_evaluation_only'])
  scored.append(dict(target_id=t,selected_dice=chosen['gt_dice_evaluation_only'],selected_iou=chosen['gt_iou_evaluation_only'],oracle_dice=oracle['gt_dice_evaluation_only'],selected_route=chosen['route_id'],selected_anchor=chosen['anchor_id'],oracle_route=oracle['route_id']))
  shutil.copy2(chosen['forward_mask_path'],dest/'selected_masks'/(t.replace('::','__')+'.png'))
 per_target[name]={r['target_id']:r for r in scored}
 save(dest/'per_target_metrics.json',scored)
 r=dict(n=100,router_dice=float(np.mean([x['selected_dice'] for x in scored])),oracle_dice=float(np.mean([x['oracle_dice'] for x in scored])),fixed_bridge={str(b):float(np.mean([x['gt_dice_evaluation_only'] for x in rows if x['bridge_count']==b])) for b in range(7)},candidate_sources=dict(collections.Counter(x['anchor_id'] for x in rows)),selected_sources=dict(collections.Counter(x['selected_anchor'] for x in scored)))
 r['oracle_gap']=r['oracle_dice']-r['router_dice'];results[name]=r
targets=sorted(per_target['original8']);rng=np.random.default_rng(2026);comparison={}
for key in ['selected_dice','oracle_dice']:
 delta=np.array([per_target['automatic8'][t][key]-per_target['original8'][t][key] for t in targets]);boot=delta[rng.integers(0,100,size=(10000,100))].mean(1)
 comparison[key]=dict(mean_delta=float(delta.mean()),paired_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()))
out=dict(results=results,comparison=comparison,protocol='val100 development comparison; frozen original router fitted on same val100; no test evaluation',verified_prediction_masks=1400)
save(R/'results.json',out)
report='''# 自动8张参考图 TP b0–b6 验证结果

仅更换参考图。原SAM3-base256检索/传播设置、beam32、7候选、独立Router保持不变；700张新预测与Router选择冻结后才读取val GT。所有1400个新旧mask哈希与指标核验。

|参考图|Val Router Dice|Val Oracle Dice|Oracle差距|
|---|---:|---:|---:|
'''
for name,r in results.items():report+=f"|{name}|{r['router_dice']:.6f}|{r['oracle_dice']:.6f}|{r['oracle_gap']:.6f}|\n"
report+='\n|桥接数|原8张|自动8张|\n|---|---:|---:|\n'
for b in range(7):report+=f"|b{b}|{results['original8']['fixed_bridge'][str(b)]:.6f}|{results['automatic8']['fixed_bridge'][str(b)]:.6f}|\n"
report+='\n原Router曾在这100张val上拟合，本轮不重拟合，结果仅为开发集描述性对照。配对bootstrap仅反映目标抽样，不代表选图或训练种子稳定性。新旧候选源头分布、选中源头分布和逐图差值见results.json与各方法per_target_metrics.json。无test、学生训练或A0修改。\n'
(R/'report.md').write_text(report,encoding='utf-8')
print(json.dumps(out),flush=True)
