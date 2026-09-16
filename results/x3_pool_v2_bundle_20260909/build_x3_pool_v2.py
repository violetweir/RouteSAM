"""Frozen-rule, history-independent TP pseudo pool; training data only."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
BASE=P/'work/kvasir_tp_filterfirst_students_20260909'
POLICY={
 'rule_version':'x3_unified_quality_v2',
 'scope':'all 792 frozen unlabeled training targets; historical pool membership does not enter decisions',
 'q_return_min':.95,
 'A':{'q_multi':.90,'q_model_mean':.90,'q_model_min':.80,'q_model_var_max':.01},
 'B':{'q_multi':.75,'q_model_mean':.75,'q_model_min':.60,'q_model_var_max':.03},
 'candidate_selection':'highest eligible quality tier, then unchanged TP Router score; bridge/route ties unchanged',
 'nonempty':'SAM mask nonempty and >=2 of 4 student masks nonempty for either tier',
 'area_quantile_gate':False,
 'q_multi_definition':'candidate mean Dice against other 6 TP masks',
 'soft_target':'0.75 * selected TP binary + 0.25 * four-student probability mean',
 'pixel_weight':'clip(exp(-5*TP_variance)*exp(-5*student_variance)*(1-abs(TP_binary-student_mean)),.05,1)',
 'image_weight':'tier_factor * (q_multi + q_model_mean)/2; A=.75, B=.50; absolute, not pool-minmax normalized',
 'training':'one uniformly sampled pseudo stream, 6 GT + 6 pseudo; unified weighted soft loss',
 'test_policy':'no test data used to generate or tune this pool',
}
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def jsonl(p,rows):Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def binary(p):return np.asarray(Image.open(p).convert('L'))>127
def probability(p):
 a=np.asarray(Image.open(p));assert a.min()>=0
 return a.astype(np.float32)/(255. if a.dtype==np.uint8 else 65535.)
def dice(a,b):
 n=int(a.sum())+int(b.sum());return 2*int((a&b).sum())/n if n else 1.
def tier(row):
 if row['q_return']<.95 or not row['nonempty_safe']:return 'C'
 for name in ['A','B']:
  t=POLICY[name]
  if row['q_multi']>=t['q_multi'] and row['q_model_mean']>=t['q_model_mean'] and row['q_model_min']>=t['q_model_min'] and row['q_model_var']<=t['q_model_var_max']:return name
 return 'C'
def pick(rows):
 for name in ['A','B']:
  eligible=[x for x in rows if x['quality_tier']==name]
  if eligible:return max(eligible,key=lambda x:(x['router_score'],-x['bridge_count'],x['route_id']))
 return None
def generate(committee_root,out,preview=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);save(out/'policy.json',POLICY)
 support={x['merged_id'] for x in read(BASE/'protocol/support_manifest.jsonl')}
 train={x['merged_id'] for x in read(BASE/'protocol/merged_manifest.jsonl') if x['split']=='train'}
 targets=train-support;assert len(support)==8 and len(targets)==792
 router=module('pool_v2_router',BASE/'code/router.py');model=router.Ridge(**json.loads((BASE/'router.json').read_text()))
 quality_path=BASE/'quality/anchor_conditioned_target_pooling/propagation_quality_train/propagation_quality.jsonl'
 quality=[{k:v for k,v in x.items() if not k.startswith('gt_') and 'evaluation_only' not in k} for x in read(quality_path)]
 groups=collections.defaultdict(list)
 for x in quality:groups[x['target_id']].append(x)
 assert set(groups)==targets
 names=['S2_valbest','S2_final','S3_valbest','S3_final']
 pred={name:{x['merged_id']:x for x in read(Path(committee_root)/f'predictions/{name}/student_predictions_train.jsonl')} for name in names}
 frozen={name:{'manifest_sha256':sha(Path(committee_root)/f'predictions/{name}/student_predictions_train.jsonl'),'checkpoint_sha256':sha(next(iter(pred[name].values()))['checkpoint'])} for name in names}
 save(out/'inputs.json',dict(committee_root=str(committee_root),preview_only=preview,quality_sha256=sha(quality_path),router_sha256=sha(BASE/'router.json'),committee=frozen))
 accepted=[];decisions=[];scores=[]
 for index,target in enumerate(sorted(targets),1):
  rr=sorted(groups[target],key=lambda x:(x['bridge_count'],x['route_id']));assert len(rr)==7
  mm=np.stack([binary(x['forward_mask_path']) for x in rr]);assert mm.shape==(7,256,256)
  ss=np.stack([probability(pred[name][target]['student_probability_map']) for name in names]);assert ss.shape==(4,256,256)
  sb=np.stack([binary(pred[name][target]['student_binary_mask']) for name in names]);assert sb.shape==ss.shape
  local=[]
  for j,x in enumerate(rr):
   models=[dice(mm[j],v) for v in sb]
   row=dict(target_id=target,route_id=x['route_id'],bridge_count=x['bridge_count'],q_return=float(x['q_cycle']),q_multi=float(np.mean([dice(mm[j],v) for i,v in enumerate(mm) if i!=j])),q_model_mean=float(np.mean(models)),q_model_min=float(np.min(models)),q_model_var=float(np.var(models)),router_score=model.score(x),nonempty_safe=bool(mm[j].any() and sum(bool(v.any()) for v in sb)>=2),source_mask_path=x['forward_mask_path'])
   row['quality_tier']=tier(row);local.append(row)
  scores.extend(local);chosen=pick(local)
  if chosen is None:
   decisions.append(dict(target_id=target,quality_tier='C',accepted=False,reason='no candidate satisfies unified A/B requirements'));continue
  j=next(i for i,x in enumerate(rr) if x['route_id']==chosen['route_id']);sam=mm[j].astype(np.float32);mean=ss.mean(0)
  soft=.75*sam+.25*mean
  weight=np.clip(np.exp(-5*mm.astype(np.float32).var(0))*np.exp(-5*ss.var(0))*(1-np.abs(sam-mean)),.05,1)
  stem=target.replace('::','__').replace('/','_');d=out/'maps';d.mkdir(exist_ok=True)
  mask_path=d/(stem+'_mask.png');soft_path=d/(stem+'_soft.png');weight_path=d/(stem+'_weight.png')
  Image.fromarray((mm[j]*255).astype(np.uint8)).save(mask_path)
  Image.fromarray(np.rint(soft*65535).astype(np.uint16)).save(soft_path)
  Image.fromarray(np.rint(weight*65535).astype(np.uint16)).save(weight_path)
  assert np.array_equal(binary(mask_path),probability(soft_path)>=.5)
  assert np.max(np.abs(probability(soft_path)-soft))<=1/65535
  factor=.75 if chosen['quality_tier']=='A' else .50
  row={**chosen,'sample_type':'pseudo','pseudo_mask_path':str(mask_path),'pseudo_consensus_path':str(soft_path),'pixel_weight_path':str(weight_path),'explicit_quality_weight':factor*(chosen['q_multi']+chosen['q_model_mean'])/2,'target_gt_used':False,'preview_only':preview,'mask_sha256':sha(mask_path),'soft_sha256':sha(soft_path),'pixel_weight_sha256':sha(weight_path)}
  accepted.append(row);decisions.append({**chosen,'accepted':True})
  if index%100==0:print(f'{index}/792 audited; accepted={len(accepted)}',flush=True)
 jsonl(out/'pseudo_manifest_x3.jsonl',accepted);jsonl(out/'decisions.jsonl',decisions);jsonl(out/'candidate_scores.jsonl',scores)
 assert len(scores)==5544 and len(decisions)==792 and len({x['target_id'] for x in accepted})==len(accepted) and not support&{x['target_id'] for x in accepted}
 # Historical membership is loaded only after all decisions and maps are frozen.
 previous={x['target_id']:x for x in read(BASE/'pseudo_manifest_x3.jsonl')}
 transitions=collections.Counter((previous.get(x['target_id'],{}).get('sample_type','excluded'),x['quality_tier']) for x in decisions)
 summary=dict(audited=792,candidates=5544,accepted=len(accepted),tiers=dict(collections.Counter(x['quality_tier'] for x in decisions)),preview_only=preview,previous_x3_count=len(previous),overlap_with_previous=len(set(previous)&{x['target_id'] for x in accepted}),transitions=[dict(old_group=k[0],new_tier=k[1],count=v) for k,v in sorted(transitions.items())],sampling='6 GT + 6 uniformly sampled accepted pseudo; no historical group quotas',all_targets_train_only=True,soft_mask_binary_equals_selected_TP=True)
 save(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
 (out/'COMPLETE').write_text('complete\n')
 return summary
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--committee-root',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--preview',action='store_true');args=parser.parse_args();generate(args.committee_root,args.output_root,args.preview)
