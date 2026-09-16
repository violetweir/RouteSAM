"""Frozen automatic-anchor TP b0-b6 test on Kvasir and ISIC2018, GPU1."""
from pathlib import Path
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'new_project/experiments/automatic_anchor_tp_test_20260913'
PY='/home/violet/anaconda3/envs/sam3/bin/python'
MODE='sam3enc_anchor_conditioned_target_pooling'
BASE=Path('/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt')
SPECS={
 'kvasir':dict(source='auto8_tp_validation_20260911',protocol='work/kvasir_1pct_anchors/protocol',ntrain=800,nval=100,ntest=100,k=8,old_test='work/rerun_kvasir_sam3base_test_20260906/quality_root/sam3enc_anchor_conditioned_target_pooling/propagation_quality_test/propagation_quality.jsonl'),
 'isic2018':dict(source='isic2018_auto21_tp_validation_20260911',protocol='work/isic18_pseudovideo_full/protocol',ntrain=2075,nval=259,ntest=260,k=21,old_test='work/isic18_sam3knn_s256_base/stage1_feature_knn_b0_b6/sam3enc_anchor_conditioned_target_pooling/propagation_quality_test/propagation_quality.jsonl')}

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def load(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def jsonl(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def resolve(p):
 p=Path(p);return p if p.is_absolute() else P/p
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def clean(r):return {k:v for k,v in r.items() if not k.startswith('gt_') and 'evaluation_only' not in k}
def bitmap(p):return np.asarray(Image.open(resolve(p)).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
def metric(a,b):
 i=int((a&b).sum());n=int(a.sum())+int(b.sum());return (2*i/n if n else 1., i/(n-i) if n-i else 1.)

def prepare():
 R.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,R/'pipeline.py')
 basehash=sha(BASE);assert basehash=='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
 for name,s in SPECS.items():
  d=R/name;d.mkdir();(d/'code').mkdir();(d/'protocol').mkdir();(d/'baseline').mkdir();(d/'logs').mkdir()
  source=P/'new_project/experiments'/s['source'];assert (source/'COMPLETE').exists()
  full=read(P/s['protocol']/'merged_manifest.jsonl')
  previous=read(source/'protocol/merged_manifest.jsonl');support=read(source/'protocol/support_manifest.jsonl')
  byid={r['merged_id']:r for r in full};test=[r for r in full if r['split']=='test']
  assert collections.Counter(r['split'] for r in full)==dict(train=s['ntrain'],validation=s['nval'],test=s['ntest'])
  assert len(previous)==s['ntrain']+s['nval'] and all(r['split']!='test' for r in previous)
  assert len(support)==s['k'] and len({r['merged_id'] for r in support})==s['k']
  for r in previous:
   original=byid[r['merged_id']];assert (r['split'],r['image_path'],r['mask_path'])==(original['split'],original['image_path'],original['mask_path'])
  for r in support:assert r['split']=='train' and r['merged_id'] in byid
  # Preserve existing cache row order exactly, append only previously unseen test images.
  records=previous+test;assert len({r['merged_id'] for r in records})==len(full)
  jsonl(d/'protocol/merged_manifest.jsonl',records);jsonl(d/'protocol/support_manifest.jsonl',support)
  jsonl(d/'protocol/feature_extraction_records.jsonl',support+test)
  (d/'protocol/frozen_labeled_images.txt').write_text('\n'.join(r['image_path'] for r in support)+'\n')
  inputs={str(BASE):basehash}
  for f in [source/'protocol/support_manifest.jsonl',source/'protocol/merged_manifest.jsonl',source/'quality_root/features/sam3_base_s256_features.npz',P/s['protocol']/'merged_manifest.jsonl']:
   inputs[str(f)]=sha(f)
  for f in support:
   for field in ['image_path','frozen_mask_path']:inputs[f[field]]=sha(f[field])
  for fn in ['stage1_feature_knn_routes.py','eval_route_propagation_quality.py','run_t21_dynamic_pseudovideo.py']:
   shutil.copy2(source/'code'/fn,d/'code'/fn);inputs[str(source/'code'/fn)]=sha(source/'code'/fn)
  baseline=P/s['old_test'];assert baseline.exists();inputs[str(baseline)]=sha(baseline)
  shutil.copy2(baseline,d/'baseline/original_test_quality.jsonl')
  policy=dict(dataset=name,source=str(source),selected_ids=[r['merged_id'] for r in support],k=s['k'],split_counts={k:s[k] for k in ['ntrain','nval','ntest']},base=str(BASE),base_sha256=basehash,gpu=1,selection_feature_size=1008,knn_feature_size=256,canvas=256,beam_width=32,knn='patch_mean',bridges=list(range(7)),primary='all seven fixed bridge test metrics',router='disabled per user request',no_student=True,no_lora=True,test_gt_for_inference=False,feature_cache_extension='retain all previous train/validation entries exactly; extract support+test for prototype check then append only test columns')
  save(d/'policy.json',policy);save(d/'input_sha256.json',inputs);save(d/'SUPPORT_FROZEN.json',dict(time=time.time(),support_sha256=sha(d/'protocol/support_manifest.jsonl')))
  save(d/'status.json',dict(stage='prepared',test_targets=s['ntest'],routes=s['ntest']*7))
 save(R/'status.json',dict(stage='prepared',gpu=1,datasets=list(SPECS)))
 (R/'PREPARED').touch()

def features(name):
 s=SPECS[name];d=R/name;source=Path(load(d/'policy.json')['source'])
 extractor=module('route_extractor_'+name,d/'code/stage1_feature_knn_routes.py')
 support=read(d/'protocol/support_manifest.jsonl');records=read(d/'protocol/feature_extraction_records.jsonl')
 tmp=d/'test_feature_extraction';tmp.mkdir(exist_ok=False)
 extractor.extract_sam3_encoder_features(records,support,tmp,feature_size=256)
 old=np.load(source/'quality_root/features/sam3_base_s256_features.npz');new=np.load(tmp/'sam3_base_s256_features.npz')
 assert np.array_equal(old['anchor_ids'],new['anchor_ids'])
 proto_delta=float(np.max(np.abs(old['anchor_prototypes']-new['anchor_prototypes'])))
 assert proto_delta<1e-6,proto_delta
 k=s['k'];out={}
 for key in old.files:
  if key=='patch_mean':out[key]=np.concatenate([old[key],new[key][k:]],axis=0)
  elif key in ['pooled','cond_target','cond_correspondence']:out[key]=np.concatenate([old[key],new[key][:,k:]],axis=1)
  else:out[key]=old[key]
 f=d/'quality_root/features/sam3_base_s256_features.npz';f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**out)
 n=s['ntrain']+s['nval'];assert out['patch_mean'].shape[0]==n+s['ntest']
 assert np.array_equal(out['patch_mean'][:n],old['patch_mean'])
 for key in ['pooled','cond_target','cond_correspondence']:assert np.array_equal(out[key][:,:n],old[key])
 save(d/'FEATURES_FROZEN.json',dict(previous_records=n,appended_test=s['ntest'],all_previous_entries_exact=True,anchor_prototype_max_abs_delta=proto_delta,sha256=sha(f)))

def run_stage(name,stage,cmd):
 d=R/name;save(R/'status.json',dict(stage=stage,dataset=name,gpu=1));save(d/'status.json',dict(stage=stage,gpu=1));print(name,stage,flush=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONPATH='/Data_8TB/lht/sam3:'+str(P/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
 with (d/'logs'/f'{stage}.log').open('x') as log:
  child=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  save(d/f'{stage}_process.json',dict(pid=child.pid,gpu=1,command=cmd));rc=child.wait()
 if rc:raise RuntimeError(f'{name}/{stage} exited {rc}')

def summarize(name):
 d=R/name;s=SPECS[name]
 new=read(d/f'quality_root/{MODE}/propagation_quality_test/propagation_quality.jsonl')
 old=read(d/'baseline/original_test_quality.jsonl');sets={'automatic':new,'original':old}
 test_ids={r['merged_id'] for r in read(d/'protocol/merged_manifest.jsonl') if r['split']=='test'}
 for variant,rows in sets.items():
  assert len(rows)==s['ntest']*7 and all(r['status']=='success' for r in rows)
  assert collections.Counter((r['target_id'],r['bridge_count']) for r in rows)==collections.Counter({(t,b):1 for t in test_ids for b in range(7)})
  if variant=='automatic':assert all(not any(k.startswith('gt_') for k in r) for r in rows)
  for r in rows:assert sha(resolve(r['forward_mask_path']))==r['forward_mask_sha256']
 save(d/'PREDICTIONS_FROZEN.json',dict(time=time.time(),router=False,prediction_hashes={v:[(r['route_id'],r['forward_mask_sha256']) for r in rows] for v,rows in sets.items()}))
 # Open test GT pixels only after all candidate predictions are frozen.
 results={};per={};gt_hashes={}
 for variant,rows in sets.items():
  dest=d/variant;dest.mkdir(exist_ok=False)
  for r in rows:
   gp=resolve(r['target_mask_path_evaluation_only']);gh=sha(gp)
   assert r['target_id'] not in gt_hashes or gh==gt_hashes[r['target_id']]
   gt_hashes[r['target_id']]=gh
   dice,iou=metric(bitmap(r['forward_mask_path']),bitmap(gp))
   if 'gt_dice_evaluation_only' in r:assert abs(dice-r['gt_dice_evaluation_only'])<1e-6
   r['dice']=dice;r['iou']=iou
  grouped=collections.defaultdict(list)
  for r in rows:grouped[r['target_id']].append(r)
  per[variant]={t:max(r['dice'] for r in rs) for t,rs in grouped.items()}
  save(dest/'per_candidate_metrics.json',[{k:r[k] for k in ['target_id','route_id','anchor_id','bridge_count','dice','iou','q_cycle','forward_mask_path']} for r in rows])
  save(dest/'per_target_oracle.json',per[variant])
  fixed={str(b):dict(n=s['ntest'],dice=float(np.mean([r['dice'] for r in rows if r['bridge_count']==b])),iou=float(np.mean([r['iou'] for r in rows if r['bridge_count']==b])),q_cycle=float(np.mean([r['q_cycle'] for r in rows if r['bridge_count']==b]))) for b in range(7)}
  results[variant]=dict(n=s['ntest'],fixed_bridge=fixed,oracle_dice=float(np.mean(list(per[variant].values()))),source_counts=dict(collections.Counter(r['anchor_id'] for r in rows)))
 rng=np.random.default_rng(2026);ids=sorted(test_ids)
 delta=np.asarray([per['automatic'][t]-per['original'][t] for t in ids])
 boot=delta[rng.integers(0,len(ids),(10000,len(ids)))].mean(1)
 comparison=dict(oracle_delta=float(delta.mean()),oracle_ci95=np.quantile(boot,[.025,.975]).tolist())
 save(d/'results.json',dict(dataset=name,results=results,comparison=comparison,policy=load(d/'policy.json')))
 lines=[f'# {name}：自动参考图＋SAM3-base KNN＋单TP b0–b6 Test','','固定此前自动选择名单；选择特征1008，检索/传播256，beam32。按照用户要求，本轮不拟合、不运行Router。','','|桥长|原参考图Dice|自动参考图Dice|差值|原IoU|自动IoU|','|---|---:|---:|---:|---:|---:|']
 for b in range(7):
  o=results['original']['fixed_bridge'][str(b)];a=results['automatic']['fixed_bridge'][str(b)]
  lines.append(f"|b{b}|{o['dice']:.6f}|{a['dice']:.6f}|{a['dice']-o['dice']:+.6f}|{o['iou']:.6f}|{a['iou']:.6f}|")
 for v in ['original','automatic']:lines+=['',f"{v} Oracle Dice：{results[v]['oracle_dice']:.6f}"]
 lines+=['','本次只新增自动参考集test传播；原参考图使用存档预测并按同GT重算。Oracle使用test GT事后计算，仅衡量候选上限。全部test参与，没有质量阈值删图。本轮不训练学生或LoRA。','',f'目录：`{d}`']
 (d/'report.md').write_text('\n'.join(lines)+'\n')
 for p,h in load(d/'input_sha256.json').items():assert sha(p)==h
 save(d/'completion_audit.json',dict(candidate_count=s['ntest']*7,original_and_new_masks_verified=s['ntest']*14,source_inputs_unchanged=True,test_gt_after_predictions=True,router_run=False,features=load(d/'FEATURES_FROZEN.json')))
 save(d/'status.json',dict(stage='complete'));(d/'COMPLETE').touch()

def pipeline():
 assert (R/'PREPARED').exists();(R/'STARTED').open('x').close()
 for name,s in SPECS.items():
  d=R/name
  run_stage(name,'test_features',[PY,str(R/'pipeline.py'),'features','--dataset',name])
  run_stage(name,'test_routes',[PY,str(d/'code/stage1_feature_knn_routes.py'),'--mode',MODE,'--feature-source','sam3_base','--feature-size','256','--knn-feature','patch_mean','--beam-width','32','--min-bridge','0','--max-bridge','6','--split','test','--protocol-root',str(d/'protocol'),'--output-root',str(d/'quality_root')])
  f=d/f'quality_root/{MODE}/test_pool0_stage1/routes.jsonl';routes=read(f);policy=load(d/'policy.json');records=read(d/'protocol/merged_manifest.jsonl')
  train={r['merged_id'] for r in records if r['split']=='train'};test={r['merged_id'] for r in records if r['split']=='test'}
  assert collections.Counter((r['target_id'],r['bridge_count']) for r in routes)==collections.Counter({(t,b):1 for t in test for b in range(7)})
  for row in routes:assert row['anchor_id'] in policy['selected_ids'] and set(row['bridge_ids'])<=train and not row['target_gt_used_for_search_or_inference']
  save(d/'ROUTES_FROZEN.json',dict(count=len(routes),sha256=sha(f),source_counts=dict(collections.Counter(r['anchor_id'] for r in routes))))
  run_stage(name,'test_propagation',[PY,str(d/'code/eval_route_propagation_quality.py'),'--checkpoint',str(BASE),'--mode',MODE,'--root',str(d/'quality_root'),'--split','test','--canvas','256','--no-target-gt'])
  run_stage(name,'summarize',[PY,str(R/'pipeline.py'),'summarize','--dataset',name])
 save(R/'results.json',{n:load(R/n/'results.json') for n in SPECS});save(R/'status.json',dict(stage='complete'));(R/'COMPLETE').touch()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','features','summarize']);ap.add_argument('--dataset',choices=list(SPECS));args=ap.parse_args()
 try:
  if args.action=='prepare':prepare()
  elif args.action=='run':pipeline()
  elif args.action=='features':features(args.dataset)
  else:summarize(args.dataset)
 except BaseException:
  if R.exists():
   (R/f'FAILED_{os.getpid()}.txt').write_text(traceback.format_exc())
   save(R/'status.json',dict(stage='failed',dataset=args.dataset,action=args.action,error=traceback.format_exc()))
  raise
