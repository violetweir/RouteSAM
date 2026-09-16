"""Freeze inputs and generate paired hard/soft manifests without target train GT."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
SRC=P/'work/kvasir_tp_filterfirst_students_20260909'
R=P/'new_project/experiments/single_student_hard_soft_20260909'
DATA=P/'work/kvasir_1pct_anchors/baseline_data'
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonl(p,rows):Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
def resolve(value,split):
 p=Path(value);return p if p.is_absolute() else DATA/split/p
def setup(bundle):
 assert not R.exists(),R
 R.mkdir(parents=True);(R/'logs').mkdir();(R/'runs').mkdir();(R/'data').mkdir();(R/'code').mkdir()
 for name in ['train_student.py','prepare_experiment.py','test_experiment.py','run_experiment.py']:shutil.copy2(bundle/name,R/'code'/name)
 shutil.copytree(P/'third_party/SC-SAM/Model',R/'code/Model',ignore=shutil.ignore_patterns('__pycache__'))
 shutil.copy2(P/'third_party/SC-SAM/dataloader/transforms.py',R/'code/pinned_transforms.py')
 shutil.copytree(SRC/'protocol',R/'data/protocol');shutil.copy2(SRC/'router.json',R/'data/router.json')
 for name in ['images','hard','soft']:(R/'data'/name).mkdir()
 protocol=read(SRC/'protocol/merged_manifest.jsonl');support={x['merged_id'] for x in read(SRC/'protocol/support_manifest.jsonl')}
 all_train={x['merged_id'] for x in protocol if x['split']=='train'};assert len(support)==8 and len(all_train-support)==792
 original=read(SRC/'pseudo_manifest_original.jsonl');assert len(original)==580
 meta={x['merged_id']:x for x in read(DATA/'train/metadata.jsonl')};assert set(meta)==all_train
 groups=collections.defaultdict(list)
 quality_path=SRC/'quality/anchor_conditioned_target_pooling/propagation_quality_train/propagation_quality.jsonl'
 for x in read(quality_path):groups[x['target_id']].append({k:v for k,v in x.items() if not k.startswith('gt_') and 'evaluation_only' not in k})
 spec=importlib.util.spec_from_file_location('frozen_router',SRC/'code/router.py');router=importlib.util.module_from_spec(spec);spec.loader.exec_module(router)
 model=router.Ridge(**json.loads((SRC/'router.json').read_text()))
 source_by={x['target_id']:x for x in original};assert len(source_by)==580 and not set(source_by)&support
 rows=[];audit=[];input_hashes={str(quality_path):sha(quality_path),str(SRC/'pseudo_manifest_original.jsonl'):sha(SRC/'pseudo_manifest_original.jsonl'),str(SRC/'router.json'):sha(SRC/'router.json')}
 for target in sorted(set(source_by)|support):
  is_gt=target in support;stem=target.replace('::','__');image=R/'data/images'/(stem+'.png');hard=R/'data/hard'/(stem+'.png')
  source_image=resolve(meta[target]['file_name'],'train');Image.open(source_image).convert('RGB').resize((256,256),Image.Resampling.NEAREST).save(image)
  if is_gt:
   source_gt=resolve(meta[target]['mask_file_name'],'train');m=np.asarray(Image.open(source_gt).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
   Image.fromarray((m*255).astype(np.uint8)).save(hard);soft=hard;input_hashes[str(source_gt)]=sha(source_gt)
  else:
   previous=source_by[target];assert previous['q_multi']>=.90
   candidates=groups[target];assert len(candidates)==7
   eligible=[x for x in candidates if x['q_cycle']>=.95];assert eligible
   selected=max(eligible,key=lambda x:(model.score(x),-x['bridge_count'],x['route_id']))
   assert selected['route_id']==previous['route_id']
   assert sha(selected['forward_mask_path'])==sha(previous['pseudo_mask_path'])==selected['forward_mask_sha256']
   mm=[]
   for x in eligible:
    path=Path(x['forward_mask_path']);assert sha(path)==x['forward_mask_sha256'];input_hashes[str(path)]=sha(path)
    a=np.asarray(Image.open(path).convert('L'))>127;assert a.shape==(256,256);mm.append(a)
   m=np.asarray(Image.open(selected['forward_mask_path']).convert('L'))>127;p=np.stack(mm).mean(0,dtype=np.float32);y=.75*m.astype(np.float32)+.25*p
   assert np.array_equal(y>=.5,m)
   shutil.copy2(selected['forward_mask_path'],hard);soft=R/'data/soft'/(stem+'.png');Image.fromarray(np.rint(y*65535).astype(np.uint16)).save(soft)
   decoded=np.asarray(Image.open(soft)).astype(np.float32)/65535
   assert np.max(np.abs(decoded-y))<=1/65535 and np.array_equal(decoded>=.5,m)
   audit.append(dict(target_id=target,route_id=selected['route_id'],eligible_route_ids=[x['route_id'] for x in eligible],eligible_count=len(eligible),main_mask_sha256=sha(hard),soft_sha256=sha(soft),soft_fraction=float(((decoded>0)&(decoded<1)).mean()),single_candidate_equals_hard=bool(np.array_equal(decoded,m)) if len(eligible)==1 else None))
  rows.append(dict(target_id=target,is_gt=is_gt,image_path=str(image),hard_label_path=str(hard),soft_label_path=str(soft),image_sha256=sha(image),hard_sha256=sha(hard),soft_sha256=sha(soft),sample_weight=1.,pixel_weight=1.))
 assert len(rows)==588 and sum(x['is_gt'] for x in rows)==8
 jsonl(R/'data/train_manifest.jsonl',rows);jsonl(R/'data/label_audit.jsonl',audit)
 for split in ['validation','test']:
  eval_rows=[]
  metadata=read(DATA/split/'metadata.jsonl');assert len(metadata)==100
  assert {x['merged_id'] for x in metadata}=={x['merged_id'] for x in protocol if x['split']==split}
  for x in sorted(metadata,key=lambda x:x['merged_id']):
   eval_rows.append(dict(target_id=x['merged_id'],is_gt=True,image_path=str(resolve(x['file_name'],split)),hard_label_path=str(resolve(x['mask_file_name'],split))))
  jsonl(R/f'data/{split}_manifest.jsonl',eval_rows)
 # Test labels are not opened here; evaluation is deferred until both models freeze.
 summary=dict(train_count=588,gt_count=8,pseudo_count=580,eligible_counts=dict(collections.Counter(x['eligible_count'] for x in audit)),softened_image_count=sum(x['soft_fraction']>0 for x in audit),mean_soft_pixel_fraction=float(np.mean([x['soft_fraction'] for x in audit])),all_main_masks_unchanged=True,all_soft_binary_equal_main=True,all_selected_routes_recomputed=True,train_gt_masks_read=8,unlabeled_gt_masks_read=0,test_masks_read=0)
 save(R/'data/summary.json',summary);save(R/'data/input_sha256.json',input_hashes)
 sys.path.insert(0,str(R/'code'));import torch
 from train_student import SamUnet,seed_all,tensor_hash
 seed_all(2026);initial=SamUnet(argparse.Namespace(in_channels=3,num_classes=2));torch.save(initial.state_dict(),R/'initial_model.pth')
 cfg=dict(seed=2026,epochs=816,batch_size=12,steps_per_epoch=49,total_iterations=39984,train_count=588,gt_count=8,pseudo_count=580,optimizer='SGD',initial_lr=.01,momentum=.9,weight_decay=1e-4,lr_schedule='lr(epoch)=.01*(1-(epoch-1)/816); constant within each epoch',validation_interval_epochs=4,checkpoint_selection='maximum full validation mean per-image Dice; tie keeps earlier epoch',loss='per-image BCEWithLogits(fg_logit-bg_logit,Y) + foreground SoftDice(smooth=1), then mean over batch',sample_weights='all 1',pixel_weights='all 1',pseudo_ramp=False,gt_resampling=False,augmentation='pinned previous weak GT / strong pseudo; per-image epoch seed, same between variants',image_resize='PIL nearest 256x256 before augmentation; same for validation/test',soft_recipe='Y=.75*Router_main+.25*mean(return>=.95 TP candidates)',initial_tensor_sha256=tensor_hash(initial.state_dict()),initial_checkpoint_sha256=sha(R/'initial_model.pth'),gpu=1,gpu_allocator_fraction_per_run=.20,test_policy='freeze both validation-best checkpoints after full training, then evaluate each once; no test tuning',python=torch.__version__)
 cfg['code_sha256']={str(f.relative_to(R)):sha(f) for f in (R/'code').rglob('*.py')};save(R/'config.json',cfg)
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
 (R/'PREPARED').write_text('prepared\n')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);args=parser.parse_args();setup(args.bundle)
