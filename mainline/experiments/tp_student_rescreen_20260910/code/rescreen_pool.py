"""Uniform all-792 TP/student re-screen; frozen rules and teacher."""
import argparse
import collections
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
import torch
from train_student import read,save,sha,SamUnet

PROJECT=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
BASE=PROJECT/'new_project/experiments/single_student_hard_soft_20260909'
TP=PROJECT/'work/kvasir_tp_filterfirst_students_20260909'
DATA=PROJECT/'work/kvasir_1pct_anchors/baseline_data'
POLICY={
 'targets':'all original 792 unlabeled training images; no automatic retention of old 580',
 'teacher':'stage-1 soft validation-best epoch 556, frozen; not test-selected final',
 'A':{'return_min':.95,'tp_candidate_mean_min':.95,'student_required':False},
 'B':{'return_min':.95,'tp_candidate_mean_min':.80,'student_dice_min':.85,'student_nonempty':True},
 'candidate_nonempty_required':True,
 'tp_consistency':'mean binary Dice of this candidate against the other six TP masks',
 'student_consistency':'binary Dice against frozen student probability >=.5',
 'selection':'prefer A if any, else B if any; within chosen tier rank by frozen TP Router, then -bridge, route_id',
 'label':'Y=.75*M+.25*mean(all TP candidates with return>=.95); teacher not mixed into target',
 'weights':'all images and pixels 1; A/B are admission categories, not training weights or sampling streams',
 'training':'same saved initial U-Net weights; uniform shuffle; batch12;816epochs; keep partial final batch',
 'calibration':'proposal thresholds frozen before any validation diagnostics; no threshold search or test tuning',
}
def jsonl(p,rows):Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
def binary(p):return np.asarray(Image.open(p).convert('L'))>127
def dice(a,b):
 n=int(a.sum())+int(b.sum());return 2*int((a&b).sum())/n if n else 1.
def tier(row):
 if not row['candidate_nonempty'] or row['q_return']<.95:return 'C'
 if row['q_tp']>=.95:return 'A'
 if row['q_tp']>=.80 and row['student_nonempty'] and row['q_student']>=.85:return 'B'
 return 'C'
def select(rows):
 for name in ['A','B']:
  eligible=[x for x in rows if x['tier']==name]
  if eligible:return max(eligible,key=lambda x:(x['router_score'],-x['bridge_count'],x['route_id']))
 return None
def image_path(row,split):
 p=Path(row['file_name']);return p if p.is_absolute() else DATA/split/p
@torch.inference_mode()
def infer_teacher(root,model,metadata,split):
 dest=root/'teacher_predictions'/split;dest.mkdir(parents=True,exist_ok=False);results={}
 for row in sorted(metadata,key=lambda x:x['merged_id']):
  target=row['merged_id'];path=image_path(row,split)
  a=np.asarray(Image.open(path).convert('RGB').resize((256,256),Image.Resampling.NEAREST),dtype=np.float32)/255
  a=(a-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
  x=torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).unsqueeze(0).cuda();_,p=model(x);p=p[0,1].cpu().numpy();mask=p>=.5
  stem=target.replace('::','__');prob_path=dest/(stem+'_prob.png');mask_path=dest/(stem+'_mask.png')
  Image.fromarray(np.rint(p*65535).astype(np.uint16)).save(prob_path);Image.fromarray((mask*255).astype(np.uint8)).save(mask_path)
  results[target]=dict(target_id=target,image_path=str(path),probability_path=str(prob_path),mask_path=str(mask_path),mask_sha256=sha(mask_path),probability_sha256=sha(prob_path))
 jsonl(dest/'manifest.jsonl',list(results.values()));return results
def audit_split(root,router,students,split):
 path=TP/f'quality/anchor_conditioned_target_pooling/propagation_quality_{split}/propagation_quality.jsonl'
 groups=collections.defaultdict(list)
 for row in read(path):
  clean={k:v for k,v in row.items() if not k.startswith('gt_') and 'evaluation_only' not in k};groups[clean['target_id']].append(clean)
 assert set(groups)==set(students)
 decisions=[];scores=[]
 for index,target in enumerate(sorted(students),1):
  candidates=sorted(groups[target],key=lambda x:(x['bridge_count'],x['route_id']));assert len(candidates)==7
  mm=[binary(x['forward_mask_path']) for x in candidates];student=binary(students[target]['mask_path']);local=[]
  assert all(x.shape==(256,256) for x in mm) and student.shape==(256,256)
  for i,c in enumerate(candidates):
   assert sha(c['forward_mask_path'])==c['forward_mask_sha256']
   row=dict(target_id=target,route_id=c['route_id'],bridge_count=c['bridge_count'],q_return=float(c['q_cycle']),q_tp=float(np.mean([dice(mm[i],m) for j,m in enumerate(mm) if j!=i])),q_student=dice(mm[i],student),candidate_nonempty=bool(mm[i].any()),student_nonempty=bool(student.any()),router_score=router.score(c),source_mask_path=c['forward_mask_path'],source_mask_sha256=c['forward_mask_sha256'])
   row['tier']=tier(row);local.append(row)
  scores.extend(local);chosen=select(local)
  if chosen is None:decisions.append(dict(target_id=target,accepted=False,tier='C',reason='no nonempty candidate satisfies A or B'))
  else:
   consensus=[c for c in candidates if c['q_cycle']>=.95];assert chosen['route_id'] in {c['route_id'] for c in consensus}
   decisions.append({**chosen,'accepted':True,'consensus_sources':[dict(route_id=c['route_id'],path=c['forward_mask_path'],sha256=c['forward_mask_sha256']) for c in consensus]})
  if index%100==0:print(f'{split}: {index}/{len(students)} screened',flush=True)
 dest=root/'screening'/split;dest.mkdir(parents=True,exist_ok=False);jsonl(dest/'candidate_scores.jsonl',scores);jsonl(dest/'decisions.jsonl',decisions)
 save(dest/'input.json',dict(quality_path=str(path),quality_sha256=sha(path)))
 return decisions
def build(root):
 assert (root/'SETUP_COMPLETE').exists() and not (root/'POOL_COMPLETE').exists()
 save(root/'status.json',dict(stage='teacher_inference_and_screening'));save(root/'policy.json',POLICY)
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 frozen=json.loads((root/'teacher.json').read_text());assert sha(root/'teacher.pth')==frozen['sha256']
 model=SamUnet(argparse.Namespace(in_channels=3,num_classes=2));model.load_state_dict(torch.load(root/'teacher.pth',map_location='cpu'));model=model.cuda().eval()
 support={x['merged_id'] for x in read(TP/'protocol/support_manifest.jsonl')};protocol=read(TP/'protocol/merged_manifest.jsonl')
 unlabeled={x['merged_id'] for x in protocol if x['split']=='train'}-support;assert len(support)==8 and len(unlabeled)==792
 metadata=[x for x in read(DATA/'train/metadata.jsonl') if x['merged_id'] in unlabeled];assert len(metadata)==792
 students=infer_teacher(root,model,metadata,'train')
 spec=importlib.util.spec_from_file_location('frozen_tp_router',TP/'code/router.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);router=m.Ridge(**json.loads((TP/'router.json').read_text()))
 decisions=audit_split(root,router,students,'train');assert len(decisions)==792
 # Freeze all train decisions before opening any validation GT or old pool membership.
 save(root/'TRAIN_DECISIONS_FROZEN.json',dict(decisions_sha256=sha(root/'screening/train/decisions.jsonl'),policy_sha256=sha(root/'policy.json'),teacher_sha256=frozen['sha256']))
 data=root/'data'
 for name in ['images','hard','soft']:(data/name).mkdir(parents=True,exist_ok=False)
 rows=[];label_audit=[]
 for d in decisions:
  if not d['accepted']:continue
  target=d['target_id'];stem=target.replace('::','__');im=data/'images'/(stem+'.png');hard=data/'hard'/(stem+'.png');soft=data/'soft'/(stem+'.png')
  Image.open(students[target]['image_path']).convert('RGB').resize((256,256),Image.Resampling.NEAREST).save(im);shutil.copy2(d['source_mask_path'],hard)
  mm=[binary(x['path']) for x in d['consensus_sources']];M=binary(hard).astype(np.float32);P=np.stack(mm).mean(0,dtype=np.float32);Y=.75*M+.25*P
  Image.fromarray(np.rint(Y*65535).astype(np.uint16)).save(soft);decoded=np.asarray(Image.open(soft)).astype(np.float32)/65535
  assert np.array_equal(decoded>=.5,M>=.5) and np.max(np.abs(decoded-Y))<=1/65535
  rows.append(dict(target_id=target,is_gt=False,image_path=str(im),hard_label_path=str(hard),soft_label_path=str(soft),image_sha256=sha(im),hard_sha256=sha(hard),soft_sha256=sha(soft),sample_weight=1.,pixel_weight=1.,admission_tier=d['tier']))
  label_audit.append(dict(target_id=target,selected_route=d['route_id'],consensus_count=len(mm),soft_pixel_fraction=float(((decoded>0)&(decoded<1)).mean())))
 previous=read(BASE/'data/train_manifest.jsonl');previous_by={x['target_id']:x for x in previous if not x['is_gt']}
 for row in previous:
  if not row['is_gt']:continue
  assert row['target_id'] in support
  stem=row['target_id'].replace('::','__');im=data/'images'/(stem+'.png');hard=data/'hard'/(stem+'.png');shutil.copy2(row['image_path'],im);shutil.copy2(row['hard_label_path'],hard)
  rows.append(dict(target_id=row['target_id'],is_gt=True,image_path=str(im),hard_label_path=str(hard),soft_label_path=str(hard),image_sha256=sha(im),hard_sha256=sha(hard),soft_sha256=sha(hard),sample_weight=1.,pixel_weight=1.))
 rows.sort(key=lambda x:x['target_id']);assert len(rows)==len({x['target_id'] for x in rows}) and sum(x['is_gt'] for x in rows)==8
 jsonl(data/'train_manifest.jsonl',rows);jsonl(data/'label_audit.jsonl',label_audit)
 for split in ['validation','test']:shutil.copy2(BASE/f'data/{split}_manifest.jsonl',data/f'{split}_manifest.jsonl')
 new={x['target_id'] for x in rows if not x['is_gt']};old=set(previous_by);retained=old&new;by={x['target_id']:x for x in rows}
 summary=dict(audited=792,candidates=5544,tiers=dict(collections.Counter(x['tier'] for x in decisions)),pseudo_count=len(new),gt_count=8,train_count=len(rows),previous_pseudo_count=580,retained=len(retained),removed=len(old-new),added=len(new-old),retained_main_mask_changed=sum(by[k]['hard_sha256']!=previous_by[k]['hard_sha256'] for k in retained),teacher_sha256=frozen['sha256'],all_792_rescreened=True,no_historical_membership_in_scoring=True,student_not_mixed_into_targets=True,unlabeled_gt_masks_read=0)
 save(data/'pool_summary.json',summary)
 jsonl(data/'membership_changes.jsonl',[dict(target_id=k,old_member=k in old,new_member=k in new) for k in sorted(old|new)])
 cfg=json.loads((root/'config.json').read_text());cfg.update(train_count=len(rows),pseudo_count=len(new),steps_per_epoch=math.ceil(len(rows)/12),total_iterations=816*math.ceil(len(rows)/12));save(root/'config.json',cfg)
 # Descriptive validation screen audit, using the frozen rules without retuning.
 valmeta=read(DATA/'validation/metadata.jsonl');valstudents=infer_teacher(root,model,valmeta,'validation');valdecisions=audit_split(root,router,valstudents,'validation');valindex={x['merged_id']:x for x in valmeta};valmetrics=[];teacher_dice=[]
 for d in valdecisions:
  x=valindex[d['target_id']];gtpath=Path(x['mask_file_name']);gtpath=gtpath if gtpath.is_absolute() else DATA/'validation'/gtpath
  gt=np.asarray(Image.open(gtpath).convert('L').resize((256,256),Image.Resampling.NEAREST))>127;teacher_dice.append(dice(binary(valstudents[d['target_id']]['mask_path']),gt))
  if d['accepted']:valmetrics.append(dict(target_id=d['target_id'],tier=d['tier'],selected_dice=dice(binary(d['source_mask_path']),gt)))
 teacher_mean=float(np.mean(teacher_dice));assert abs(teacher_mean-frozen['validation_dice'])<1e-10
 validation=dict(count=100,accepted=len(valmetrics),coverage=len(valmetrics)/100,accepted_subset_dice=float(np.mean([x['selected_dice'] for x in valmetrics])) if valmetrics else None,teacher_validation_dice=teacher_mean,teacher_checkpoint_reproduced=True,note='descriptive subset audit, not full-set segmentation Dice or OOF validation; thresholds not adjusted')
 jsonl(root/'screening/validation/accepted_metrics.jsonl',valmetrics);save(root/'screening/validation/summary.json',validation)
 assert sha(root/'teacher.pth')==frozen['sha256'];save(root/'status.json',dict(stage='pool_complete',pool=summary,validation_screen=validation));(root/'POOL_COMPLETE').write_text('complete\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(validation,ensure_ascii=False,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();build(args.root)
