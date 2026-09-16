"""Prepare the original 448-image subset under the new soft student recipe."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import argparse
import numpy as np
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
SOURCE=P/'new_project/experiments/single_student_hard_soft_20260909'
OLD=P/'work/kvasir_tp_student_mainline_20260907'
OUT=P/'new_project/experiments/tp448_single_student_soft_20260912'

def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def binary(p):return np.asarray(Image.open(p).convert('L'))>127

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 (OUT/'data').mkdir();(OUT/'logs').mkdir()
 shutil.copytree(SOURCE/'code',OUT/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copy2(SOURCE/'initial_model.pth',OUT/'initial_model.pth')
 shutil.copy2(__file__,OUT/'code/setup_tp448_soft.py')
 old={r['target_id']:r for r in read(OLD/'pseudo_manifest_original.jsonl')};assert len(old)==448
 source_rows=read(SOURCE/'data/train_manifest.jsonl')
 selected=[dict(r,augmentation_index=i) for i,r in enumerate(source_rows) if r['is_gt'] or r['target_id'] in old]
 assert len(selected)==456 and sum(r['is_gt'] for r in selected)==8
 assert {r['target_id'] for r in selected if not r['is_gt']}==set(old)
 q=read(OLD/'quality/anchor_conditioned_target_pooling/propagation_quality_train/propagation_quality.jsonl')
 groups={}
 for row in q:groups.setdefault(row['target_id'],[]).append(row)
 hashes={};label_audit=[]
 for row in selected:
  for field,key in [('image_path','image_sha256'),('hard_label_path','hard_sha256'),('soft_label_path','soft_sha256')]:
   if field in row:
    h=sha(row[field]);assert h==row[key];hashes[row[field]]=h
  if not row['is_gt']:
   m=binary(row['hard_label_path']);assert np.array_equal(m,binary(old[row['target_id']]['pseudo_mask_path']))
   candidates=[r for r in groups[row['target_id']] if r['q_cycle']>=.95]
   assert candidates and any(r['route_id']==old[row['target_id']]['route_id'] for r in candidates)
   consensus=np.mean([binary(r['forward_mask_path']).astype(np.float32) for r in candidates],axis=0)
   expected=.75*m+.25*consensus
   actual=np.asarray(Image.open(row['soft_label_path']),dtype=np.float32)/65535
   assert np.max(np.abs(actual-expected))<1/65535+1e-7
   assert np.array_equal(actual>=.5,m)
   label_audit.append(dict(target_id=row['target_id'],qualified_candidates=len(candidates),max_quantization_error=float(np.max(np.abs(actual-expected)))))
 (OUT/'data/train_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
 for name in ['validation_manifest.jsonl','test_manifest.jsonl']:
  shutil.copy2(SOURCE/'data'/name,OUT/'data'/name)
 train_ids={r['target_id'] for r in selected}
 val_ids={r['target_id'] for r in read(OUT/'data/validation_manifest.jsonl')}
 test_ids={r['target_id'] for r in read(OUT/'data/test_manifest.jsonl')}
 assert len(val_ids)==len(test_ids)==100 and not train_ids&val_ids and not train_ids&test_ids and not val_ids&test_ids
 cfg=json.loads((SOURCE/'config.json').read_text())
 cfg.update(train_count=456,pseudo_count=448,steps_per_epoch=38,total_iterations=31008,gpu=0,source_experiment=str(SOURCE),source_448_manifest=str(OLD/'pseudo_manifest_original.jsonl'),test_policy='After 816 epochs freeze validation-best and final, reproduce corresponding validation, then test both.',augmentation='Preserve original 580-pool row index for per-image augmentation seed; subset shuffled independently each epoch.')
 cfg.pop('code_sha256',None)
 trainer=OUT/'code/train_student.py';src=trainer.read_text()
 assert src.count('588')==3 and src.count('39984')==1
 src=src.replace('588','456').replace('39984','31008')
 assert src.count('index*97')==1
 src=src.replace('index*97',"row.get('augmentation_index',index)*97")
 trainer.write_text(src)
 cfg['code_sha256']={str(f.relative_to(OUT)):sha(f) for f in (OUT/'code').rglob('*.py')}
 save(OUT/'config.json',cfg);save(OUT/'data/input_sha256.json',hashes);save(OUT/'data/label_audit.json',label_audit)
 sys.path.insert(0,str(OUT/'code'))
 import torch
 import train_student as t
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 ds=t.StudentDataset(selected,'soft',True,2026);sampler=t.EpochSampler(len(selected),2026)
 assert sorted(sampler.order())==list(range(456))
 original=t.StudentDataset(source_rows,'soft',True,2026)
 for i in [0,50,200,455]:
  a=ds[(1,i)];b=original[(1,selected[i]['augmentation_index'])]
  assert torch.equal(a['image'],b['image']) and torch.equal(a['target'],b['target'])
 model=t.SamUnet(argparse.Namespace(in_channels=3,num_classes=2))
 model.load_state_dict(torch.load(OUT/'initial_model.pth',map_location='cpu'))
 assert t.tensor_hash(model.state_dict())==cfg['initial_tensor_sha256']
 model=model.cuda().train()
 batch=next(iter(t.loader(ds,12,sampler,workers=0)))
 logits,_=model(batch['image'].cuda());loss,_,_=t.equal_loss(logits,batch['target'].cuda())
 loss.backward()
 assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 save(OUT/'preflight.json',dict(status='PASS',gt=8,pseudo=448,total=456,epochs=816,steps_per_epoch=38,total_steps=31008,all_input_hashes_verified=True,all_448_soft_labels_recomputed=True,original_448_main_masks_unchanged=True,same_initial_weights=True,common_sample_augmentation_matches=True,cuda_batch12_finite_gradient=True,initial_loss=float(loss.detach())))
 (OUT/'PREFLIGHT_OK').touch()
 print(json.dumps(dict(root=str(OUT),preflight='PASS',steps=31008)),flush=True)

if __name__=='__main__':main()
