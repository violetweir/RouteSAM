import hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import pycocotools.mask as mu
import yaml
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');R=P/'new_project/experiments/round2_selection_ablation_20260910'
TP=P/'work/kvasir_tp_filterfirst_students_20260909'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 code=R/'code'
 for name in ['train_sam3_lora_kvasir_e50.py']:
  s=(P/'scripts'/name).read_text();old='Path(__file__).resolve().parent.parent / "MedSAM3" / "lora_layers.py"'
  assert old in s;s=s.replace(old,'Path(__file__).resolve().parent / "lora_layers.py"')
  (code/name).write_text(s)
 shutil.copy2(P/'MedSAM3/lora_layers.py',code/'lora_layers.py')
 arms=json.loads((R/'POOLS_FROZEN.json').read_text())['training_arms']
 support=read(TP/'protocol/support_manifest.jsonl');assert len(support)==8
 gtids={r['merged_id'] for r in support}
 for arm in arms:
  root=R/'data'/arm;(root/'train/images').mkdir(parents=True,exist_ok=False)
  pseudo=read(R/'pools'/arm/'manifest.jsonl')
  records=[dict(target_id=r['target_id'],image_path=r['image_path'],label_path=r['pseudo_mask_path'],is_gt=False) for r in pseudo]
  records += [dict(target_id=r['merged_id'],image_path=r['image_path'],label_path=r['mask_path'],is_gt=True) for r in support]
  records.sort(key=lambda r:r['target_id']);assert len(records)==len({r['target_id'] for r in records})
  images=[];annotations=[]
  for i,r in enumerate(records,1):
   assert r['is_gt']==(r['target_id'] in gtids)
   dst=root/'train/images'/(r['target_id'].replace('::','__')+'.png')
   Image.open(r['image_path']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR).save(dst)
   m=np.asarray(Image.open(r['label_path']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
   assert m.any();ys,xs=np.where(m)
   rle=mu.encode(np.asfortranarray(m.astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii')
   assert np.array_equal(mu.decode(rle)>0,m)
   images.append(dict(id=i,file_name='images/'+dst.name,width=256,height=256))
   annotations.append(dict(id=i,image_id=i,category_id=1,bbox=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],segmentation=rle,area=int(m.sum()),iscrowd=0))
  save(root/'train/_annotations.coco.json',dict(images=images,annotations=annotations,categories=[dict(id=1,name='colon polyp')]))
  save(root/'records.json',records)
  cfg=yaml.safe_load((P/'configs/c0_256_base_b7_medsam3_lora_e50.yaml').read_text())
  cfg['training'].update(data_dir=str(root),num_epochs=10,num_workers=2,seed=2026)
  cfg['output']['output_dir']=str(R/'runs'/arm)
  cfg['evaluation']=dict(metric='direct_validation_dice256',test='after both validation-best checkpoints frozen')
  (R/f'{arm}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 shared=R/'data/shared';shared.mkdir(parents=True,exist_ok=False)
 for split in ['validation','test']:
  records=read(P/f'work/kvasir_1pct_anchors/baseline_data/{split}/metadata.jsonl');assert len(records)==100
  save(shared/f'{split}.json',sorted(records,key=lambda r:r['merged_id']))
 save(R/'training_data_audit.json',dict(arms={arm:len(json.loads((R/'data'/arm/'records.json').read_text())) for arm in arms},
      gt_count=8,train_masks_encoded_verified=True,train_images_and_masks_same_canvas=True,
      no_unlabeled_gt_read_for_training=True,validation_and_test_gt_not_in_inference_inputs=True,
      hashes={str(p):sha(p) for p in code.glob('*.py')}))
 (R/'TRAINING_PREPARED').write_text('ready\n')
 print('Prepared',arms)
if __name__=='__main__':main()
