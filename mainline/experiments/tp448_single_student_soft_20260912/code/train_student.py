"""Uniform epoch training: one U-Net, equal per-image BCE + soft Dice."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,Sampler
from Model.model import SamUnet
from pinned_transforms import build_weak_strong_transforms

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_hash(state):
 h=hashlib.sha256()
 for key,value in sorted(state.items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def decode(p):
 im=Image.open(p)
 if im.mode in ('RGB','RGBA'):im=im.convert('L')
 a=np.asarray(im)
 if a.dtype==np.uint8:return a.astype(np.float32)/255
 if a.dtype in (np.uint16,np.int32) and a.min()>=0 and a.max()<=65535:return a.astype(np.float32)/65535
 raise ValueError(f'Unsupported label encoding {p}: {a.dtype}')
def equal_loss(logits,target):
 # Difference of the two class logits is the foreground log-odds.
 z=logits[:,1]-logits[:,0];prob=torch.sigmoid(z)
 bce=F.binary_cross_entropy_with_logits(z,target,reduction='none').flatten(1).mean(1)
 inter=(prob*target).flatten(1).sum(1)
 dice=1-(2*inter+1)/(prob.flatten(1).sum(1)+target.flatten(1).sum(1)+1)
 return (bce+dice).mean(),bce.mean(),dice.mean()
class EpochSampler(Sampler):
 def __init__(self,n,seed):self.n=n;self.seed=seed;self.epoch=1
 def order(self):return np.random.default_rng(self.seed+self.epoch).permutation(self.n).tolist()
 def __iter__(self):return iter((self.epoch,i) for i in self.order())
 def __len__(self):return self.n
class StudentDataset(Dataset):
 def __init__(self,rows,variant,training,seed=2026):
  self.rows=rows;self.variant=variant;self.training=training;self.seed=seed
  self.transforms=build_weak_strong_transforms(argparse.Namespace(image_size=256)) if training else None
 def __len__(self):return len(self.rows)
 def __getitem__(self,key):
  epoch,index=key if isinstance(key,tuple) else (0,key);row=self.rows[index]
  image=np.asarray(Image.open(row['image_path']).convert('RGB').resize((256,256),Image.Resampling.NEAREST),dtype=np.float32)/255
  label_path=row['hard_label_path'] if row['is_gt'] or self.variant=='hard' else row['soft_label_path']
  target=decode(label_path)
  if target.shape!=(256,256):target=np.asarray(Image.fromarray(target).resize((256,256),Image.Resampling.NEAREST),dtype=np.float32)
  if self.training:
   item_seed=(self.seed+epoch*1000003+row.get('augmentation_index',index)*97)%(2**32)
   random.seed(item_seed);np.random.seed(item_seed)
   out=self.transforms['train_weak' if row['is_gt'] else 'train_strong'](image=image,mask=target)
   image,target=out['image'],out['mask']
  assert image.shape==(256,256,3) and target.shape==(256,256)
  assert np.isfinite(target).all() and target.min()>=0 and target.max()<=1
  image=(image-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
  return dict(image=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))),target=torch.from_numpy(np.ascontiguousarray(target)).float(),target_id=row['target_id'],is_gt=row['is_gt'])
@torch.inference_mode()
def evaluate(model,loader,output=None):
 previous=model.training;model.eval();metrics=[]
 if output is not None:(output/'masks').mkdir(parents=True,exist_ok=True)
 for batch in loader:
  _,prob=model(batch['image'].cuda(non_blocking=True));pred=(prob[:,1]>=.5).cpu().numpy();gt=batch['target'].numpy()>=.5
  for i,target in enumerate(batch['target_id']):
   a=pred[i];b=gt[i];inter=int((a&b).sum());total=int(a.sum())+int(b.sum());union=total-inter
   row=dict(target_id=target,dice=2*inter/total if total else 1.,iou=inter/union if union else 1.,nonempty=bool(a.any()))
   if output is not None:
    path=output/'masks'/(target.replace('::','__')+'.png');Image.fromarray((a*255).astype(np.uint8)).save(path);row.update(mask_path=str(path),mask_sha256=sha(path))
   metrics.append(row)
 assert len(metrics)==100
 summary=dict(count=len(metrics),dice=float(np.mean([x['dice'] for x in metrics])),iou=float(np.mean([x['iou'] for x in metrics])))
 if output is not None:
  (output/'per_target_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in metrics));save(output/'summary.json',summary)
 model.train(previous);return summary
def loader(dataset,batch_size,sampler=None,workers=4):
 return DataLoader(dataset,batch_size=batch_size,sampler=sampler,shuffle=False,drop_last=False,num_workers=workers,persistent_workers=workers>0,pin_memory=True,generator=torch.Generator().manual_seed(902026))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--variant',choices=['hard','soft'],required=True);args=parser.parse_args()
 root=args.root;dest=root/'runs'/args.variant;dest.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((root/'config.json').read_text());seed_all(cfg['seed']);torch.set_num_threads(4)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(.20)
 rows=read(root/'data/train_manifest.jsonl');assert len(rows)==456 and sum(x['is_gt'] for x in rows)==8
 dataset=StudentDataset(rows,args.variant,True,cfg['seed']);sampler=EpochSampler(len(rows),cfg['seed']);train_loader=loader(dataset,12,sampler)
 val_loader=loader(StudentDataset(read(root/'data/validation_manifest.jsonl'),'hard',False),1,workers=1)
 model=SamUnet(argparse.Namespace(in_channels=3,num_classes=2));model.load_state_dict(torch.load(root/'initial_model.pth',map_location='cpu'))
 initial_hash=tensor_hash(model.state_dict());assert initial_hash==cfg['initial_tensor_sha256'];model=model.cuda().train()
 optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=1e-4)
 save(dest/'protocol.json',{**cfg,'variant':args.variant,'initial_tensor_sha256_verified':initial_hash})
 best=-1.;best_info=None;total_steps=0;start=time.time()
 try:
  for epoch in range(1,817):
   sampler.epoch=epoch;lr=.01*(1-(epoch-1)/816)
   for group in optimizer.param_groups:group['lr']=lr
   order=sampler.order();seen=[];gt_count=0;zero_gt_batches=0;losses=[]
   for batch in train_loader:
    logits,_=model(batch['image'].cuda(non_blocking=True));loss,bce,dice=equal_loss(logits,batch['target'].cuda(non_blocking=True))
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total_steps+=1
    seen.extend(batch['target_id']);n_gt=int(batch['is_gt'].sum());gt_count+=n_gt;zero_gt_batches+=int(n_gt==0);losses.append(float(loss.detach()))
    if total_steps==1:save(dest/'status.json',dict(stage='training',epoch=epoch,iteration=1,first_batch_gt=n_gt,loss=losses[-1]))
   assert seen==[rows[i]['target_id'] for i in order] and len(set(seen))==456 and gt_count==8
   row=dict(epoch=epoch,iteration=total_steps,steps_in_epoch=len(losses),loss=float(np.mean(losses)),lr=lr,gt_count=gt_count,pseudo_count=456-gt_count,zero_gt_batches=zero_gt_batches,unique_images=len(set(seen)),order_sha256=hashlib.sha256('\n'.join(seen).encode()).hexdigest(),elapsed_seconds=time.time()-start)
   with (dest/'train_epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   save(dest/'status.json',dict(stage='training',**row,best_validation=best if best>=0 else None))
   if epoch%4==0:
    val=evaluate(model,val_loader);improved=val['dice']>best
    record=dict(epoch=epoch,iteration=total_steps,**val,is_new_best=improved)
    with (dest/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if improved:
     best=val['dice'];best_info=record;torch.save(model.state_dict(),dest/'student_best.pth');save(dest/'student_best.json',record)
    torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict()),dest/'training_latest.pth')
    print(json.dumps(dict(epoch=epoch,iteration=total_steps,validation_dice=val['dice'],best_validation_dice=best)),flush=True)
  assert total_steps==31008
  torch.save(model.state_dict(),dest/'student_final.pth');save(dest/'summary.json',dict(epochs=816,iterations=total_steps,best_validation=best_info,final_validation=val,initial_tensor_sha256=initial_hash,test_evaluated=False))
  save(dest/'status.json',dict(stage='complete',epoch=816,iteration=total_steps,best_validation=best));(dest/'COMPLETE').write_text('complete\n')
 except BaseException:
  import traceback
  error=traceback.format_exc();(dest/'FAILED.txt').write_text(error);save(dest/'status.json',dict(stage='failed',error=error));raise
if __name__=='__main__':main()
