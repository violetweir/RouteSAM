"""Verify no-replacement epochs, equal loss and matched hard/soft inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_student import read,save,sha,decode,equal_loss,EpochSampler,StudentDataset,loader,SamUnet,seed_all,tensor_hash

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();r=args.root
torch.set_num_threads(4);seed_all(2026);rows=read(r/'data/train_manifest.jsonl');assert len(rows)==588
for row in rows:
 assert sha(row['image_path'])==row['image_sha256'] and sha(row['hard_label_path'])==row['hard_sha256'] and sha(row['soft_label_path'])==row['soft_sha256']
 hard=decode(row['hard_label_path']);soft=decode(row['soft_label_path'])
 assert set(np.unique(hard))<={0.,1.} and np.array_equal(soft>=.5,hard>=.5)
 if row['is_gt']:assert np.array_equal(hard,soft)
 assert row['sample_weight']==row['pixel_weight']==1
sampler=EpochSampler(588,2026);orders=[]
for epoch in [1,2,816]:
 sampler.epoch=epoch;order=sampler.order();assert sorted(order)==list(range(588));assert sum(rows[i]['is_gt'] for i in order)==8;orders.append(order)
assert orders[0]!=orders[1]
z=torch.randn(12,2,8,8,requires_grad=True);y=torch.rand(12,8,8)
batch_loss=equal_loss(z,y)[0];individual=torch.stack([equal_loss(z[i:i+1],y[i:i+1])[0] for i in range(12)]).mean()
assert torch.allclose(batch_loss,individual,atol=1e-6)
perm=torch.randperm(12);assert torch.allclose(batch_loss,equal_loss(z[perm],y[perm])[0],atol=1e-6)
assert abs(float(equal_loss(torch.zeros_like(z),y)[0]-equal_loss(torch.zeros_like(z),(y>=.5).float())[0]))>1e-6
hard_ds=StudentDataset(rows,'hard',True);soft_ds=StudentDataset(rows,'soft',True)
indices=[next(i for i,x in enumerate(rows) if x['is_gt']),next(i for i,x in enumerate(rows) if not x['is_gt'])]
for epoch in [1,2]:
 for i in indices:
  a=hard_ds[(epoch,i)];b=soft_ds[(epoch,i)];assert torch.equal(a['image'],b['image']);assert torch.equal(a['target']>=.5,b['target']>=.5)
# A whole real epoch must contain each ID exactly once, without fixed GT quotas.
real_loader=loader(soft_ds,12,EpochSampler(588,2026),workers=2);seen=[];gt=0;zero=0;first=None
for batch in real_loader:
 if first is None:first=batch
 seen.extend(batch['target_id']);n=int(batch['is_gt'].sum());gt+=n;zero+=int(n==0)
assert len(seen)==len(set(seen))==588 and gt==8 and zero>0
del real_loader
# Real CUDA forward/backward at the agreed batch size, with no GT/pseudo slicing.
torch.cuda.set_per_process_memory_fraction(.20)
model=SamUnet(argparse.Namespace(in_channels=3,num_classes=2));model.load_state_dict(torch.load(r/'initial_model.pth',map_location='cpu'))
assert tensor_hash(model.state_dict())==json.loads((r/'config.json').read_text())['initial_tensor_sha256']
model=model.cuda().train();logits,_=model(first['image'].cuda());loss=equal_loss(logits,first['target'].cuda())[0];loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
result=dict(all_588_input_hashes_verified=True,all_580_soft_targets_binary_match_main=True,epochs_each_have_588_unique_images_and_8_GT=True,per_image_equal_loss_verified=True,loss_permutation_invariant=True,soft_targets_not_thresholded=True,hard_soft_augmentations_identical=True,real_epoch_batch_count=49,real_epoch_zero_gt_batches=zero,cuda_batch12_forward_backward_finite=True,cuda_smoke_loss=float(loss.detach()))
save(r/'preflight_results.json',result);(r/'PREFLIGHT_OK').write_text('ok\n');print(json.dumps(result,indent=2))
