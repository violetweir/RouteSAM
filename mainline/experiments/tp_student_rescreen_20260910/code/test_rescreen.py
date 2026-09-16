"""Check policy edge cases, all-target decisions and actual dynamic epoch loader."""
import argparse
import collections
import json
import math
from pathlib import Path
import numpy as np
import torch
from rescreen_pool import tier,select
from train_student import read,save,sha,decode,EpochSampler,StudentDataset,loader,equal_loss,SamUnet,tensor_hash

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();r=args.root
torch.set_num_threads(4)
base=dict(candidate_nonempty=True,q_return=.95,q_tp=.95,q_student=0.,student_nonempty=False,router_score=.2,bridge_count=3,route_id='A')
assert tier(base)=='A'
b={**base,'q_tp':.80,'q_student':.85,'student_nonempty':True,'router_score':.99,'route_id':'B'};assert tier(b)=='B'
for altered in [{**b,'q_return':.9499},{**b,'q_tp':.7999},{**b,'q_student':.8499},{**b,'student_nonempty':False},{**base,'candidate_nonempty':False}]:assert tier(altered)=='C'
pair=[{**x,'tier':tier(x)} for x in [base,b]];assert select(pair)['route_id']=='A'
assert select([{**b,'tier':'B'}])['route_id']=='B'
for old_member in [True,False]:assert tier({**b,'old_member':old_member})=='B'
rows=read(r/'data/train_manifest.jsonl');cfg=json.loads((r/'config.json').read_text());n=len(rows)
assert n==cfg['train_count'] and n>8 and sum(x['is_gt'] for x in rows)==8
decisions=read(r/'screening/train/decisions.jsonl');scores=read(r/'screening/train/candidate_scores.jsonl');assert len(decisions)==792 and len(scores)==5544
groups=collections.defaultdict(list)
for x in scores:assert x['tier']==tier(x);groups[x['target_id']].append(x)
for x in decisions:
 choice=select(groups[x['target_id']]);assert x['accepted']==(choice is not None)
 if choice:assert choice['route_id']==x['route_id']
assert {x['target_id'] for x in rows if not x['is_gt']}=={x['target_id'] for x in decisions if x['accepted']}
for row in rows:
 assert sha(row['image_path'])==row['image_sha256'] and sha(row['hard_label_path'])==row['hard_sha256'] and sha(row['soft_label_path'])==row['soft_sha256']
 assert row['sample_weight']==row['pixel_weight']==1.
 assert np.array_equal(decode(row['soft_label_path'])>=.5,decode(row['hard_label_path'])>=.5)
dataset=StudentDataset(rows,'soft',True);sampler=EpochSampler(n,cfg['seed']);batches=loader(dataset,12,sampler,workers=2)
seen=[];gt=0;batch_sizes=[];first=None
for batch in batches:
 if first is None:first=batch
 seen.extend(batch['target_id']);gt+=int(batch['is_gt'].sum());batch_sizes.append(len(batch['target_id']))
assert len(seen)==len(set(seen))==n and gt==8 and len(batch_sizes)==math.ceil(n/12)
assert batch_sizes[-1]==(n%12 or 12) and cfg['total_iterations']==816*len(batch_sizes)
z=torch.randn(12,2,8,8);y=torch.rand(12,8,8);assert torch.allclose(equal_loss(z,y)[0],torch.stack([equal_loss(z[i:i+1],y[i:i+1])[0] for i in range(12)]).mean(),atol=1e-6)
torch.cuda.set_per_process_memory_fraction(.20);model=SamUnet(argparse.Namespace(in_channels=3,num_classes=2));model.load_state_dict(torch.load(r/'initial_model.pth',map_location='cpu'))
assert tensor_hash(model.state_dict())==cfg['initial_tensor_sha256'];model=model.cuda().train();logits,_=model(first['image'].cuda());loss=equal_loss(logits,first['target'].cuda())[0];loss.backward();assert torch.isfinite(loss)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
result=dict(policy_edge_cases_passed=True,A_not_vetoed_by_student=True,B_requires_both_models=True,historical_membership_not_used=True,all792_decisions_recomputed=True,all_training_hashes_verified=True,train_count=n,pseudo_count=n-8,GT_count=8,unique_images_in_real_epoch=n,batches_per_epoch=len(batch_sizes),last_batch_size=batch_sizes[-1],total_iterations=cfg['total_iterations'],same_initial_tensors_verified=True,equal_per_image_loss=True,cuda_smoke_finite=True)
save(r/'preflight_results.json',result);(r/'PREFLIGHT_OK').write_text('ok\n');print(json.dumps(result,indent=2))
