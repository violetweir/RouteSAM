"""Regression checks for real sampler, loss partitions, and manifest weights."""
import argparse
import ast
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import run_t24_student as train
from supervision_batch import align_supervision_batch
from build_x3_pool_v2 import pick,tier

torch.set_num_threads(4)
np.random.seed(2026)
sampler=train.LongTwoStreamBatchSampler(list(range(8)),list(range(8,588)),12,6,1000)
wrong=[]
for ids in sampler:
 flags=torch.tensor([i<8 for i in ids]);wrong.append(int((~flags[:6]).sum()))
 batch=dict(is_labeled=flags,image=torch.tensor(ids)[:,None],target=torch.tensor(ids),soft_target=torch.tensor(ids).float(),pixel_weight=torch.tensor(ids).float(),quality_weight=torch.tensor(ids).float(),merged_id=[str(i) for i in ids])
 fixed=align_supervision_batch(batch,6)
 assert all(i<8 for i in fixed['target'][:6]) and all(i>=8 for i in fixed['target'][6:])
 for key in ['soft_target','pixel_weight','quality_weight']:assert torch.equal(fixed[key],fixed['target'].float())
 assert fixed['merged_id']==[str(int(x)) for x in fixed['target']]
 assert torch.equal(fixed['image'][:,0],fixed['target'])

# Exercise the exact loss block used by training, with aligned vs shuffled input.
source=Path(train.__file__).read_text()
block=source.split('            labeled_bs = ',1)[1].split('            if not torch.isfinite(loss):',1)[0]
block='            labeled_bs = '+block
fn='def actual_loss(batch, logits, args):\n    target=batch["target"]\n    probabilities=torch.softmax(logits,1)\n    dice_loss=train.DiceLoss(2)\n    image=logits\n    iteration=2000\n'+''.join('    '+x[12:]+'\n' for x in block.splitlines())+'    return loss,supervised\n'
class CPUOnly(ast.NodeTransformer):
 def visit_Call(self,node):
  node=self.generic_visit(node)
  return node.func.value if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda' else node
tree=ast.fix_missing_locations(CPUOnly().visit(ast.parse(fn)))
ns=dict(torch=torch,F=F,train=train);exec(compile(tree,'actual_training_loss','exec'),ns)
torch.manual_seed(2026)
base=dict(is_labeled=torch.tensor([True]*6+[False]*6),target=torch.randint(0,2,(12,4,4)),soft_target=torch.rand(12,4,4),pixel_weight=torch.rand(12,4,4),quality_weight=torch.linspace(.2,1,12))
logits=torch.randn(12,2,4,4);perm=torch.randperm(12)
for experiment in ['S2','S3','X3']:
 args=argparse.Namespace(experiment=experiment,labeled_bs=6,pseudo_ramp_iterations=2000,lambda_pseudo=.5)
 a={**base,'logits':logits.clone().requires_grad_(True)}
 b={k:v[perm] for k,v in base.items()};b['logits']=logits[perm].clone().requires_grad_(True)
 raw=b['logits'];aligned=align_supervision_batch(b,6)
 la,sa=ns['actual_loss'](a,a['logits'],args);lb,sb=ns['actual_loss'](aligned,aligned['logits'],args)
 assert torch.allclose(la,lb,atol=1e-6) and torch.allclose(sa,sb,atol=1e-6)
 ga=torch.autograd.grad(la,a['logits'])[0];gb=torch.autograd.grad(lb,raw)[0][torch.argsort(perm)]
 assert torch.allclose(ga,gb,atol=1e-6)

# Admission precedes Router ranking; failed masks cannot win on Router score.
good=dict(q_return=.95,nonempty_safe=True,q_multi=.91,q_model_mean=.91,q_model_min=.81,q_model_var=.001,router_score=.8,bridge_count=1,route_id='good')
bad={**good,'q_return':.94,'router_score':.99,'route_id':'bad'}
for x in [good,bad]:x['quality_tier']=tier(x)
assert pick([bad,good])['route_id']=='good'
assert pick([bad]) is None

# X3 must use declared absolute weights, not renormalize with pool membership.
base_root=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_tp_filterfirst_students_20260909')
rows=[json.loads(x) for x in (base_root/'pseudo_manifest_original.jsonl').read_text().splitlines()][:2]
for row,w in zip(rows,[.27,.71]):row.update(explicit_quality_weight=w,preview_only=False)
with tempfile.TemporaryDirectory() as tmp:
 manifest=Path(tmp)/'manifest.jsonl';manifest.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 config=json.loads((base_root/'students/S2/protocol.json').read_text());config.update(pseudo_manifest=str(manifest),quality_weight_mode='manifest')
 ds=train.T22StudentDataset(argparse.Namespace(**config),'train',None)
 expected={x['target_id']:x['explicit_quality_weight'] for x in rows}
 assert len(ds.rows)==10 and ds.labeled_count==8
 for row in ds.rows[8:]:assert row['quality_weight']==expected[row['merged_id']] and row['target']!=row['gt']
 rows[0]['preview_only']=True;manifest.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 try:train.T22StudentDataset(argparse.Namespace(**config),'train',None)
 except ValueError:pass
 else:raise AssertionError('Preview manifest was not rejected')
result=dict(sampler_batches=1000,old_mean_pseudo_in_supervised_six=float(np.mean(wrong)),old_batches_with_wrong_membership=int(np.count_nonzero(wrong)),fixed_batches_with_wrong_membership=0,all_per_image_fields_aligned=True,S2_S3_X3_loss_and_gradient_permutation_invariant=True,absolute_manifest_weights_preserved=True,preview_rejected=True,candidate_admission_before_ranking=True)
print(json.dumps(result,indent=2))
(Path(__file__).parent.parent/'regression_results.json').write_text(json.dumps(result,indent=2)+'\n')
