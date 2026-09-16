import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import collections,hashlib,json,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'new_project/experiments/tp_tracker_endpoint_20260910'
D=R/'pilot'
S=P/'new_project/experiments/tp_student_rescreen_20260910'
sys.path.insert(0,str(R/'code'));sys.path.insert(1,str(P/'scripts'))
from endpoint_api import EndpointCapture,public_forward,terminal_loss
from train_round3_tracker import configure_trainable,hash_named_parameters,save_json,read_jsonl
from eval_route_propagation_quality import propagate_with_trace,t21
from sam3.model_builder import build_sam3_video_model
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def append(p,x):
 with p.open('a') as f:f.write(json.dumps(x)+'\n')
def binary(p):return np.asarray(Image.open(p).convert('L'))>127
def dice(a,b):
 n=int(a.sum())+int(b.sum());return 2*int((a&b).sum())/n if n else 1.
def state(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
def save_checkpoint(model,optimizer,step,name):
 torch.save(dict(adapter=state(model),optimizer=optimizer.state_dict(),step=step,epoch=1 if step==628 else 0,
                 torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),D/name)

def validate(model,cfg):
 dest=D/'validation_epoch1';(dest/'masks').mkdir(parents=True,exist_ok=False)
 save_json(D/'status.json',dict(stage='validation',completed_routes=0,total_routes=700))
 routes=read_jsonl(D/'validation_routes.jsonl');scores=[]
 for i,row in enumerate(sorted(routes,key=lambda r:(r['target_id'],r['bridge_count'])),1):
  paths=[row['anchor_image_path'],*row['bridge_image_paths'],row['target_image_path']]
  with torch.inference_mode():
   out=propagate_with_trace(model,paths,row['anchor_box_xywh_normalized'],256)
   cycle=t21.propagate_return_from_predicted_mask(model,paths,out['final_mask'],256)
  m=out['final_mask'];f=dest/'masks'/(row['route_id']+'.png');Image.fromarray((m*255).astype(np.uint8)).save(f)
  record=dict(target_id=row['target_id'],route_id=row['route_id'],bridge_count=row['bridge_count'],mask_path=str(f),mask_sha256=sha(f),
              q_return=dice(cycle['mask'],t21.load_mask(row['anchor_mask_path'],256)),cycle_success=cycle['success'])
  scores.append(record);append(dest/'candidate_manifest.jsonl',record)
  if i%10==0:save_json(D/'status.json',dict(stage='validation',completed_routes=i,total_routes=700));print('validation',i,flush=True)
 groups=collections.defaultdict(list)
 for r in scores:groups[r['target_id']].append(r)
 student={r['target_id']:r for r in read_jsonl(S/'runs/soft/validation_best/per_target_metrics.jsonl')}
 selected=[]
 for target,group in sorted(groups.items()):
  mm=[binary(r['mask_path']) for r in group];sm=binary(student[target]['mask_path'])
  for i,r in enumerate(group):
   a=r['q_return'];b=float(np.mean([dice(mm[i],m) for j,m in enumerate(mm) if i!=j]));c=dice(mm[i],sm)
   r.update(q_multi=b,q_model=c,b7_score=(max(a,1e-6)*max(b,1e-6)**2*max(c,1e-6)**2)**.2)
  selected.append(dict(max(group,key=lambda r:(r['b7_score'],r['q_multi'],r['q_return'],r['route_id']))))
 for r in selected:append(dest/'selected_masks_frozen.jsonl',r)
 save_json(dest/'SELECTION_FROZEN.json',dict(sha256=sha(dest/'selected_masks_frozen.jsonl')))
 gtmap={r['target_id']:r for r in read_jsonl(S/'data/validation_manifest.jsonl')}
 metrics=[];bridge_metrics=collections.defaultdict(list)
 for r in selected:
  gt=t21.load_mask(gtmap[r['target_id']]['hard_label_path'],256)
  candidates=groups[r['target_id']]
  oracle=max(dice(binary(c['mask_path']),gt) for c in candidates)
  for c in candidates:bridge_metrics[c['bridge_count']].append(dice(binary(c['mask_path']),gt))
  metrics.append({**r,**t21.metrics(binary(r['mask_path']),gt),'oracle':oracle})
 for r in metrics:append(dest/'per_target_metrics.jsonl',r)
 result=dict(count=100,dice=float(np.mean([r['dice'] for r in metrics])),iou=float(np.mean([r['iou'] for r in metrics])),
             oracle=float(np.mean([r['oracle'] for r in metrics])),by_bridge_dice={k:float(np.mean(v)) for k,v in bridge_metrics.items()})
 result['delta_b7']=result['dice']-cfg['base_validation_b7'];result['delta_oracle']=result['oracle']-cfg['base_validation_oracle']
 result['selected_epoch']=1 if result['delta_b7']>0 else 0
 save_json(dest/'summary.json',result);return result

def main():
 assert (D/'PREPARED').exists() and not (D/'STARTED').exists()
 (D/'STARTED').write_text(str(os.getpid()))
 cfg=json.loads((D/'config.json').read_text());rows=read_jsonl(D/'train_targets.jsonl')
 assert len(rows)==628 and len({r['target_id'] for r in rows})==628
 assert all(sha(r['label_path'])==r['label_sha256'] for r in rows)
 torch.set_num_threads(4);random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
 model=build_sam3_video_model(checkpoint_path=cfg['initialization'],load_from_HF=False,device='cuda',compile=False).eval()
 torch.set_autocast_cache_enabled(False);torch.clear_autocast_cache()
 audit=configure_trainable(model,cfg['lora']['policy'],cfg['lora']);save_json(D/'module_audit.json',audit)
 frozen_before,_=hash_named_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
 optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
 order=np.random.default_rng(2027).permutation(628).tolist();save_json(D/'epoch1_order.json',[rows[i]['target_id'] for i in order])
 save_checkpoint(model,optimizer,0,'initial.pt');start=time.time();seen=[];fallbacks=0
 for step,index in enumerate(order,1):
  label=rows[index];cap=None;selected=None;attempts=0
  for route in label['paths']:
   attempts+=1
   with EndpointCapture(model,len(route['bridge_image_paths'])+1) as cap:
    selected=public_forward(model,route,t21.load_rgb,t21.select_top)
   if selected['object_id'] is not None and any(selected['object_id'] in c['obj_ids'] for c in cap.records):break
   del cap;cap=None
  if cap is None:raise RuntimeError('No supervised terminal tracking output for '+label['target_id'])
  logits,expected=cap.replay(selected['object_id'])
  agreement=dice((logits.detach()[0,0]>0).cpu().numpy(),(expected[0,0]>0).cpu().numpy())
  assert agreement>.995,agreement
  arr=np.asarray(Image.open(label['label_path']));scale=255 if arr.dtype==np.uint8 else 65535
  y=torch.tensor(arr.astype(np.float32)/scale,device='cuda')[None,None]
  assert y.shape==(1,1,256,256) and y.min()>=0 and y.max()<=1
  loss=terminal_loss(logits,y);assert torch.isfinite(loss)
  optimizer.zero_grad(set_to_none=True);loss.backward()
  grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['grad_clip'],error_if_nonfinite=True)
  assert grad>0
  if step==1:
   nonzero=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is not None and bool(p.grad.count_nonzero())]
   assert any(n.startswith('tracker.transformer.') for n in nonzero)
   assert any(n.startswith('tracker.sam_mask_decoder.') for n in nonzero)
   assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
   save_json(D/'first_backward.json',dict(nonzero_adapter_gradients=nonzero,frozen_gradient_count=0,replay_mask_dice=agreement))
  optimizer.step();torch.clear_autocast_cache()
  fallbacks+=int(attempts>1);seen.append(label['target_id'])
  record=dict(stage='training',epoch=1,step=step,total_steps=628,target_id=label['target_id'],is_gt=label['is_gt'],
              bridge=route['bridge_count'],route_id=route['route_id'],attempts=attempts,loss=float(loss.detach()),gradient_norm=float(grad),
              elapsed_seconds=time.time()-start,replay_mask_dice=agreement)
  append(D/'train_steps.jsonl',record);save_json(D/'status.json',record)
  if step%25==0 or step==1:print(json.dumps(record),flush=True)
  if step%100==0:save_checkpoint(model,optimizer,step,'latest.pt')
  del logits,expected,loss,cap,y;torch.cuda.empty_cache()
 assert len(set(seen))==628
 save_checkpoint(model,optimizer,628,'epoch1.pt')
 frozen_after,_=hash_named_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
 assert frozen_after==frozen_before
 save_json(D/'train_audit.json',dict(unique_targets=628,gt=8,pseudo=620,steps=628,fallback_targets=fallbacks,frozen_hash_before=frozen_before,frozen_hash_after=frozen_after))
 (D/'TRAINING_COMPLETE').write_text('1 epoch\n')
 result=validate(model,cfg)
 save_json(D/'results.json',result)
 report=f'''# TP 传播端点微调：首轮结果

采用完整 TP 推理接口，以当前模型生成的历史记忆作为固定上下文，仅重算终点并反向传播。
训练池保持 620 张软伪标签 + 8 张 GT，每张目标一次、等权；LoRA rank4/alpha8，仅记忆注意力与解码器；AdamW 1e-5，1 epoch。
这不是整条路径的端到端反向传播。没有重新划分数据，没有使用无标注训练图 GT 或 test GT。

| Validation 指标 | SAM3-base | 微调 epoch1 |
|---|---:|---:|
| TP+B7 Dice | {cfg['base_validation_b7']:.6f} | {result['dice']:.6f} |
| TP Oracle Dice | {cfg['base_validation_oracle']:.6f} | {result['oracle']:.6f} |

按完整 validation B7 选择：epoch {result['selected_epoch']}（0 表示保留 base）。本轮未测试 test。
逐图预测、候选清单、冻结选择、训练及梯度核查位于 `{D}`。
'''
 (D/'report.md').write_text(report,encoding='utf-8');(P/'new_project/tp_tracker_pilot_results.md').write_text(report,encoding='utf-8')
 save_json(D/'status.json',dict(stage='complete',**result));(D/'COMPLETE').write_text('complete\n')

if __name__=='__main__':
 try:main()
 except BaseException:
  error=traceback.format_exc();(D/'FAILED.txt').write_text(error);save_json(D/'status.json',dict(stage='failed',error=error));raise
