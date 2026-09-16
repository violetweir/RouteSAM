import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys,json,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'new_project/experiments/tp_tracker_endpoint_20260910'
sys.path.insert(0,str(R/'code'));sys.path.insert(1,str(P/'scripts'))
from endpoint_api import EndpointCapture,public_forward,terminal_loss
from train_round3_tracker import configure_trainable,hash_named_parameters,save_json,read_jsonl
from eval_route_propagation_quality import t21
from sam3.model_builder import build_sam3_video_model
BASE='/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
def dice(a,b):
 n=int(a.sum())+int(b.sum());return 2*int((a&b).sum())/n if n else 1.
def main():
 torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026)
 dest=R/'preflight_public_v3';dest.mkdir(exist_ok=False)
 rows=read_jsonl(P/'work/kvasir_tp_filterfirst_students_20260909/quality/anchor_conditioned_target_pooling/propagation_quality_train/propagation_quality.jsonl')
 ids=sorted({r['target_id'] for r in rows})[:3]
 rows=sorted([r for r in rows if r['target_id'] in ids and r['bridge_count'] in [0,6]],key=lambda r:(r['target_id'],r['bridge_count']))
 model=build_sam3_video_model(checkpoint_path=BASE,load_from_HF=False,device='cuda',compile=False).eval()
 torch.set_autocast_cache_enabled(False);torch.clear_autocast_cache()
 audit=configure_trainable(model,'memory_attention_decoder_lora',dict(rank=4,alpha=8,dropout=0.))
 save_json(dest/'module_audit.json',audit)
 before,_=hash_named_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
 results=[]
 for row in rows:
  with EndpointCapture(model,len(row['bridge_image_paths'])+1) as cap:
   selected=public_forward(model,row,t21.load_rgb,t21.select_top)
  saved=np.asarray(Image.open(row['forward_mask_path']).convert('L'))>127
  logits,expected=cap.replay(selected['object_id'])
  a=(logits.detach()[0,0]>0).cpu().numpy();b=(expected[0,0]>0).cpu().numpy()
  compare=dict(target_id=row['target_id'],bridge=row['bridge_count'],public_vs_saved=dice(selected['mask'],saved),replay_vs_original_raw=dice(a,b),max_logit_difference=float((logits.detach()-expected).abs().max()))
  target=torch.tensor(saved,device='cuda',dtype=torch.float32)[None,None]
  with torch.enable_grad():loss=terminal_loss(logits,target)
  assert torch.isfinite(loss);loss.backward()
  groups={}
  for n,p in model.named_parameters():
   if not p.requires_grad:assert p.grad is None
   elif p.grad is not None:
    assert torch.isfinite(p.grad).all()
    k='memory' if n.startswith('tracker.transformer.') else 'decoder'
    groups[k]=groups.get(k,0.)+float(p.grad.float().square().sum())
  compare['gradient_norms']={k:v**.5 for k,v in groups.items()};compare['loss']=float(loss.detach())
  assert groups.get('memory',0)>0 and groups.get('decoder',0)>0
  model.zero_grad(set_to_none=True);results.append(compare);save_json(dest/'results.json',results);print(json.dumps(compare),flush=True)
  del logits,expected,loss,cap;torch.cuda.empty_cache()
 after,_=hash_named_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
 assert before==after
 passed=all(r['public_vs_saved']>.995 and r['replay_vs_original_raw']>.995 for r in results)
 save_json(dest/'summary.json',dict(passed=passed,frozen_unchanged=True,gradient_checks=6,
  method='Unchanged public pipeline; replay exact terminal step with detached current-model history; no full-chain backpropagation',formal_training_started=False))
 assert passed
if __name__=='__main__':
 try:main()
 except BaseException:
  (R/'PREFLIGHT_PUBLIC_FAILED.txt').write_text(traceback.format_exc());raise
