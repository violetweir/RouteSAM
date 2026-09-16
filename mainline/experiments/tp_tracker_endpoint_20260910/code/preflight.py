import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R = P / 'new_project/experiments/tp_tracker_endpoint_20260910'
sys.path.insert(0, str(R / 'code'))
sys.path.insert(1, str(P / 'scripts'))
from train_round3_tracker import FrozenSpatialCache, configure_trainable, hash_named_parameters, save_json, read_jsonl
from eval_route_propagation_quality import propagate_with_trace
from sam3.model_builder import build_sam3_video_model
BASE = '/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'

def dice(a, b):
    n = int(a.sum()) + int(b.sum())
    return 2 * int((a & b).sum()) / n if n else 1.

def forward(model, cache, row):
    frames = [row['anchor_image_path'], *row['bridge_image_paths'], row['target_image_path']]
    fpn, position = cache.load(frames)
    fpn[0] = model.tracker.sam_mask_decoder.conv_s0(fpn[0])
    fpn[1] = model.tracker.sam_mask_decoder.conv_s1(fpn[1])
    x, y, w, h = row['anchor_box_xywh_normalized']
    coords = torch.tensor([[[x,y],[x+w,y+h]]],device='cuda') * 1008.
    n = len(frames)
    backbone = dict(backbone_fpn=fpn, vision_pos_enc=position, num_frames=n,
                    init_cond_frames=[0], frames_to_add_correction_pt=[],
                    frames_not_in_init_cond=list(range(1,n)),
                    point_inputs_per_frame={0:dict(point_coords=coords,point_labels=torch.tensor([[2,3]],device='cuda',dtype=torch.int32))},
                    mask_inputs_per_frame={})
    batch=SimpleNamespace(img_batch=torch.zeros(n,3,1,1,device='cuda',dtype=torch.bfloat16),
                          find_inputs=[SimpleNamespace(img_ids=torch.tensor([i],device='cuda')) for i in range(n)])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out=model.tracker.forward_tracking(backbone,batch,return_dict=True)
    logits=out['non_cond_frame_outputs'][n-1]['pred_masks_high_res'].float()
    return F.interpolate(logits,size=(256,256),mode='bilinear',align_corners=False)

def main():
    torch.set_num_threads(4); torch.manual_seed(2026); np.random.seed(2026)
    dest=R/'preflight';dest.mkdir(parents=True,exist_ok=True)
    q=P/'work/kvasir_tp_filterfirst_students_20260909/quality/anchor_conditioned_target_pooling/propagation_quality_train/propagation_quality.jsonl'
    clean=[{k:v for k,v in r.items() if 'evaluation_only' not in k and not k.startswith('gt_')} for r in read_jsonl(q)]
    ids=sorted({r['target_id'] for r in clean})[:3]
    rows=sorted([r for r in clean if r['target_id'] in ids and r['bridge_count'] in [0,6]],key=lambda r:(r['target_id'],r['bridge_count']))
    save_json(dest/'protocol.json',dict(targets=ids,bridges=[0,6],comparison='public TP vs old differentiable tracker interface; train only; no hidden GT', checkpoint=BASE))
    model=build_sam3_video_model(checkpoint_path=BASE,load_from_HF=False,device='cuda',compile=False).eval()
    for p in model.parameters():p.requires_grad_(False)
    cache=FrozenSpatialCache(model,R/'feature_cache',256,torch.device('cuda'))
    model.tracker.teacher_force_obj_scores_for_mem=False
    model.tracker.prob_to_dropout_spatial_mem=0.
    model.tracker.rng=np.random.default_rng(2026)
    results=[]
    for r in rows:
        frames=[r['anchor_image_path'],*r['bridge_image_paths'],r['target_image_path']]
        with torch.inference_mode():
            public=propagate_with_trace(model,frames,r['anchor_box_xywh_normalized'],256)['final_mask']
        with torch.no_grad():
            train_mask=(forward(model,cache,r)[0,0]>0).cpu().numpy()
        saved=np.asarray(Image.open(r['forward_mask_path']).convert('L'))>127
        result=dict(target_id=r['target_id'],bridge=r['bridge_count'],route_id=r['route_id'],
                    public_vs_saved_dice=dice(public,saved),tracker_vs_public_dice=dice(train_mask,public),
                    disagree_pixels=int((train_mask!=public).sum()),public_area=int(public.sum()),tracker_area=int(train_mask.sum()))
        results.append(result);save_json(dest/'equivalence.json',results);print(json.dumps(result),flush=True)
    audit=configure_trainable(model,'memory_attention_decoder_lora',dict(rank=4,alpha=8,dropout=0.))
    save_json(dest/'module_audit.json',audit)
    frozen_before,_=hash_named_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    model.tracker.train(); r=rows[-1]
    logits=forward(model,cache,r)
    target=torch.tensor(np.asarray(Image.open(r['forward_mask_path']).convert('L'))>127,device='cuda',dtype=torch.float32)[None,None]
    prob=logits.sigmoid()
    loss=F.binary_cross_entropy_with_logits(logits,target)+1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
    assert torch.isfinite(loss)
    loss.backward()
    groups={}
    for name,p in model.tracker.named_parameters():
        if not p.requires_grad:assert p.grad is None
        elif p.grad is not None:
            assert torch.isfinite(p.grad).all()
            key='memory' if name.startswith('transformer.') else 'decoder'
            groups[key]=groups.get(key,0.)+float(p.grad.float().square().sum())
    assert groups.get('memory',0)>0 and groups.get('decoder',0)>0, groups
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    optim.step()
    frozen_after,_=hash_named_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    assert frozen_before==frozen_after
    gradient=dict(loss=float(loss.detach()),gradient_norms={k:v**.5 for k,v in groups.items()},frozen_unchanged=True)
    save_json(dest/'gradient.json',gradient);print(json.dumps(gradient),flush=True)
    passed=all(r['public_vs_saved_dice']>.995 and r['tracker_vs_public_dice']>.995 for r in results)
    save_json(dest/'summary.json',dict(equivalence_passed=passed,gradient_passed=True,formal_training_started=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        R.mkdir(parents=True,exist_ok=True);(R/'PREFLIGHT_FAILED.txt').write_text(traceback.format_exc());raise
