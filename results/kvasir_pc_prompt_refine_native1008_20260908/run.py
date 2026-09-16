from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import argparse,json,time,hashlib,gc,copy
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,label
import torch
import sam3.model.vitdet as vitdet
import sam3.model_builder as builder
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.decoder import TransformerDecoder

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');R=P/'work/kvasir_pc_prompt_refine_native1008_20260908';TOK=P/'work/kvasir_pc_evidence_20260908/tokens'
CKPT='/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(name,x):(R/name).write_text(json.dumps(x,indent=2)+'\n')
def path(p):
    p=Path(p);return p if p.is_absolute() else P/p
def mask(p):return np.asarray(Image.open(path(p)).convert('L'))>127
def token(t):return np.load(TOK/(t.replace('::','__')+'.npy')).astype(np.float64)
def dice(a,b):
    n=int(a.sum())+int(b.sum());return 2*int((a&b).sum())/n if n else 1.
def box_of(m,margin=0):
    y,x=np.where(m)
    if not len(x):return None
    h,w=m.shape;x0=max(0,int(x.min())-margin);x1=min(w,int(x.max())+1+margin);y0=max(0,int(y.min())-margin);y1=min(h,int(y.max())+1+margin)
    return [(x0+x1)/2/w,(y0+y1)/2/h,(x1-x0)/w,(y1-y0)/h]
def pc_box(score,base):
    weights=np.asarray(Image.fromarray(base.astype('uint8')*255).resize((18,18),Image.Resampling.BOX),dtype=np.float64)/255
    support=weights>0.25
    if not support.any():support.flat[int(weights.argmax())]=True
    roi=binary_dilation(support,structure=np.ones((3,3)),iterations=2)
    k=max(1,min(int(round(weights.sum())),int(roi.sum())))
    ids=np.flatnonzero(roi);order=ids[np.argsort(score[ids],kind='stable')[-k:]]
    chosen=np.zeros((18,18),bool);chosen.flat[order]=True
    labs,n=label(chosen,structure=np.ones((3,3)))
    # Prefer the selected component most supported by the TP prediction; no GT.
    c=max(range(1,n+1),key=lambda i:(float(weights[labs==i].sum()),int((labs==i).sum()),-i))
    component=labs==c
    return box_of(component,margin=1),{'k':k,'roi_patches':int(roi.sum()),'selected_component_patches':int(component.sum())}
def cpu_addmm_act(activation,linear,x):
    y=linear(x)
    if activation in [torch.nn.functional.gelu,torch.nn.GELU]:return torch.nn.functional.gelu(y)
    if activation in [torch.nn.functional.relu,torch.nn.ReLU]:return torch.nn.functional.relu(y)
    raise ValueError(activation)
ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=100);args=ap.parse_args()
torch.cuda.set_per_process_memory_fraction(.23);torch.set_num_threads(4);torch.set_num_interop_threads(1);R.mkdir(parents=True,exist_ok=True)
config={'split':'validation only','student':False,'fixed_inputs':'Original five-fold OOF TP anchor, path and default mask; original 8 train supports',
 'variants':['tp_box_control','prototype_box','fgbg_box'],
 'tp_box_control':'Bounding box of TP prediction; same SAM3 re-decoding as correspondence variants',
 'correspondence_box':'18x18 area weights from TP mask; ROI=TP weights>.25 dilated2 cells; K=round(TP grid area); highest K scores; component maximizing TP overlap; bbox enlarged1 cell',
 'fgbg':'Mean top-min(3,FGcount) FG cosine minus mean top3 BG cosine under frozen TP anchor',
 'prototype':'L2-normalized mean anchor FG token cosine',
 'decoder':'SAM3-base image geometric box grounding, internal resolution1008, input-output canvas256; implicit visual token, no polyp text; highest model confidence query; empty result falls back to TP',
 'fixed_guard':'Also report guarded output: use decoded mask iff Dice(decoded,TP)>=.9, area ratio in [.5,2], score>=.5; otherwise TP. No learned selector.',
 'primary':'fgbg_box guarded improvement over both original TP and identically guarded TP-box control; mean >=.003, paired95 bootstrap lower>0, >=3 positive original OOF folds for each comparison',
 'selection_policy':'No choice of primary based on best ablation. All raw/guarded outputs and candidate oracles reported; failed gate means no test.',
 'precision':'GPU native1008 bfloat16 autocast; no CPU compatibility patches; memory cap23percent; input-output canvas256; prior corresponding tokens remain256 CPUfloat32',
 'caveat':'One fixed prompt construction, not an exhaustive test of dense correspondence guidance. Existing validation reused across research rounds.',
 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if not (R/'config.json').exists():save('config.json',config)
tp=read(P/'work/kvasir_tp_guided_pc_joint_20260907/tp_baseline/validation_oof.jsonl')
quality=read(P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6/sam3enc_anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl');qi={r['route_id']:r for r in quality}
oldpc=read(P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6/sam3enc_anchor_conditioned_patch_correspondence/propagation_quality_validation/propagation_quality.jsonl')
pool={}
for r in quality+oldpc:pool.setdefault(r['target_id'],[]).append(r['gt_dice_evaluation_only'])
fold_rows=read(P/'work/kvasir_tp_pc_gain_gate_20260907/validation_nested_oof.jsonl');folds={r['target_id']:r['outer_fold'] for r in fold_rows}
supports=read(P/'work/kvasir_1pct_anchors/protocol/support_manifest.jsonl');anchors={}
for a in supports:
    fg=np.asarray(Image.open(a['frozen_mask_path']).convert('L').resize((18,18),Image.Resampling.NEAREST)).reshape(-1)>127
    anchors[a['merged_id']]=(token(a['merged_id']),fg)
old=read(R/'per_target.jsonl') if (R/'per_target.jsonl').exists() else [];done={r['target_id'] for r in old}
pending=[r for r in sorted(tp,key=lambda r:r['target_id'])[:args.limit] if r['target_id'] not in done]
if pending:
    t0=time.time();print('building GPU image model native1008 memory cap23percent; pending',len(pending),flush=True)
    model=build_sam3_image_model(device='cuda',checkpoint_path=CKPT,load_from_HF=False,enable_inst_interactivity=False,compile=False).eval()
    processor=Sam3Processor(model,resolution=1008,device='cuda',confidence_threshold=0.)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        visual=model.backbone.forward_text(['visual'],device='cuda')
        print('model ready',round(time.time()-t0,1),flush=True)
        for ni,r in enumerate(pending):
            q=qi[r['route_id']];base=mask(r['source_mask_path']);z=token(r['target_id']);a,fg=anchors[r['anchor_id']]
            proto=a[fg].mean(0);proto/=max(np.linalg.norm(proto),1e-12);sim=a@z.T
            contrast=np.sort(sim[fg],axis=0)[-min(3,int(fg.sum())):].mean(0)-np.sort(sim[~fg],axis=0)[-3:].mean(0)
            pb,pdiag=pc_box(z@proto,base);fb,fdiag=pc_box(contrast,base)
            boxes={'tp_box_control':box_of(base),'prototype_box':pb,'fgbg_box':fb}
            image=Image.open(q['target_image_path']).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
            state=processor.set_image(image);state['backbone_out'].update(visual)
            outputs={};masks=[]
            for name,box in boxes.items():
                st={'original_height':256,'original_width':256,'backbone_out':state['backbone_out']}
                score=0.;fallback=box is None
                if box is not None:
                    out=processor.add_geometric_prompt(box,True,st)
                    if out['scores'].numel():
                        j=int(out['scores'].argmax());m=out['masks'][j,0].cpu().numpy();score=float(out['scores'][j]);fallback=not bool(m.any())
                    else:fallback=True
                if fallback:m=base.copy()
                agree=dice(m,base);ratio=float((m.sum()+1)/(base.sum()+1));accept=bool(not fallback and agree>=.9 and .5<=ratio<=2 and score>=.5)
                guarded=m if accept else base
                dest=R/name/'masks';dest.mkdir(parents=True,exist_ok=True);mp=dest/(r['target_id'].replace('::','__')+'.png');Image.fromarray(m.astype('uint8')*255).save(mp)
                outputs[name]={'box_cxcywh':box,'mask_path':str(mp),'mask_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'score':score,'empty_fallback':fallback,'dice_with_TP':agree,'area_ratio_to_TP':ratio,'guard_accept':accept}
                masks.append((name,m,guarded))
            # Decisions and saved masks above are GT-free; evaluation starts here.
            gt=np.asarray(Image.open(q['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
            bd=dice(base,gt);assert abs(bd-r['dice'])<1e-12
            for name,m,guarded in masks:outputs[name].update(raw_dice=dice(m,gt),guarded_dice=dice(guarded,gt))
            row={'target_id':r['target_id'],'fold':folds[r['target_id']],'tp_route_id':r['route_id'],'tp_anchor_id':r['anchor_id'],'tp_dice':bd,'prompts':{'prototype':pdiag,'fgbg':fdiag},'outputs':outputs,
              'oracle_TP_plus_3_refined':max([bd]+[v['raw_dice'] for v in outputs.values()]),'oracle_full_TP_PC':max(pool[r['target_id']]),'oracle_full_TP_PC_plus_refined':max(pool[r['target_id']]+[v['raw_dice'] for v in outputs.values()])}
            with (R/'per_target.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            torch.cuda.synchronize();print('refined',len(done)+ni+1,'/',args.limit,'elapsed',round(time.time()-t0,1),flush=True)
rows=read(R/'per_target.jsonl')
if len(rows)==100:
    def stats(values,baseline):
        d=np.array(values)-np.array(baseline);rng=np.random.default_rng(2026);ci=np.quantile(d[rng.integers(0,100,(10000,100))].mean(1),[.025,.975]);fd=[float(np.mean([d[i] for i,r in enumerate(rows) if r['fold']==k])) for k in range(5)]
        return {'dice':float(np.mean(values)),'delta':float(d.mean()),'ci95':ci.tolist(),'wins_ties_losses':[int((d>1e-9).sum()),int((abs(d)<=1e-9).sum()),int((d< -1e-9).sum())],'fold_delta':fd,'passes_gate':bool(d.mean()>=.003 and ci[0]>0 and sum(x>0 for x in fd)>=3)}
    baseline=[r['tp_dice'] for r in rows];methods={}
    for name in config['variants']:
        for selection in ['raw','guarded']:methods[name+'_'+selection]=stats([r['outputs'][name][selection+'_dice'] for r in rows],baseline)
    vs_control=stats([r['outputs']['fgbg_box']['guarded_dice'] for r in rows],[r['outputs']['tp_box_control']['guarded_dice'] for r in rows])
    result={'count':100,'baseline_TP':float(np.mean(baseline)),'methods':methods,'primary_vs_TP_box_control':vs_control,
      'primary_passes':bool(methods['fgbg_box_guarded']['passes_gate'] and vs_control['passes_gate']),
      'oracles':{k:float(np.mean([r[k] for r in rows])) for k in ['oracle_TP_plus_3_refined','oracle_full_TP_PC','oracle_full_TP_PC_plus_refined']},
      'guard_accept_counts':{n:sum(r['outputs'][n]['guard_accept'] for r in rows) for n in config['variants']},'test_evaluated':False}
    assert abs(result['baseline_TP']-.8517552851672318)<1e-12
    save('results.json',result);(R/'VALIDATION_COMPLETE').touch()
    if not result['primary_passes']:save('TEST_NOT_RUN.json',{'reason':'Predeclared FG/BG guarded primary failed improvement over TP and matched TP-box decoder control.'})
    print(json.dumps(result),flush=True)
