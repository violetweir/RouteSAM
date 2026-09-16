from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');R=P/'work/kvasir_pc_prompt_refine_native1008_20260908'
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def path(p):
    p=Path(p);return p if p.is_absolute() else P/p
def mask(p):return np.asarray(Image.open(path(p)).convert('L'))>127
def dice(a,b):
    n=a.sum()+b.sum();return float(2*(a&b).sum()/n) if n else 1.
rows=read(R/'per_target.jsonl');assert len(rows)==100 and len({r['target_id'] for r in rows})==100
quality=read(P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6/sam3enc_anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl');qi={r['route_id']:r for r in quality}
checks=[];prompt_diag=[];metrics={}
for r in rows:
    q=qi[r['tp_route_id']];base=mask(q['forward_mask_path']);gt=np.asarray(Image.open(q['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
    assert abs(dice(base,gt)-r['tp_dice'])<1e-12
    for name,v in r['outputs'].items():
        m=mask(v['mask_path']);assert m.shape==(256,256);assert hashlib.sha256(Path(v['mask_path']).read_bytes()).hexdigest()==v['mask_sha256'];assert abs(dice(m,gt)-v['raw_dice'])<1e-12
        guarded=m if v['guard_accept'] else base;assert abs(dice(guarded,gt)-v['guarded_dice'])<1e-12
        dest=R/name/'guarded_masks';dest.mkdir(parents=True,exist_ok=True);mp=dest/(r['target_id'].replace('::','__')+'.png')
        shutil.copy2(path(v['mask_path'] if v['guard_accept'] else q['forward_mask_path']),mp)
        checks.append({'target_id':r['target_id'],'variant':name,'guard_accept':v['guard_accept'],'final_mask_path':str(mp),'sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'dice':v['guarded_dice']})
        b=v['box_cxcywh']
        if b is not None:
            cx,cy,w,h=b;assert 0<=cx-w/2<=cx+w/2<=1+1e-12 and 0<=cy-h/2<=cy+h/2<=1+1e-12
            bm=np.zeros((256,256),bool);x0=max(0,int(np.floor((cx-w/2)*256)));x1=min(256,int(np.ceil((cx+w/2)*256)));y0=max(0,int(np.floor((cy-h/2)*256)));y1=min(256,int(np.ceil((cy+h/2)*256)));bm[y0:y1,x0:x1]=True
            prompt_diag.append({'target_id':r['target_id'],'variant':name,'gt_coverage':float((bm&gt).sum()/max(gt.sum(),1)),'box_area_ratio':float(bm.mean()),'gt_fraction_inside_box':float((bm&gt).sum()/max(bm.sum(),1))})
        for mode,pred in [('raw',m),('guarded',guarded)]:
            key=name+'_'+mode;inter=int((pred&gt).sum());union=int((pred|gt).sum());metrics.setdefault(key,[]).append({'dice':dice(pred,gt),'iou':inter/union if union else 1.})
summary={'masks_verified':300,'guarded_masks_exported':300,'mask_dice_recomputed':True,'box_coordinates_verified':True,'test_evaluated':False,
 'methods':{k:{j:float(np.mean([r[j] for r in rr])) for j in ['dice','iou']} for k,rr in metrics.items()},
 'prompt_diagnostics_GT_only':{n:{k:float(np.mean([r[k] for r in prompt_diag if r['variant']==n])) for k in ['gt_coverage','box_area_ratio','gt_fraction_inside_box']} for n in rows[0]['outputs']},
 'runtime':'GPU native1008 bfloat16 autocast, input-output256, 23percent allocator memory cap',
 'source_files_unmodified':'Native GPU SAM3 inference; no shared source modifications.',
 'final_code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')}}
(R/'completion_audit.json').write_text(json.dumps(summary,indent=2)+'\n');(R/'guarded_selected_masks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in checks));(R/'prompt_diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prompt_diag))
print(json.dumps(summary,indent=2))
