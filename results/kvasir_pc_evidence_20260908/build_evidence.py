from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import json,hashlib,argparse
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion,sobel
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');R=P/'work/kvasir_pc_evidence_20260908'
ap=argparse.ArgumentParser();ap.add_argument('--split',default='validation',choices=['validation','test']);args=ap.parse_args()
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def path(p):
    p=Path(p);return p if p.is_absolute() else P/p
def load(t):return np.load(R/'tokens'/(t.replace('::','__')+'.npy')).astype(np.float64)
def avg(x,w):return float(x@w/max(float(w.sum()),1e-12))
def topmean(x,k,axis):return np.sort(x,axis=axis).take(indices=range(x.shape[axis]-min(k,x.shape[axis]),x.shape[axis]),axis=axis).mean(axis=axis)
supports=read(P/'work/kvasir_1pct_anchors/protocol/support_manifest.jsonl')
anchors={}
for r in supports:
    fg=np.asarray(Image.open(r['frozen_mask_path']).convert('L').resize((18,18),Image.Resampling.NEAREST)).reshape(-1)>127
    assert fg.any() and (~fg).any();anchors[r['merged_id']]=(load(r['merged_id']),fg)
root=P/('work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6' if args.split=='validation' else 'work/rerun_kvasir_sam3base_test_20260906/quality_root')
groups={}
for mode in ['target_pooling','patch_correspondence']:
    for r in read(root/f'sam3enc_anchor_conditioned_{mode}/propagation_quality_{args.split}/propagation_quality.jsonl'):groups.setdefault(r['target_id'],[]).append(r)
schema={'proto':['top8','inside','inside_minus_ring','inside_std'],
 'fgbg':['inside_contrast','inside_minus_ring','positive_fraction','anchor_fg_coverage'],
 'mutual':['inside_reciprocal_fg','reciprocal_fg_minus_ring','mutual_contrast','mutual_fg_coverage'],
 'image':['color_separation','boundary_gradient','boundary_vs_ring_gradient','inside_gradient','ring_gradient','solidity_proxy']}
(R/'feature_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
result={}
for ni,(target,rr) in enumerate(sorted(groups.items())):
    token=load(target);image=np.asarray(Image.open(rr[0]['target_image_path']).convert('RGB').resize((256,256),Image.Resampling.BICUBIC),dtype=np.float64)/255
    gray=image.mean(2);gradient=np.hypot(sobel(gray,0),sobel(gray,1));gradient/=max(float(np.quantile(gradient,.95)),1e-8)
    masks={};imgfeatures={}
    for r in rr:
        m=np.asarray(Image.open(path(r['forward_mask_path'])).convert('L'))>127
        ring=binary_dilation(m,iterations=8)&~m;bd=m & ~binary_erosion(m,iterations=2)
        w=np.asarray(Image.fromarray(m.astype('uint8')*255).resize((18,18),Image.Resampling.BOX),dtype=np.float64).reshape(-1)/255
        rw=np.asarray(Image.fromarray(ring.astype('uint8')*255).resize((18,18),Image.Resampling.BOX),dtype=np.float64).reshape(-1)/255
        masks[r['route_id']]=(w,rw)
        c1=image[m].mean(0) if m.any() else np.zeros(3);c0=image[ring].mean(0) if ring.any() else np.zeros(3)
        coord=np.argwhere(m);bbox=np.prod(coord.max(0)-coord.min(0)+1) if len(coord) else 1
        ig=float(gradient[m].mean()) if m.any() else 0;rg=float(gradient[ring].mean()) if ring.any() else 0;bg=float(gradient[bd].mean()) if bd.any() else 0
        imgfeatures[r['route_id']]=[float(np.linalg.norm(c1-c0)),bg,bg-rg,ig,rg,float(m.sum()/bbox)]
    per_anchor={}
    for aid,(a,fg) in anchors.items():
        sim=a@token.T;proto=a[fg].mean(0);proto/=max(np.linalg.norm(proto),1e-12);raw=token@proto
        contrast=topmean(sim[fg],3,0)-topmean(sim[~fg],3,0)
        best_anchor=sim.argmax(0);back_target=sim.argmax(1);returned=back_target[best_anchor]
        coords=np.stack(np.unravel_index(np.arange(324),(18,18)),axis=1)
        dist2=((coords-coords[returned])**2).sum(1);reciprocal=np.exp(-dist2/2.)
        rfg=reciprocal*fg[best_anchor];rcontrast=contrast*reciprocal
        aa={}
        for rid,(w,rw) in masks.items():
            inside=avg(raw,w);v0=[float(np.sort(raw)[-8:].mean()),inside,inside-avg(raw,rw),float(np.sqrt(avg((raw-inside)**2,w)))]
            v1=[avg(contrast,w),avg(contrast,w)-avg(contrast,rw),avg((contrast>0).astype(float),w),float(w[back_target[fg]].mean())]
            v2=[avg(rfg,w),avg(rfg,w)-avg(rfg,rw),avg(rcontrast,w),float((w[back_target[fg]]*rfg[back_target[fg]]).mean())]
            aa[rid]={'proto':v0,'fgbg':v1,'mutual':v2}
        per_anchor[aid]=aa
    result[target]={'image':imgfeatures,'anchors':per_anchor}
    if ni%10==0:print('evidence',ni+1,'/',len(groups),flush=True)
assert len(result)==100
(R/f'evidence_{args.split}.json').write_text(json.dumps(result)+'\n')
print('COMPLETE',args.split,flush=True)
