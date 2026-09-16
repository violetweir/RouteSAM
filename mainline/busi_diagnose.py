from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import json,hashlib,collections,numpy as np
from PIL import Image
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');E=P/'new_project/experiments';B=E/'busi_auto5_tp_1pct_20260913';R=E/'busi_failure_diagnosis_20260913';RAW=Path('/Data_8TB/lht/MK-UNet/BUSI/Dataset_BUSI_with_GT')
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,a):Path(p).write_text(json.dumps(a,indent=2,ensure_ascii=False)+'\n')
def mask(p,size=None):
 im=Image.open(p).convert('L');im=im.resize((size,size),Image.Resampling.NEAREST) if size else im;return np.array(im)>127
def dice(a,b):return float(2*(a&b).sum()/max(a.sum()+b.sum(),1))
def main():
 R.mkdir(exist_ok=True);records=read(B/'protocol/merged_manifest.jsonl');support=read(B/'protocol/support_manifest.jsonl');byid={r['merged_id']:r for r in records}
 # Offline label integrity audit only; these masks are not inputs to selection/model fitting.
 audit=[];bad=[];extras=[];resize=[]
 for r in records:
  p=Path(r['image_path']);cls=p.stem.split(' ')[0];raw=RAW/cls/p.name;mr=RAW/cls/(p.stem+'_mask.png');im=Image.open(p);ma=Image.open(r['mask_path']);m=mask(r['mask_path']);ext=sorted((RAW/cls).glob(p.stem+'_mask_*.png'))
  if im.size!=ma.size:bad.append(dict(id=r['merged_id'],issue='shape_mismatch'))
  if not m.any():bad.append(dict(id=r['merged_id'],issue='empty_mask'))
  if mr.exists() and not np.array_equal(mask(mr),m):bad.append(dict(id=r['merged_id'],issue='split_vs_raw_main_diff'))
  if ext:
   union=m.copy()
   for f in ext:union|=mask(f)
   extras.append(dict(id=r['merged_id'],split=r['split'],extra_masks=[str(f) for f in ext],additional_pixels=int((union&~m).sum()),main_vs_union_dice=dice(m,union)))
  if r['split'] in ['validation','test']:
   roundtrip=Image.fromarray(mask(r['mask_path'],256).astype(np.uint8)*255).resize(im.size,Image.Resampling.NEAREST);resize.append(dict(id=r['merged_id'],split=r['split'],roundtrip_dice=dice(m,np.array(roundtrip)>127)))
  audit.append(dict(id=r['merged_id'],split=r['split'],cls=cls,shape=list(im.size),area=float(m.mean()),fg_patches18=int(mask(r['mask_path'],18).sum())))
 save(R/'data_integrity.json',dict(n=len(records),bad=bad,multimask=extras,roundtrip=resize,records=audit))
 z=np.load(B/'quality_root/features/sam3_base_s256_features.npz');cond=z['cond_target'].astype(float);ix={r['merged_id']:i for i,r in enumerate(records)};train=[i for i,r in enumerate(records) if r['split']=='train'];anchors=z['anchor_ids'].tolist();ranking=[]
 for ai,a in enumerate(anchors):
  rr=next(x for x in audit if x['id']==a);row=dict(anchor=a,area=rr['area'],fg_patches18=rr['fg_patches18'],train_mean=float(cond[ai,train].mean()),train_std=float(cond[ai,train].std()))
  for split in ['validation','test']:
   ids=[i for i,r in enumerate(records) if r['split']==split];row[split+'_mean']=float(cond[ai,ids].mean());row[split+'_wins']=int((cond[:,ids].argmax(0)==ai).sum())
  ranking.append(row)
 # train-only score centering illustration, no rerouting or segmentation tuning
 mean=cond[:,train].mean(1,keepdims=True);std=np.maximum(cond[:,train].std(1,keepdims=True),1e-8);normed=(cond-mean)/std;centered=cond-mean
 illustrated={}
 for split in ['validation','test']:
  ids=[i for i,r in enumerate(records) if r['split']==split];illustrated[split]={k:dict(collections.Counter(anchors[j] for j in x[:,ids].argmax(0))) for k,x in [('raw',cond),('centered',centered),('zscore',normed)]}
 save(R/'score_bias.json',dict(anchors=ranking,b0_anchor_assignment=illustrated,warning='centering/zscore only diagnose score offsets; not evaluated segmentation improvement'))
 out={}
 for split in ['validation','test']:
  q=read(B/f'quality_root/sam3enc_anchor_conditioned_target_pooling/propagation_quality_{split}/propagation_quality.jsonl');scored=json.loads((B/('validation_candidate_metrics.json' if split=='validation' else 'automatic/per_candidate_metrics.json')).read_text());metric={r['route_id']:r for r in scored};groups=collections.defaultdict(list)
  for r in q:r.update(metric[r['route_id']]);groups[r['target_id']].append(r)
  rr=E/'busi_auto5_tp_router_20260913/busi/selected_router/per_target.json';router={r['target_id']:r for r in json.loads(rr.read_text())} if split=='test' else {}
  per=[]
  for t,rs in groups.items():
   info=next(x for x in audit if x['id']==t);best=max(rs,key=lambda r:r['dice']);per.append(dict(info,oracle=best['dice'],b0=next(r['dice'] for r in rs if r['bridge_count']==0),router=router[t]['dice'] if t in router else None,oracle_route_id=best['route_id'],empty_all=all(not mask(r['forward_mask_path']).any() for r in rs)))
  def agg(a):return dict(n=len(a),oracle=float(np.mean([r['oracle'] for r in a])),b0=float(np.mean([r['b0'] for r in a])),router=float(np.mean([r['router'] for r in a])) if router else None,oracle_below02=sum(r['oracle']<.2 for r in a),oracle_below05=sum(r['oracle']<.5 for r in a),area_median=float(np.median([r['area'] for r in a])))
  out[split]=dict(overall=agg(per),classes={c:agg([r for r in per if r['cls']==c]) for c in ['benign','malignant']},by_size={name:agg(a) for name,a in [('lt5pct',[r for r in per if r['area']<.05]),('ge5pct',[r for r in per if r['area']>=.05])] if a},per_target=per,cycle_dice_correlation=float(np.corrcoef([r['q_cycle'] for r in q],[r['dice'] for r in q])[0,1]))
 save(R/'error_groups.json',out)
 print('SCORES',json.dumps(ranking));print('DATA',json.dumps(dict(bad=bad,multimask=extras)));print('ERROR',json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='per_target'} for k,v in out.items()}))
if __name__=='__main__':main()