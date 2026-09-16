from pathlib import Path
import json,os,hashlib,subprocess,shutil,time,collections,traceback,sys
import numpy as np
from PIL import Image
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');E=P/'new_project/experiments'
R=E/'busi_mask_prompt_ablation_20260913/mask_no_text';SOURCE=E/'isic2018_auto21_tp_validation_20260911'
D=Path('/Data_8TB/lht/MK-UNet/BUSI/BUSI_split')
PY='/home/violet/anaconda3/envs/sam3/bin/python';CPU='/home/violet/anaconda3/envs/mkunet_mamba/bin/python'
BASE='/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt';MODE='sam3enc_anchor_conditioned_target_pooling'
def save(p,a):Path(p).write_text(json.dumps(a,indent=2,ensure_ascii=False)+'\n')
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def jl(p,rows):Path(p).write_text(''.join(json.dumps(r)+'\n' for r in rows))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
 R.mkdir(exist_ok=False)
 for f in ['protocol','code','logs','selected/images','selected/masks','automatic']:(R/f).mkdir(parents=True,exist_ok=True)
 shutil.copy2(__file__,R/'pipeline.py')
 records=[];hashgroups=collections.defaultdict(list)
 for split,alias in [('train','train'),('val','validation'),('test','test')]:
  images=sorted((D/split/'images').glob('*.png'));masks=set((D/split/'masks').glob('*.png'));paired=set()
  for f in images:
   m=D/split/'masks'/(f.stem+'_mask.png');assert m.exists();paired.add(m)
   ident='BUSI::'+f.stem;records.append(dict(file_name=str(f),image_path=str(f),mask_file_name=str(m),mask_path=str(m),merged_dataset='BUSI',merged_id=ident,sample_id=f.stem,source_dataset='BUSI',split=alias,width=None,height=None))
   hashgroups[sha(f)].append(dict(id=ident,split=alias))
  assert paired==masks,(split,'unmatched masks')
 assert len({r['merged_id'] for r in records})==len(records)
 assert collections.Counter(r['split'] for r in records)==dict(train=517,validation=64,test=66)
 jl(R/'protocol/merged_manifest.jsonl',records)
 train=[dict(id=r['merged_id'],image_path=r['image_path'],split='train') for r in records if r['split']=='train'];save(R/'train_images_only.json',train)
 save(R/'dataset_audit.json',dict(split_counts=dict(collections.Counter(r['split'] for r in records)),image_mask_pairs_complete=True,exact_duplicate_groups=[g for g in hashgroups.values() if len(g)>1],cross_split_duplicates=[g for g in hashgroups.values() if len({r['split'] for r in g})>1],split_unchanged=True))
 save(R/'policy.json',dict(budget=5,train=517,actual_fraction=5/517,rounding='nearest integer of 1% train, max(1,round(N*.01))',seed=2026,selection='same train-only global_local_facility, RGB1008, 64 spatial patch tokens, 64-cluster dictionary, equal global/local similarity',knn_feature='SAM3-base patch_mean256',path_score='Target Pooling',beam=32,bridges=list(range(7)),canvas=256,selected_labels_only_after_freeze=True,train_hidden_masks_used=False,router='same legacy/relative-peer four configs and nested image-grouped CV; validation only',text_prompt='none; selected anchor GT tight box',no_student=True,no_lora=True,gpu=1))
 for fn in ['stage1_feature_knn_routes.py','eval_route_propagation_quality.py','run_t21_dynamic_pseudovideo.py','router.py']:shutil.copy2(SOURCE/'code'/fn,R/'code'/fn)
 extractor=(SOURCE/'code/isic_auto_extract.py').read_text().replace("os.environ['CUDA_VISIBLE_DEVICES']='0'","os.environ['CUDA_VISIBLE_DEVICES']='1'").replace('len(rows)==2075','len(rows)==517')
 (R/'code/selection_extract.py').write_text(extractor)
 router=(P/'new_project/automatic_tp_router.py').read_text()
 router=router.replace("R=E/'automatic_anchor_tp_router_20260913'","R=E/'busi_auto5_tp_router_20260913'")
 router=router.replace("SOURCES={'kvasir':'auto8_tp_validation_20260911','isic2018':'isic2018_auto21_tp_validation_20260911'}","SOURCES={'busi':'busi_auto5_tp_1pct_20260913'}")
 router=router.replace("E/SOURCES['kvasir']/'code/router.py'","E/SOURCES['busi']/'code/router.py'").replace("T/name/","E/source/")
 (R/'code/run_router.py').write_text(router)
 save(R/'code_hashes.json',{str(p):sha(p) for p in (R/'code').glob('*.py')});save(R/'status.json',dict(stage='prepared'))
def select():
 from sklearn.cluster import MiniBatchKMeans
 F=R/'selection';F.mkdir(exist_ok=False);rows=json.loads((R/'train_images_only.json').read_text());z=np.load(R/'selection_features.npz');assert z['ids'].tolist()==[r['id'] for r in rows]
 def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
 n=len(rows);g=norm(z['global_features'].astype(np.float32));sample=norm(z['sampled_patches'].astype(np.float32));assert g.shape==(517,1024) and sample.shape==(517,64,1024)
 hashes=[sha(r['image_path']) for r in rows];seen=set();eligible=[]
 for i,h in enumerate(hashes):
  if h not in seen:eligible.append(i);seen.add(h)
 km=MiniBatchKMeans(n_clusters=64,random_state=2026,n_init=3,batch_size=2048,max_iter=100).fit(sample.reshape(-1,1024))
 assignment=km.predict(sample.reshape(-1,1024)).reshape(n,64);h=np.stack([np.bincount(x,minlength=64) for x in assignment]).astype(np.float32)/64;idf=np.log((n+1)/(1+(h>0).sum(0)))+1;h=norm(np.sqrt(h*idf));similarity=(np.clip(g@g.T,0,1)+np.clip(h@h.T,0,1))*.5
 best=np.zeros(n,dtype=np.float32);chosen=[];trace=[]
 for k in range(5):
  gains=np.maximum(similarity-best[:,None],0).mean(0);allowed=np.zeros(n,dtype=bool);allowed[eligible]=True;allowed[chosen]=False;gains[~allowed]=-np.inf;j=int(np.argmax(gains));chosen.append(j);best=np.maximum(best,similarity[:,j]);trace.append(dict(k=k+1,id=rows[j]['id'],gain=float(gains[j]),coverage=float(best.mean())))
 ids=[rows[j]['id'] for j in chosen];save(F/'SELECTIONS_FROZEN.json',dict(time=time.time(),selected_ids=ids,method='global_local_facility',seed=2026,trace=trace,mask_pixels_read_before_freeze=0,train_only=True,eligible_count=len(eligible),features_sha256=sha(R/'selection_features.npz')))
 np.savez_compressed(F/'descriptors.npz',global_features=g,local_histogram=h,local_dictionary=km.cluster_centers_,idf=idf)
 byid={r['merged_id']:r for r in read(R/'protocol/merged_manifest.jsonl')};support=[];audit=[]
 for i,ident in enumerate(ids,1):
  row=byid[ident].copy();row.update(frozen_image_path=row['image_path'],frozen_mask_path=row['mask_path']);support.append(row)
  im=Image.open(row['image_path']).convert('RGB');m=np.array(Image.open(row['mask_path']).convert('L'));assert m.shape==np.asarray(im).shape[:2];assert m.max()>127 and (m>127).any(),('selected mask invalid',ident)
  imagefile=f'{i:02d}.png';im.save(R/'selected/images'/imagefile);shutil.copy2(row['mask_path'],R/'selected/masks'/imagefile)
  audit.append(dict(id=ident,foreground_fraction=float((m>127).mean()),mask_sha256=sha(row['mask_path']),image_sha256=sha(row['image_path'])))
 jl(R/'protocol/support_manifest.jsonl',support);save(R/'SELECTED_SUPPORT_FROZEN.json',dict(selected_ids=ids,selection_sha256=sha(F/'SELECTIONS_FROZEN.json'),audit=audit))
 cards=''.join(f'<article><h3>{i:02d} {a["id"]}</h3><p>标注面积占比 {a["foreground_fraction"]:.2%}</p><img src="images/{i:02d}.png"><img src="masks/{i:02d}.png"></article>' for i,a in enumerate(audit,1))
 (R/'selected/index.html').write_text('<!doctype html><meta charset="utf-8"><title>BUSI 自动5张参考图</title><style>body{font-family:system-ui;margin:24px;background:#eff3f7}article{background:white;padding:20px;margin:15px}img{width:44%;vertical-align:top;margin:1%}</style><h1>BUSI 1%：自动选择5张参考图</h1><p>仅训练RGB特征选图，名单冻结后读取标注。原划分不变。</p>'+cards)
 print('SELECTED',ids,flush=True)
def runstage(stage,cmd):
 save(R/'status.json',dict(stage=stage));env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONPATH='/Data_8TB/lht/sam3:'+str(P/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
 with (R/'logs'/f'{stage}.log').open('x') as f:
  p=subprocess.Popen(cmd,env=env,cwd=P,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);save(R/f'{stage}_process.json',dict(pid=p.pid,cmd=cmd));rc=p.wait()
 if rc:raise RuntimeError(f'{stage} failed {rc}')
def summarize():
 for split,n in [('validation',64),('test',66)]:
  q=R/f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl';rows=read(q);assert len(rows)==n*7 and all(r['status']=='success' for r in rows)
  save(R/f'{split}_PREDICTIONS_FROZEN.json',dict(time=time.time(),quality_sha256=sha(q)))
  groups=collections.defaultdict(list)
  for r in rows:groups[r['target_id']].append(r)
  scored=[]
  for t,rs in groups.items():
   assert sorted(r['bridge_count'] for r in rs)==list(range(7))
   gt=np.asarray(Image.open(rs[0]['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127;assert gt.any(),('empty GT',t)
   for r in rs:
    assert sha(r['forward_mask_path'])==r['forward_mask_sha256'];m=np.asarray(Image.open(r['forward_mask_path']).convert('L'))>127;i=int((m&gt).sum());total=int(m.sum())+int(gt.sum());r.update(dice=2*i/total if total else 1.,iou=i/(total-i) if total-i else 1.)
    scored.append({k:r[k] for k in ['target_id','route_id','bridge_count','dice','iou','q_cycle','forward_mask_path']})
  save(R/('automatic/per_candidate_metrics.json' if split=='test' else 'validation_candidate_metrics.json'),scored)
  fixed={str(b):dict(dice=float(np.mean([r['dice'] for r in rows if r['bridge_count']==b])),iou=float(np.mean([r['iou'] for r in rows if r['bridge_count']==b]))) for b in range(7)}
  save(R/f'{split}_results.json',dict(n=n,fixed_bridge=fixed,oracle=float(np.mean([max(r['dice'] for r in rs) for rs in groups.values()])),source_counts=dict(collections.Counter(r['anchor_id'] for r in rows))))
def main():
 runstage('selection_features',[PY,str(R/'code/selection_extract.py')]);runstage('automatic_selection',[CPU,str(R/'pipeline.py'),'select'])
 for split,n in [('validation',64),('test',66)]:
  runstage(split+'_routes',[PY,str(R/'code/stage1_feature_knn_routes.py'),'--mode',MODE,'--feature-source','sam3_base','--feature-size','256','--knn-feature','patch_mean','--beam-width','32','--min-bridge','0','--max-bridge','6','--split',split,'--protocol-root',str(R/'protocol'),'--output-root',str(R/'quality_root')])
  f=R/f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl';rows=read(f);assert len(rows)==n*7
  support={r['merged_id'] for r in read(R/'protocol/support_manifest.jsonl')};train={r['merged_id'] for r in read(R/'protocol/merged_manifest.jsonl') if r['split']=='train'}
  for row in rows:assert row['anchor_id'] in support and set(row['bridge_ids'])<=train and not row['target_gt_used_for_search_or_inference']
  save(R/(split+'_ROUTES_FROZEN.json'),dict(count=len(rows),sha256=sha(f)))
  runstage(split+'_propagation',[PY,str(R/'code/eval_route_propagation_quality.py'),'--checkpoint',BASE,'--mode',MODE,'--root',str(R/'quality_root'),'--split',split,'--canvas','256','--no-target-gt'])
 runstage('candidate_summary',[PY,str(R/'pipeline.py'),'summarize']);runstage('router',[PY,str(R/'code/run_router.py')]);save(R/'status.json',dict(stage='complete'));(R/'COMPLETE').touch()
if __name__=='__main__':
 try:
  action=sys.argv[1]
  if action=='prepare':prepare()
  elif action=='select':select()
  elif action=='summarize':summarize()
  else:main()
 except BaseException:
  if R.exists():save(R/'status.json',dict(stage='failed',error=traceback.format_exc()))
  raise