from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np
from PIL import Image
import json,hashlib,importlib.util,sys,collections,shutil,time,traceback
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E=P/'new_project/experiments'
T=E/'automatic_anchor_tp_test_20260913'
R=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/reproduction_guides/automatic_selection_20260914/verify_router_busi')
MODE='sam3enc_anchor_conditioned_target_pooling'
SOURCES={'busi':'busi_auto5_tp_1pct_20260913'}
CONFIGS=[('legacy',1.),('rank_peer',1.),('rank_peer',10.),('rank_peer',100.)]
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,a):Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resolve(p):return Path(p) if Path(p).is_absolute() else P/p
def mask(p):return np.array(Image.open(resolve(p)).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
def dice(a,b):
 n=int(a.sum())+int(b.sum());return 2*int((a&b).sum())/n if n else 1.
def cname(c):return f'{c[0]}_a{c[1]:g}'
spec=importlib.util.spec_from_file_location('legacy',E/SOURCES['busi']/'code/router.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
def extract(rows,validation):
 groups=legacy.grouped(rows);ids=sorted(groups);raw=[];extended=[];ys=[];ious=[];ordered=[];hashes={}
 for no,t in enumerate(ids):
  rs=sorted(groups[t],key=lambda r:r['bridge_count']);assert [r['bridge_count'] for r in rs]==list(range(7)) and all(r['status']=='success' for r in rs)
  masks=[]
  for row in rs:
   assert not row['target_gt_used_for_search_or_inference']
   f=resolve(row['forward_mask_path']);h=sha(f);assert h==row['forward_mask_sha256'];hashes[str(f)]=h;masks.append(mask(f))
  x=np.array([legacy.feature_vector(row,False) for row in rs],dtype=float)
  peer=np.array([[dice(a,b) for b in masks] for a in masks]);others=peer[~np.eye(7,dtype=bool)].reshape(7,6)
  areas=np.array([m.mean() for m in masks]);med=np.median(areas)
  extras=np.column_stack([others.mean(1),others.min(1),others.max(1),others.std(1),areas,np.abs(areas-med),np.log((areas+1e-5)/(med+1e-5)),others.mean(1)*x[:,7]])
  raw.append(x);extended.append(np.column_stack([x,extras]));ordered.append(rs)
  if validation:
   gtpath=rs[0]['target_mask_path_evaluation_only'];gt=mask(gtpath);hashes[str(resolve(gtpath))]=sha(resolve(gtpath));y=[dice(m,gt) for m in masks];ys.append(y)
  if no%50==0:print('features',validation,no,len(ids),flush=True)
 return dict(ids=ids,raw=np.array(raw),extended=np.array(extended),y=np.array(ys) if validation else None,rows=ordered,hashes=hashes)
def array(data,kind):
 x=data['raw'] if kind=='legacy' else data['extended'];return x if kind=='legacy' else x-x.mean(1,keepdims=True)
def fit(data,indices,c):
 kind,alpha=c;x=array(data,kind)[indices].reshape(-1,array(data,kind).shape[-1]);y=data['y'][indices].copy()
 if kind!='legacy':y-=y.mean(1,keepdims=True)
 y=y.ravel();mean=x.mean(0);std=np.maximum(x.std(0),1e-8);z=(x-mean)/std;ym=y.mean();w=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-ym))
 return dict(kind=kind,alpha=alpha,means=mean.tolist(),stds=std.tolist(),weights=w.tolist(),intercept=float(ym))
def predict(data,indices,m):
 x=array(data,m['kind'])[indices];return (x-np.array(m['means']))/np.array(m['stds'])@np.array(m['weights'])+m['intercept']
def score(y,s):return float(y[np.arange(len(y)),s.argmax(1)].mean())
def cv(data,indices,folds,c):
 out=np.empty((len(indices),7));local=folds[indices]
 for f in sorted(set(local)):
  train=indices[local!=f];test=indices[local==f];out[local==f]=predict(data,test,fit(data,train,c))
 return out
def evaluate(data,choices,metrics):
 selected=[]
 for i,b in enumerate(choices):
  row=data['rows'][i][int(b)];m=metrics[row['route_id']];selected.append(dict(target_id=data['ids'][i],bridge_count=int(b),route_id=row['route_id'],dice=m['dice'],iou=m['iou'],mask_path=row['forward_mask_path']))
 return dict(dice=float(np.mean([r['dice'] for r in selected])),iou=float(np.mean([r['iou'] for r in selected])),histogram=dict(collections.Counter(int(b) for b in choices)),per_target=selected)
def run():
 R.mkdir(exist_ok=False);shutil.copy2(__file__,R/'run.py');save(R/'predefined_policy.json',dict(configs=CONFIGS,selection='mean selected validation Dice in frozen image-grouped 5-fold CV; outer nested CV evaluates config selection',test='evaluate only after model and all test route choices frozen',seed=2026,features='legacy 28 numeric features; rank_peer adds 8 mask-consensus features and centers X and y within each image',oracle='diagnostic only',prior_test_seen=True,no_test_tuning=True))
 allout={}
 for name,source in SOURCES.items():
  d=R/name;d.mkdir();save(R/'status.json',dict(dataset=name,stage='validation_features'))
  vf=E/source/f'quality_root/{MODE}/propagation_quality_validation/propagation_quality.jsonl';tf=E/source/f'quality_root/{MODE}/propagation_quality_test/propagation_quality.jsonl'
  v=extract(read(vf),True);n=len(v['ids']);indices=np.arange(n)
  if name=='isic2018':mapping=json.loads((E/source/'validation_folds_frozen.json').read_text())['target_to_fold']
  else:
   shuffled=np.random.default_rng(2026).permutation(v['ids']);mapping={str(t):i%5 for i,t in enumerate(shuffled)}
  folds=np.array([mapping[t] for t in v['ids']]);save(d/'folds_frozen.json',mapping)
  scores={cname(c):cv(v,indices,folds,c) for c in CONFIGS};cvvalues={k:score(v['y'],s) for k,s in scores.items()};winner=max(CONFIGS,key=lambda c:cvvalues[cname(c)])
  nested=np.empty((n,7));outer=[]
  for f in range(5):
   tr=indices[folds!=f];te=indices[folds==f];inner={cname(c):score(v['y'][tr],cv(v,tr,folds,c)) for c in CONFIGS};chosen=max(CONFIGS,key=lambda c:inner[cname(c)])
   nested[te]=predict(v,te,fit(v,tr,chosen));outer.append(dict(fold=f,chosen=cname(chosen),inner_scores=inner))
  models={'baseline_legacy':fit(v,indices,CONFIGS[0]),'selected_router':fit(v,indices,winner)}
  # Verify legacy implementation equivalence on validation (same standardization and ridge objective).
  vr=[]
  for i,rs in enumerate(v['rows']):
   for b,row in enumerate(rs):vr.append(dict(row,gt_dice_evaluation_only=float(v['y'][i,b])))
  old=legacy.Ridge.fit(vr,1.,False);oldpred=np.array([[old.score(r) for r in rs] for rs in v['rows']]);diff=float(np.max(np.abs(oldpred-predict(v,indices,models['baseline_legacy']))));assert diff<1e-6,diff
  fixed=int(v['y'].mean(0).argmax());save(d/'models_frozen.json',dict(models=models,winner=cname(winner),fixed_bridge_selected_on_validation=fixed,cv_scores=cvvalues,nested_cv_dice=score(v['y'],nested),outer=outer,legacy_equivalence_max_abs=diff,time=time.time()))
  save(d/'validation_results.json',dict(n=n,cv_scores=cvvalues,selected_config=cname(winner),selected_config_oof_dice=cvvalues[cname(winner)],nested_selection_oof_dice=score(v['y'],nested),oracle=float(v['y'].max(1).mean()),fixed_bridge_means=v['y'].mean(0).tolist()))
  save(R/'status.json',dict(dataset=name,stage='test_features_no_gt'))
  test=extract(read(tf),False);assert not(set(test['ids'])&set(v['ids']));ti=np.arange(len(test['ids']))
  choices={k:predict(test,ti,m).argmax(1).tolist() for k,m in models.items()}
  choices.update(fixed_val_best=[fixed]*len(ti),cycle_only=np.array([[float(r['q_cycle'] or 0) for r in rs] for rs in test['rows']]).argmax(1).tolist(),peer_consensus=test['extended'][:,:,28].argmax(1).tolist())
  save(d/'test_choices_frozen.json',dict(time=time.time(),models_sha256=sha(d/'models_frozen.json'),choices=choices,target_ids=test['ids'],route_ids={k:[test['rows'][i][b]['route_id'] for i,b in enumerate(bs)] for k,bs in choices.items()}))
  # Candidate test metrics were computed in the previous experiment, never supplied to fitting or choices.
  mf=E/source/'automatic/per_candidate_metrics.json';metricrows=json.loads(mf.read_text());metrics={r['route_id']:r for r in metricrows}
  results={k:evaluate(test,bs,metrics) for k,bs in choices.items()};oracle=float(np.mean([max(metrics[r['route_id']]['dice'] for r in rs) for rs in test['rows']]))
  for k,res in results.items():
   res['oracle_gap']=oracle-res['dice'];dest=d/k/'masks';dest.mkdir(parents=True)
   save(d/k/'per_target.json',res.pop('per_target'))
   for i,b in enumerate(choices[k]):shutil.copy2(test['rows'][i][b]['forward_mask_path'],dest/(test['ids'][i].replace('::','__')+'.png'))
  rng=np.random.default_rng(2026);paired={}
  for ref in ['baseline_legacy','fixed_val_best']:
   delta=np.array([metrics[test['rows'][i][choices['selected_router'][i]]['route_id']]['dice']-metrics[test['rows'][i][choices[ref][i]]['route_id']]['dice'] for i in ti]);boot=delta[rng.integers(0,len(ti),(10000,len(ti)))].mean(1);paired[ref]=dict(delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
  out=dict(dataset=name,n_validation=n,n_test=len(ti),selected_config=cname(winner),validation=json.loads((d/'validation_results.json').read_text()),test=results,test_oracle=oracle,paired=paired)
  save(d/'results.json',out);allout[name]=out
  hashes={**v['hashes'],**test['hashes'],str(vf):sha(vf),str(tf):sha(tf),str(mf):sha(mf)};save(d/'input_hashes.json',hashes)
  for f,h in hashes.items():assert sha(f)==h
  save(d/'completion_audit.json',dict(validation_image_groups=n,test_image_groups=len(ti),all_test_images_included=True,legacy_equivalence_max_abs=diff,predictions_unchanged=True,models_and_choices_frozen_before_test_metric_load=True))
  (d/'COMPLETE').touch();print(json.dumps(out,ensure_ascii=False),flush=True)
 save(R/'results.json',allout);save(R/'status.json',dict(stage='complete'));(R/'COMPLETE').touch()
if __name__=='__main__':
 try:run()
 except BaseException:
  if R.exists():save(R/'status.json',dict(stage='failed',error=traceback.format_exc()))
  raise