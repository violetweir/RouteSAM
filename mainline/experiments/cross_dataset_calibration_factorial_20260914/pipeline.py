"""Kvasir/ISIC2018：迁移BUSI固定的分数校准×top1/top2消融。"""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import sys,json,time,hashlib,shutil,subprocess,collections,importlib.util,traceback
import numpy as np

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');E=P/'new_project/experiments'
G=P/'new_project/reproduction_guides/automatic_selection_20260914'
ROOT=E/'cross_dataset_calibration_factorial_20260914'
MODE='sam3enc_anchor_conditioned_target_pooling';PY='/home/violet/anaconda3/envs/sam3/bin/python'
GROUPS=['raw_top1','centered_top1','raw_top2','centered_top2','original_per_bridge']
SPECS={
 'kvasir':dict(ntrain=800,nval=100,ntest=100,k=8,val='auto8_tp_validation_20260911',legacy_oof=.8513170057291387,legacy_test=.8713744761396023),
 'isic2018':dict(ntrain=2075,nval=259,ntest=260,k=21,val='isic2018_auto21_tp_validation_20260911',legacy_oof=.8623961438691973,legacy_test=.8665732155247388)}
def rdir(ds):return ROOT/ds
def base(ds):return E/'automatic_anchor_tp_test_20260913'/ds
def old(ds,split):return E/SPECS[ds]['val'] if split=='validation' else base(ds)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def jl(p,rr):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def status(ds,stage,**kw):save(rdir(ds)/'status.json',dict(dataset=ds,stage=stage,time=time.time(),**kw));print(ds,stage,kw,flush=True)
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def core(ds):
 c=module('factorial_'+ds,rdir(ds)/'code/factorial_core.py');c.R=rdir(ds);return c
def qp(root,split):return root/f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl'
def rp(root,split):return root/f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl'

def prepare(ds):
 r=rdir(ds);s=SPECS[ds];r.mkdir(parents=True,exist_ok=False)
 for d in ['code','protocol','logs','snapshots','quality_root/features']:(r/d).mkdir(parents=True)
 shutil.copy2(__file__,r/'pipeline.py')
 for fn in ['stage1_feature_knn_routes.py','eval_route_propagation_quality.py','run_t21_dynamic_pseudovideo.py']:
  shutil.copy2(base(ds)/'code'/fn,r/'code'/fn)
 shutil.copy2(E/'auto8_tp_validation_20260911/code/router.py',r/'code/router.py')
 shutil.copy2(E/'busi_calibration_factorial_20260914/pipeline.py',r/'code/factorial_core.py')
 shutil.copy2(E/'busi_calibrated_multi_anchor_20260913/pipeline.py',r/'code/previous_pipeline.py')
 for fn in ['merged_manifest.jsonl','support_manifest.jsonl']:shutil.copy2(base(ds)/'protocol'/fn,r/'protocol'/fn)
 shutil.copy2(base(ds)/'quality_root/features/sam3_base_s256_features.npz',r/'quality_root/features/sam3_base_s256_features.npz')
 shutil.copy2(E/'automatic_anchor_tp_router_20260913'/ds/'folds_frozen.json',r/'folds_frozen.json')
 save(r/'predefined_policy.json',dict(dataset=ds,counts=s,groups=GROUPS,
  calibration='subtract per-anchor mean TP over all own training RGBs; no hidden training masks',
  ranking='raw TP(anchor,target) or centered TP, descending; anchor ID ascending on ties',
  pool='top1/top2 references fixed per target for all b0-b6, plus original per-bridge control',
  router='same legacy28 Ridge alpha1 refit separately for each pool; same frozen image-grouped 5-fold validation; no new hyperparameter search',
  invariants='same automatic 8/21 anchors, split, SAM3-base, box, no text, canvas256, patch_mean KNN, beam32, original raw per-anchor path objective',
  gt='validation GT for fitting only; freeze all test choices before test scoring; test previously seen',
  control='legacy Router refitted on original automatic candidates; ISIC historical selected rank_peer is reported separately, not substituted for the legacy control',
  propagation='serial GPU stages with free-memory guard; do not stop other experiments',
  reuse='original candidates first, then stable snapshots of matching P1/P2 masks; identical propagation code + route IDs + anchor/bridge/box + mask hashes required',
  floats='retain original route fields for historical route IDs to preserve baseline Router input exactly; log overlap route checks'))
 inputs={str(f):sha(f) for f in list((r/'code').glob('*.py'))+list((r/'protocol').glob('*'))+[r/'pipeline.py',r/'folds_frozen.json',r/'quality_root/features/sam3_base_s256_features.npz']}
 save(r/'input_hashes.json',inputs);status(ds,'prepared')

def build_routes(ds):
 r=rdir(ds);s=SPECS[ds]
 if (r/'ROUTES_COMPLETE').exists():return
 status(ds,'building_routes_cpu')
 code=module('route_source_'+ds,r/'code/stage1_feature_knn_routes.py')
 records=read(r/'protocol/merged_manifest.jsonl');support=read(r/'protocol/support_manifest.jsonl')
 z=np.load(r/'quality_root/features/sam3_base_s256_features.npz');aids=z['anchor_ids'].tolist();pm=z['patch_mean'];cond=z['cond_target']
 assert len(aids)==s['k'] and len(records)==s['ntrain']+s['nval']+s['ntest']
 index={x['merged_id']:i for i,x in enumerate(records)};train=[i for i,x in enumerate(records) if x['split']=='train'];assert len(train)==s['ntrain']
 mu=cond[:,train].astype(np.float64).mean(1);scores={};rankings={}
 for row in records:
  tid=row['merged_id'];j=index[tid];values=cond[:,j].astype(np.float64);center=values-mu
  raw=sorted(range(len(aids)),key=lambda a:(-float(values[a]),aids[a]));cal=sorted(range(len(aids)),key=lambda a:(-float(center[a]),aids[a]))
  rankings[tid]=dict(raw=[aids[a] for a in raw],centered=[aids[a] for a in cal])
  scores[tid]={aids[a]:dict(anchor_target_raw=float(values[a]),anchor_target_centered=float(center[a]),anchor_train_mean=float(mu[a]),calibrated_anchor_rank=cal.index(a)) for a in range(len(aids))}
 save(r/'calibration_frozen.json',dict(time=time.time(),anchor_ids=aids,train_means=mu.tolist(),rankings=rankings,scores=scores,hidden_train_GT_read=False))
 fullbank=G/'P2_fullbank_20260914'/ds
 havebank=all(rp(fullbank,sp).exists() for sp in ['validation','test'])
 state=None;cache=None;anchors=None
 if not havebank:
  sim=pm@pm.T
  state=dict(mode=MODE,patch_mean=pm,sim=sim,knn_sim=sim,knn_feature='patch_mean',cond_scores=cond,id_to_anchor={a:i for i,a in enumerate(aids)},id_to_index=index,text_blend=0.)
  anchors={a['anchor_id']:a for a in code.t21.human_pool(support,512)}
  cache=code.build_rank_cache(state,records,support,train)
 members={};audits={}
 for split,n in [('validation',s['nval']),('test',s['ntest'])]:
  oldrows=read(rp(old(ds,split),split));orig={x['route_id']:x for x in oldrows};origgroups=collections.defaultdict(list)
  for row in oldrows:origgroups[row['target_id']].append(row)
  bank={}
  if havebank:
   bank={(x['target_id'],x['anchor_id'],x['bridge_count']):x for x in read(rp(fullbank,split))}
  tgts=sorted([x for x in records if x['split']==split],key=lambda x:x['merged_id']);assert len(tgts)==n
  groups={g:{} for g in GROUPS};routes={};overlap=0
  for no,t in enumerate(tgts,1):
   tid=t['merged_id'];j=index[tid];required=set(rankings[tid]['raw'][:2]+rankings[tid]['centered'][:2]);byab={}
   for aid in sorted(required):
    if havebank:
     candidates=[dict(bank[(tid,aid,b)]) for b in range(7)]
     for rt in candidates:
      for f,v in scores[tid][aid].items():assert abs(rt[f]-v)<1e-10,(ds,split,tid,aid,f)
    else:
     beams=[([],code.route_score(state,aid,[],j))];forbidden={j,index[aid]};candidates=[]
     for b in range(7):
      if b:
       expanded=[]
       for path,_ in beams:
        tail=j if not path else path[0]
        for node in code.top_ranked_nodes(cache,state,aid,tail,forbidden|set(path),32):
         path2=[node,*path];expanded.append((path2,code.route_score(state,aid,path2,j)))
       beams=sorted(expanded,key=lambda item:item[1],reverse=True)[:32]
      path,score=max(beams,key=lambda item:item[1]);candidates.append(code.make_route(t,b,anchors[aid],path,score,records))
    for row in candidates:
     match=[x for x in origgroups[tid] if x['anchor_id']==aid and x['bridge_count']==row['bridge_count']]
     if match:
      assert len(match)==1 and match[0]['route_id']==row['route_id'],(ds,split,tid,aid,'original path changed')
      row=dict(match[0]);overlap+=1
     row.update(scores[tid][aid]);routes[row['route_id']]=row;byab[(aid,row['bridge_count'])]=row
   for kind in ['raw','centered']:
    for k in [1,2]:groups[f'{kind}_top{k}'][tid]=[byab[(aid,b)]['route_id'] for aid in rankings[tid][kind][:k] for b in range(7)]
   rows=sorted(origgroups[tid],key=lambda x:x['bridge_count']);assert len(rows)==7
   groups['original_per_bridge'][tid]=[x['route_id'] for x in rows]
   for row in rows:
    row=dict(row);row.update(scores[tid][row['anchor_id']]);routes[row['route_id']]=row
   if no%20==0:status(ds,'building_routes_cpu',split=split,targets=no,total=n,routes=len(routes))
  jl(rp(r,split),list(routes.values()));members[split]=groups
  audits[split]=dict(targets=n,candidate_union=len(routes),old_routes=len(oldrows),overlapping_rebuilt_old_routes_exact=overlap,fullbank_route_reuse=havebank)
 save(r/'pool_membership_frozen.json',dict(time=time.time(),groups=members,test_GT_read=False))
 save(r/'route_audit.json',audits);(r/'ROUTES_COMPLETE').touch();status(ds,'routes_complete',audit=audits)

def stable_rows(p):
 for attempt in range(10):
  try:
   stat=p.stat();raw=p.read_bytes();stat2=p.stat()
   if (stat.st_size,stat.st_mtime_ns)!=(stat2.st_size,stat2.st_mtime_ns):continue
   return [json.loads(l) for l in raw.decode().splitlines() if l.strip()],hashlib.sha256(raw).hexdigest()
  except (json.JSONDecodeError,UnicodeDecodeError):time.sleep(.2)
 raise RuntimeError('Cannot obtain consistent snapshot: '+str(p))

def seed(ds,split):
 r=rdir(ds);dest=qp(r,split)
 if (r/f'{split}_PROPAGATION_COMPLETE').exists():return
 routes={x['route_id']:x for x in read(rp(r,split))};existing={x['route_id']:x for x in read(dest)} if dest.exists() else {}
 sources=[old(ds,split),G/'P1_pilot_20260914'/ds,G/'P2_fullbank_20260914'/ds];snapshots=[]
 for no,src in enumerate(sources):
  p=qp(src,split)
  if not p.exists():continue
  if no:
   for fn in ['eval_route_propagation_quality.py','run_t21_dynamic_pseudovideo.py']:
    assert sha(src/'code'/fn)==sha(r/'code'/fn),(src,fn,'different propagation code')
  rows,h=stable_rows(p);snapshot=r/'snapshots'/f'{split}_source{no}.jsonl'
  jl(snapshot,rows);snapshots.append(dict(source=str(p),read_sha256=h,snapshot=str(snapshot),snapshot_sha256=sha(snapshot)))
  for q in rows:
   rid=q['route_id']
   if rid not in routes or rid in existing or q.get('status')!='success':continue
   rt=routes[rid]
   for key in ['anchor_id','bridge_ids','anchor_mask_sha256','anchor_box_xywh_normalized']:assert q[key]==rt[key],(src,rid,key)
   mp=Path(q['forward_mask_path']);mp=mp if mp.is_absolute() else P/mp
   assert sha(mp)==q['forward_mask_sha256'];assert not q['target_gt_used_for_search_or_inference']
   existing[rid]={**{k:v for k,v in q.items() if not k.startswith('gt_')},**rt,'forward_mask_path':str(mp)}
 jl(dest,[existing[k] for k in sorted(existing)])
 save(r/f'{split}_reuse_audit.json',dict(candidate_union=len(routes),reused=len(existing),missing=len(routes)-len(existing),snapshots=snapshots))
 status(ds,'candidates_seeded',split=split,total=len(routes),reused=len(existing),missing=len(routes)-len(existing))

def validate(ds):
 r=rdir(ds);s=SPECS[ds]
 if (r/'models_frozen.json').exists():return
 status(ds,'fitting_validation_routers_cpu');c=core(ds);m=c.prev();data,values=c.extract('validation',True)
 ids=data[GROUPS[0]]['ids'];assert len(ids)==s['nval'];mapping=json.loads((r/'folds_frozen.json').read_text());folds=np.array([mapping[t] for t in ids]);ix=np.arange(len(ids))
 models={};summary={};vectors={};selections={}
 for name,d in data.items():
  config=(d['raw'].shape[1]//7,'legacy',1.);pred=m.cv(d,ix,folds,config);choice=pred.argmax(1);y=d['y'][ix,choice]
  models[name]=m.fit(d,ix,config);vectors[name]=y.tolist();oracle=float(d['y'].max(1).mean())
  rs=[d['rows'][i][j] for i,j in enumerate(choice)]
  selections[name]=[dict(target_id=x['target_id'],route_id=x['route_id'],dice=values[x['route_id']]['dice']) for x in rs]
  summary[name]=dict(candidates_per_target=d['raw'].shape[1],oof_dice=float(y.mean()),oracle_dice=oracle,oracle_gap=oracle-float(y.mean()),selected_anchors=dict(collections.Counter(x['anchor_id'] for x in rs)))
 assert abs(summary['original_per_bridge']['oof_dice']-s['legacy_oof'])<1e-9,summary['original_per_bridge']
 save(r/'validation_results.json',dict(groups=summary,paired_oof=c.paired(vectors),per_target_oof=selections))
 save(r/'models_frozen.json',dict(time=time.time(),models=models,folds_sha256=sha(r/'folds_frozen.json'),membership_sha256=sha(r/'pool_membership_frozen.json'),test_GT_read=False))
 status(ds,'validation_complete',results=summary)

def evaluate(ds):
 r=rdir(ds);s=SPECS[ds];c=core(ds);m=c.prev();status(ds,'freezing_test_choices_without_GT')
 data,_=c.extract('test',False);ix=np.arange(s['ntest']);frozen=json.loads((r/'models_frozen.json').read_text());choices={}
 for name,d in data.items():
  assert len(d['ids'])==len(ix);js=m.predict(d,ix,frozen['models'][name]).argmax(1)
  choices[name]=[d['rows'][i][j]['route_id'] for i,j in enumerate(js)]
 save(r/'test_choices_frozen.json',dict(time=time.time(),choices=choices,models_sha256=sha(r/'models_frozen.json'),test_GT_read=False))
 status(ds,'scoring_frozen_test_choices');_,values=c.extract('test',True);metrics={};vectors={};per={}
 for name,d in data.items():
  lookup={x['route_id']:x for rr in d['rows'] for x in rr};rs=[lookup[k] for k in choices[name]]
  per[name]=[dict(target_id=x['target_id'],route_id=x['route_id'],anchor_id=x['anchor_id'],bridge_count=x['bridge_count'],mask_path=x['forward_mask_path'],**values[x['route_id']]) for x in rs]
  y=[x['dice'] for x in per[name]];vectors[name]=y;oracle=float(np.mean([max(values[x['route_id']]['dice'] for x in rr) for rr in d['rows']]))
  metrics[name]=dict(candidates_per_target=d['raw'].shape[1],dice=float(np.mean(y)),iou=float(np.mean([x['iou'] for x in per[name]])),oracle_dice=oracle,oracle_gap=oracle-float(np.mean(y)),
    selected_anchors=dict(collections.Counter(x['anchor_id'] for x in rs)),selected_bridges=dict(collections.Counter(x['bridge_count'] for x in rs)),candidate_anchors=dict(collections.Counter(x['anchor_id'] for rr in d['rows'] for x in rr)),
    fixed_rank1_b0_b6=[float(np.mean([values[rr[b]['route_id']]['dice'] for rr in d['rows']])) for b in range(7)],
    historical_candidate_seconds_per_target_mean=float(np.mean([sum(float(x['seconds']) for x in rr) for rr in d['rows']])))
  dest=r/name/'masks';dest.mkdir(parents=True,exist_ok=True)
  for x in rs:shutil.copy2(x['forward_mask_path'],dest/(x['target_id'].replace('::','__')+'.png'))
 assert abs(metrics['original_per_bridge']['dice']-s['legacy_test'])<1e-9,metrics['original_per_bridge']
 val=json.loads((r/'validation_results.json').read_text());out=dict(dataset=ds,counts=s,validation={k:v for k,v in val.items() if k!='per_target_oof'},test=metrics,paired_test=c.paired(vectors))
 save(r/'results.json',out);save(r/'test_per_target.json',per);save(r/'test_candidate_metrics.json',values)
 for f,h in json.loads((r/'input_hashes.json').read_text()).items():assert sha(f)==h,(f,'changed')
 save(r/'completion_audit.json',dict(all_test_included=True,same_alpha_and_folds=True,original_legacy_reproduced=True,test_choices_frozen_before_GT_read=True,inputs_unchanged=True))
 lines=[f'# {ds}：校准 × top1/top2 消融','',f"固定自动{s['k']}张参考，原划分，SAM3-base，box无文本，256，同一legacy28 Ridge(alpha=1)。",'',
  '| 组别 | 候选数 | val OOF | test Dice | Oracle | Oracle差距 |','|---|---:|---:|---:|---:|---:|']
 for name,x in metrics.items():lines.append(f"| {name} | {x['candidates_per_target']} | {val['groups'][name]['oof_dice']:.6f} | {x['dice']:.6f} | {x['oracle_dice']:.6f} | {x['oracle_gap']:.6f} |")
 lines+=['','四组预先固定，test此前已用于诊断；不是按本轮test重新调参。配对区间、源参考占比和逐图记录见JSON。Oracle仅作上界诊断，候选数量分别报告。']
 (r/'report.md').write_text('\n'.join(lines)+'\n');(r/'COMPLETE').touch();status(ds,'complete',results=metrics)

def run(ds):
 r=rdir(ds)
 if (r/'COMPLETE').exists():return
 build_routes(ds)
 for split in ['validation','test']:
  seed(ds,split);c=core(ds);m=c.prev()
  wanted={x['route_id'] for x in read(rp(r,split))};ready=read(qp(r,split))
  if {x['route_id'] for x in ready}==wanted and len(ready)==len(wanted):
   assert all(x['status']=='success' for x in ready)
   save(r/f'{split}_PREDICTIONS_FROZEN.json',dict(time=time.time(),candidates=len(ready),quality_sha256=sha(qp(r,split)),all_predictions_reused=True))
   (r/f'{split}_PROPAGATION_COMPLETE').touch()
  else:m.propagate(split)
  if split=='validation':validate(ds)
 evaluate(ds)

def run_all():
 for ds in ['isic2018','kvasir']:
  save(ROOT/'status.json',dict(stage='running',dataset=ds,time=time.time()))
  # A separate prepare/build process may have created the routes already.
  while not (rdir(ds)/'ROUTES_COMPLETE').exists():
   st=json.loads((rdir(ds)/'status.json').read_text())
   if st.get('stage')=='failed':raise RuntimeError((ds,st))
   time.sleep(10)
  run(ds)
 save(ROOT/'results.json',{ds:json.loads((rdir(ds)/'results.json').read_text()) for ds in SPECS})
 (ROOT/'COMPLETE').touch();save(ROOT/'status.json',dict(stage='complete',time=time.time()))

if __name__=='__main__':
 try:
  action=sys.argv[1]
  if action=='run-all':run_all()
  else:
   ds=sys.argv[2];assert ds in SPECS
   {'prepare':prepare,'routes':build_routes,'run':run,'validate':validate,'evaluate':evaluate}[action](ds)
 except BaseException:
  if len(sys.argv)>2 and sys.argv[2] in SPECS and rdir(sys.argv[2]).exists():status(sys.argv[2],'failed',error=traceback.format_exc())
  elif ROOT.exists():save(ROOT/'status.json',dict(stage='failed',error=traceback.format_exc()))
  raise
