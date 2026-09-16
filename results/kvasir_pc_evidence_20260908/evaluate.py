from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import sys,json,importlib.util,collections
import numpy as np
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');R=P/'work/kvasir_pc_evidence_20260908'
sys.path.insert(0,str(R/'legacy_code'))
import run_candidate_rerank as cr
g=cr.g;g.R=R
spec=importlib.util.spec_from_file_location('router',R/'legacy_code/router.py');g.ROUTER=importlib.util.module_from_spec(spec);spec.loader.exec_module(g.ROUTER)
FAMILIES=['legacy','pc_proto','pc_fg_bg','pc_mutual','image_region']
E={};FAMILY='legacy';original_examples=cr.examples
def extra(target,anchor,route,family):
    if family=='legacy':return []
    e=E[target]
    if family=='image_region':return e['image'][route]
    a=e['anchors'][anchor][route]
    if family=='pc_proto':return a['proto']
    if family=='pc_fg_bg':return a['fgbg']
    if family=='pc_mutual':return a['fgbg']+a['mutual']
    raise ValueError(family)
def examples(train,held,split='validation'):
    base=original_examples(train,held,split);rr=[]
    for r in base:
        target=r['target_id'];t=next(x for x in g.DATA[split][target][g.MODES[0]] if x['route_id']==r['tp_route_id'])
        tf=np.array(extra(target,t['anchor_id'],r['tp_route_id'],FAMILY));pf=np.array(extra(target,t['anchor_id'],r['pc_route_id'],FAMILY))
        rr.append({**r,'features':r['features']+tf.tolist()+(pf-tf).tolist()})
    return rr
cr.examples=examples;g.examples=examples
def brief(rr):
    s=cr.stats(rr);s['fold_delta']=[float(np.mean([r['delta'] for r in rr if r['outer_fold']==i])) for i in range(5)]
    s['passes_gate']=bool(s['mean_delta']>=.003 and s['paired_bootstrap_95ci'][0]>0 and sum(d>0 for d in s['fold_delta'])>=3)
    return s
def independent_pc(ids):
    # New evidence evaluated against historical PC-only router on unchanged PC7 pool.
    def rows(part):return [r for t in part for r in g.DATA['validation'][t][g.MODES[1]]]
    def fit(part,family,lam):
        rr=rows(part)
        x=[g.ROUTER.feature_vector(g.clean(r),False)+extra(r['target_id'],r['anchor_id'],r['route_id'],family) for r in rr]
        return g.linear_fit(x,[r['gt_dice_evaluation_only'] for r in rr],lam)
    def select(part,family,model):
        out=[]
        for t in sorted(part):
            rr=g.DATA['validation'][t][g.MODES[1]]
            score=lambda r:g.predict(model,g.ROUTER.feature_vector(g.clean(r),False)+extra(t,r['anchor_id'],r['route_id'],family))
            chosen=max(rr,key=lambda r:(score(r),-r['bridge_count'],r['route_id']))
            out.append({'target_id':t,'route_id':chosen['route_id'],'dice':chosen['gt_dice_evaluation_only']})
        return out
    allrows=[];familyrows=collections.defaultdict(list)
    for fold,held in enumerate(g.split_folds(ids,5,2026)):
        train=sorted(set(ids)-set(held));options=[]
        for family in FAMILIES:
            scores=[]
            for lam in ([1.] if family=='legacy' else [1.,10.]):
                ds=[]
                for ih in g.split_folds(train,4,2027):ds+=select(ih,family,fit(sorted(set(train)-set(ih)),family,lam))
                scores.append({'family':family,'ridge':lam,'inner_dice':float(np.mean([r['dice'] for r in ds]))})
            cfg=max(scores,key=lambda r:(r['inner_dice'],r['ridge']));options.append(cfg)
            selected=select(held,family,fit(train,family,cfg['ridge']))
            familyrows[family]+=[{**r,'outer_fold':fold} for r in selected]
        cfg=max(options,key=lambda r:(r['inner_dice'],-FAMILIES.index(r['family']),r['ridge']))
        allrows += [{**r,'selected_family':cfg['family']} for r in familyrows[cfg['family']] if r['outer_fold']==fold]
        print('PC-only fold',fold,cfg,flush=True)
    baseline={r['target_id']:r['dice'] for r in familyrows['legacy']}
    def st(rr):
        ds=np.array([r['dice']-baseline[r['target_id']] for r in rr]);rng=np.random.default_rng(2026);ci=np.quantile(ds[rng.integers(0,100,(10000,100))].mean(1),[.025,.975])
        folds=[float(np.mean([r['dice']-baseline[r['target_id']] for r in rr if r['outer_fold']==f])) for f in range(5)]
        return {'dice':float(np.mean([r['dice'] for r in rr])),'delta_vs_PC':float(ds.mean()),'ci95':ci.tolist(),'fold_delta':folds,
          'passes_gate':bool(ds.mean()>=.003 and ci[0]>0 and sum(d>0 for d in folds)>=3)}
    result={'primary_nested_family_selection':st(allrows),'ablations':{f:st(rr) for f,rr in familyrows.items()},'test_evaluated':False}
    assert abs(result['ablations']['legacy']['dice']-.8173293125085627)<1e-10
    g.save(R/'pc_only_results.json',result);g.jsonl(R/'pc_only_oof.jsonl',allrows)
    return result

g.save(R/'protocol.json',{'stage':'Evaluate correspondence evidence, then auxiliary role; fixed original candidates, no new propagation or student',
 'families':FAMILIES,'PC_changes':'prototype localization; top-3 FG minus top-3 BG cosine; bidirectional nearest-patch cycle with exp(-grid_distance_squared/2)',
 'non_PC_control':'RGB region/ring separation, Sobel boundary and region gradients, bbox occupancy',
 'PC_self':'Candidate-own anchor evidence appended to original PC router features; nested 5/4 target folds, family and ridge selected inside inner folds',
 'TP_aux':'Fixed independent TP expert per held fold; candidate evidence under the same TP-chosen anchor; default evidence and candidate-minus-default difference appended to historical 15 features',
 'candidate_set_aux':'Fixed selected TP plus all seven original PC masks; no TP reranking; masks are proposals, auxiliary evidence need not be PC',
 'aux_validation':'Nested 5 outer / 4 policy / 3 expert-training folds; families, ridge10/100 and threshold.02/.05/.10 selected inside inner validation; fallback included',
 'test_gate':'Primary nested family-selection gain >=.003, paired bootstrap95 lower>0, at least3 positive outer folds; component ablation peaks cannot trigger test',
 'original_split':'800/100/100;8/792 unchanged',
 'scope_limit':'No claim of regenerated PC routes, spatial refinement or improved candidate oracle; validation reused across research rounds',
 'test_evaluated':False})
E=json.loads((R/'evidence_validation.json').read_text());g.load('validation');ids=sorted(g.DATA['validation'])
pc=independent_pc(ids)
primary=[];ablation=collections.defaultdict(list)
for fold,held in enumerate(g.split_folds(ids,5,2026)):
    train=sorted(set(ids)-set(held));policies={}
    for family in FAMILIES:
        FAMILY=family;pol=g.train_policy(train,f'{family}_outer{fold}');policies[family]=pol
        rr=cr.decisions(pol['model'],pol['threshold'],examples(train,held))
        ablation[family]+=[{**r,'outer_fold':fold} for r in rr]
    family=max(FAMILIES,key=lambda f:(policies[f]['mean_delta'],-policies[f]['switches'],-FAMILIES.index(f)))
    primary += [{**r,'selected_family':family} for r in ablation[family] if r['outer_fold']==fold]
    print('TP-aux fold',fold,'family',family,'delta',np.mean([r['delta'] for r in primary if r['outer_fold']==fold]),flush=True)
result={'primary_nested_family_selection':brief(primary),'ablations':{f:brief(rr) for f,rr in ablation.items()},'test_evaluated':False}
assert abs(result['primary_nested_family_selection']['tp_dice']-.8517552851672318)<1e-10
assert abs(result['ablations']['legacy']['rerank_dice']-.8516389616341843)<1e-10
g.save(R/'aux_results.json',result);g.jsonl(R/'aux_oof.jsonl',primary)
for f,rr in ablation.items():g.jsonl(R/f'aux_{f}_oof.jsonl',rr)
g.save(R/'fold_provenance.json',g.AUDIT)
assert all(not set(x['expert_train_ids'])&set(x['held_ids']) for x in g.AUDIT)
g.save(R/'validation_results.json',{'pc_self':pc,'tp_aux':result})
print('RESULT',json.dumps({'pc_self':pc,'tp_aux':result}),flush=True)
if not pc['primary_nested_family_selection']['passes_gate'] and not result['primary_nested_family_selection']['passes_gate']:
    g.save(R/'TEST_NOT_RUN.json',{'reason':'Both predeclared primary nested family-selection checks failed; retain frozen TP.'})
(R/'VALIDATION_COMPLETE').touch()
