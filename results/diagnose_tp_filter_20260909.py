"""Fixed TP candidates/thresholds: select-then-filter vs filter-then-select."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
OLD=P/'work/kvasir_tp_student_mainline_20260907'
R=P/'work/kvasir_tp_pseudo_filter_20260909'

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def jsonl(p,rr):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
def clean(r):return {k:v for k,v in r.items() if not k.startswith('gt_') and 'evaluation_only' not in k}

def main():
    R.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('filter_router',OLD/'code/router.py');router=importlib.util.module_from_spec(spec);sys.modules[spec.name]=router;spec.loader.exec_module(router)
    def fit(rr):
        x=np.array([router.feature_vector(clean(r),False) for r in rr]);y=np.array([r['gt_dice_evaluation_only'] for r in rr])
        mu=x.mean(0);sd=np.maximum(x.std(0),1e-8);z=np.column_stack([np.ones(len(x)),(x-mu)/sd]);reg=np.eye(z.shape[1]);reg[0,0]=0
        w=np.linalg.solve(z.T@z+reg,z.T@y);return router.Ridge(mu.tolist(),sd.tolist(),w.tolist(),False)
    def pick(rr,model):return max(rr,key=lambda r:(model.score(clean(r)),-r['bridge_count'],r['route_id']))
    def decide(rr,model,qmulti):
        best=pick(rr,model);passing=[r for r in rr if r['q_cycle']>=.95]
        alternative=pick(passing,model) if passing else None
        oldpass=bool(qmulti>=.9 and best['q_cycle']>=.95);newpass=bool(qmulti>=.9 and alternative is not None)
        if oldpass:assert alternative['route_id']==best['route_id']
        return best,alternative,oldpass,newpass
    config=dict(date='2026-09-09',thresholds=dict(q_multi=.9,q_return=.95),scope='TP-only frozen b0-b6, no SAM3/student training, no test',
        baseline='Router picks best, then q_multi and chosen q_return must pass',
        proposed='Image q_multi unchanged; restrict candidates to q_return>=.95, then rank using same Router',
        validation='5-fold target-level OOF Router, seed2026, ridge1; quality of accepted subsets and rescued subset, not test performance',
        train='Use original frozen Router; target GT never read or used for selection',
        candidate_preview='Separate 580-row preview only; old 448-row pool and students untouched')
    save(R/'config.json',config)
    trainrows=read(OLD/'quality/anchor_conditioned_target_pooling/propagation_quality_train/propagation_quality.jsonl')
    assert all(not any(k.startswith('gt_') for k in r) for r in trainrows)
    train=router.grouped(trainrows);oldsel={r['target_id']:r for r in read(OLD/'train_top1_and_quality.jsonl')}
    model=router.Ridge(**json.loads((OLD/'router.json').read_text()))
    partitions=collections.Counter();preview=[];pertrain=[]
    for target,rr in sorted(train.items()):
        q=oldsel[target]['q_multi'];best,alt,op,np_=decide(rr,model,q)
        assert best['route_id']==oldsel[target]['route_id']
        part='both_pass' if op else 'multi_only_fail' if q<.9 and best['q_cycle']>=.95 else 'return_only_fail' if q>=.9 else 'both_fail'
        partitions[part]+=1
        pertrain.append(dict(target_id=target,partition=part,old_accepted=op,new_accepted=np_,old_route=best['route_id'],new_route=alt['route_id'] if np_ else None))
        if np_:
            preview.append(dict(target_id=target,pseudo_mask_path=alt['forward_mask_path'],q_multi=q,q_return=float(alt['q_cycle']),route_id=alt['route_id'],
                bridge_count=alt['bridge_count'],anchor_id=alt['anchor_id'],feature_mode='anchor_conditioned_target_pooling',sample_type='original',
                routeco_selected_score=model.score(clean(alt)),n_candidates=7,preview_only=True))
    q=np.array([r['q_multi'] for r in preview]);lo,hi=q.min(),q.max()
    for r in preview:r['explicit_quality_weight']=float(np.clip((r['q_multi']-lo)/max(hi-lo,1e-12),.2,1))
    assert len(preview)==580 and sum(r['old_accepted'] for r in pertrain)==448
    jsonl(R/'train_preview_manifest.jsonl',preview);jsonl(R/'train_selection_diagnostic.jsonl',pertrain)
    valrows=read(OLD/'quality/anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl');val=router.grouped(valrows)
    mq={r['target_id']:r['q_multi'] for r in read(OLD/'validation_top1_and_quality.jsonl')}
    prior={r['target_id']:r['tp_route_id'] for r in read(P/'work/kvasir_rethink_20260908/anchor_factorial/B/selection/validation_oof.jsonl')}
    ids=sorted(val);order=np.random.default_rng(2026).permutation(ids);folds={str(t):i%5 for i,t in enumerate(order)};perval=[]
    for fold in range(5):
        tr=[r for r in valrows if folds[r['target_id']]!=fold];model=fit(tr)
        for target in ids:
            if folds[target]!=fold:continue
            best,alt,op,np_=decide(val[target],model,mq[target]);assert best['route_id']==prior[target]
            perval.append(dict(target_id=target,fold=fold,old_accepted=op,new_accepted=np_,old_route=best['route_id'],new_route=alt['route_id'] if np_ else None,
                old_dice_evaluation_only=best['gt_dice_evaluation_only'],new_dice_evaluation_only=alt['gt_dice_evaluation_only'] if np_ else None))
    assert abs(np.mean([r['old_dice_evaluation_only'] for r in perval])-.8517552851672318)<1e-12
    def summary(rr,key):
        v=np.array([r[key] for r in rr],dtype=float)
        return dict(count=len(rr),mean_dice=float(v.mean()) if len(v) else None,median_dice=float(np.median(v)) if len(v) else None,
            below_05=int((v<.5).sum()),below_08=int((v<.8).sum()),minimum=float(v.min()) if len(v) else None)
    existing=[r for r in perval if r['old_accepted']];expanded=[r for r in perval if r['new_accepted']];rescued=[r for r in expanded if not r['old_accepted']]
    result=dict(train=dict(partitions=dict(partitions),original_accepted=448,new_accepted=580,newly_accepted=132,original_masks_preserved=True),
        validation=dict(original_pool=summary(existing,'old_dice_evaluation_only'),expanded_pool=summary(expanded,'new_dice_evaluation_only'),
            newly_accepted_new_choice=summary(rescued,'new_dice_evaluation_only'),newly_accepted_previous_choice=summary(rescued,'old_dice_evaluation_only')),
        no_test=True,no_training=True,validation_oof_original_routes_verified=100,
        interpretation='Train coverage increase is deterministic under unchanged thresholds, not proof of true pseudo-label correctness. Subset Dice is not full-validation or test Dice. Pool expansion requires rebuilding S3, committee and students before downstream comparisons.')
    jsonl(R/'validation_oof_diagnostic.jsonl',perval);save(R/'results.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
