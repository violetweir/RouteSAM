"""Student-aware candidate gain regression; validation-only nested grouped audit."""
from pathlib import Path
import json, collections
import numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'work/kvasir_b7_gap_20260908'
OLD=P/'work/rerun_c0_256_sam3knn_s256_base'
NEW=P/'work/kvasir_tp_student_mainline_20260907'
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
def resolve(p):
    p=Path(p);return p if p.is_absolute() else P/p
def mask(p):return np.asarray(Image.open(resolve(p)).convert('L'))>127
def dice(a,b):
    den=a.sum()+b.sum();return float(2*(a&b).sum()/den) if den else 1.
def boundary(m):
    inner=m.copy();inner[1:]&=m[:-1];inner[:-1]&=m[1:];inner[:,1:]&=m[:,:-1];inner[:,:-1]&=m[:,1:]
    return m & ~inner
save('gain_protocol.json',{'split':'validation only; script never reads test inputs',
 'candidate_pool':'All 14 TP+PC b0-b6; baseline new-X3 TP-B7; no candidate pruning',
 'target':'candidate Dice minus default TP-B7 Dice, validation labels only',
 'learner':'StandardScaler + Ridge; image-grouped nested 5 outer / 4 inner; seed2026',
 'alphas':[10.,100.],'thresholds':[0.,.02,.05],
 'inner_objective':'mean target Dice improvement, ties prefer fallback then larger threshold and alpha; fallback available',
 'outer_gate':{'mean_delta_at_least':.003,'paired_bootstrap_ci95_lower_gt':0,'positive_folds_at_least':3},
 'limitation':'Conditional selector CV only. Frozen X3 checkpoint and upstream pseudo-label router previously used all validation. This is not fully independent end-to-end OOF evidence.',
 'test_policy':'No automatic test evaluation even if gate passes; preserve report and frozen specification first'})
groups=collections.defaultdict(list)
for mode in ['sam3enc_anchor_conditioned_target_pooling','sam3enc_anchor_conditioned_patch_correspondence']:
    for row in read(OLD/f'stage1_feature_knn_b0_b6/{mode}/propagation_quality_validation/propagation_quality.jsonl'):
        row['pc']=int('patch_correspondence' in mode);groups[row['target_id']].append(row)
students={r['merged_id']:r for r in read(NEW/'predictions/X3_best/student_predictions_validation.jsonl')}
base={r['target_id']:r for r in read(NEW/'b7_validation/selected_masks.jsonl')}
feature_names=['q_return','q_model','q_soft','q_multi_TP','q_multi_PC','final_sam_score',
 'log_area_ratio_to_student','log_components','boundary_student_uncertainty','foreground_student_mean','background_student_mean']
X=[];Y=[];base_dice=[];routeids=[];targets=sorted(groups);baseline_indices=[]
for t in targets:
    rr=sorted(groups[t],key=lambda r:(r['pc'],r['bridge_count'],r['route_id']));assert len(rr)==14
    mm=[mask(r['forward_mask_path']) for r in rr];st=mask(students[t]['student_binary_mask'])
    raw=np.asarray(Image.open(resolve(students[t]['student_probability_map'])));assert raw.max()>255
    prob=raw.astype(np.float64)/65535.;unc=4*prob*(1-prob)
    pair=np.array([[dice(a,b) for b in mm] for a in mm]);features=[]
    for k,(r,m) in enumerate(zip(rr,mm)):
        bd=boundary(m)
        tp=[i for i in range(7) if i!=k];pc=[i for i in range(7,14) if i!=k]
        features.append([r['q_cycle'],dice(m,st),2*float((m*prob).sum())/max(m.sum()+prob.sum(),1e-12),
          float(pair[k,tp].mean()),float(pair[k,pc].mean()),r['final_sam_score'],
          float(np.log((m.sum()+1)/(st.sum()+1))),float(np.log1p(r['trace_component_final'])),
          float(unc[bd].mean()) if bd.any() else 1.,float(prob[m].mean()) if m.any() else 0.,float(prob[~m].mean()) if (~m).any() else 1.])
    f=np.array(features);b=next(i for i,r in enumerate(rr) if r['route_id']==base[t]['route_id']);baseline_indices.append(b)
    # Candidate/base differences plus default quality and candidate mode/bridge. No GT features.
    xx=np.concatenate([f-f[b],np.repeat(f[b:b+1],14,axis=0),np.array([[r['pc'],r['bridge_count']/6] for r in rr])],axis=1)
    assert np.isfinite(xx).all();X.append(xx);routeids.append([r['route_id'] for r in rr])
    gd=np.array([r['gt_dice_evaluation_only'] for r in rr]);Y.append(gd-gd[b]);base_dice.append(float(gd[b]))
X=np.array(X);Y=np.array(Y);base_dice=np.array(base_dice);baseline_indices=np.array(baseline_indices)
assert X.shape==(100,14,24)
save('gain_feature_names.json',[f'delta_{n}' for n in feature_names]+[f'default_{n}' for n in feature_names]+['candidate_PC','candidate_bridge_div6'])
def fit_predict(train,valid,alpha):
    # Exclude the default candidate, equal weight 1/13 per candidate gives unit weight per image.
    xx=np.concatenate([np.delete(X[i],baseline_indices[i],axis=0) for i in train]);yy=np.concatenate([np.delete(Y[i],baseline_indices[i]) for i in train])
    pipe=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    pipe.fit(xx,yy,ridge__sample_weight=np.full(len(yy),1/13))
    return pipe.predict(X[valid].reshape(-1,24)).reshape(-1,14)
def decide(pred,ids,threshold):
    pred=pred.copy();pred[np.arange(len(ids)),baseline_indices[ids]]=0.
    jj=pred.argmax(1);return np.where(pred[np.arange(len(ids)),jj]>threshold,jj,baseline_indices[ids])
chosen=baseline_indices.copy();folds=[]
outer=KFold(5,shuffle=True,random_state=2026)
for fold,(train,test) in enumerate(outer.split(targets)):
    configs=[{'alpha':None,'threshold':None,'mean_gain':0.}]
    for alpha in [10.,100.]:
        ip=np.zeros((len(train),14))
        for it,iv in KFold(4,shuffle=True,random_state=2026+fold).split(train): ip[iv]=fit_predict(train[it],train[iv],alpha)
        for threshold in [0.,.02,.05]:
            jj=decide(ip,train,threshold);configs.append({'alpha':alpha,'threshold':threshold,'mean_gain':float(Y[train,jj].mean())})
    cfg=max(configs,key=lambda c:(c['mean_gain'],c['alpha'] is None,c['threshold'] or 0,c['alpha'] or 0))
    if cfg['alpha'] is not None:chosen[test]=decide(fit_predict(train,test,cfg['alpha']),test,cfg['threshold'])
    gain=Y[test,chosen[test]]
    folds.append({'fold':fold,'inner_selected':cfg,'outer_count':len(test),'outer_delta':float(gain.mean()),'switches':int((chosen[test]!=baseline_indices[test]).sum()),'inner_configs':configs})
    print('fold',fold,'config',cfg,'outer gain',gain.mean(),flush=True)
gain=Y[np.arange(100),chosen];rng=np.random.default_rng(2026);boot=gain[rng.integers(0,100,(10000,100))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
passed=bool(gain.mean()>=.003 and lo>0 and sum(f['outer_delta']>0 for f in folds)>=3)
result={'baseline_dice':float(base_dice.mean()),'selector_conditional_oof_dice':float((base_dice+gain).mean()),'delta':float(gain.mean()),
 'paired_bootstrap_ci95':[float(lo),float(hi)],'win_tie_loss':[int((gain>1e-9).sum()),int((abs(gain)<=1e-9).sum()),int((gain< -1e-9).sum())],
 'switches':int((chosen!=baseline_indices).sum()),'positive_folds':sum(f['outer_delta']>0 for f in folds),'gate_passed':passed,
 'test_evaluated':False,'folds':folds}
save('gain_validation_results.json',result)
(R/'gain_validation_oof.jsonl').write_text(''.join(json.dumps({'target_id':t,'baseline_route':routeids[i][baseline_indices[i]],'selected_route':routeids[i][chosen[i]],'baseline_dice':float(base_dice[i]),'selected_dice':float(base_dice[i]+gain[i]),'delta':float(gain[i])})+'\n' for i,t in enumerate(targets)))
print(json.dumps({k:v for k,v in result.items() if k!='folds'}),flush=True)
