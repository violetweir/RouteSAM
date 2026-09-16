"""Nested target-level evaluation of a TP-preserving, PC gain gate."""
import collections
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'work/kvasir_tp_pc_gain_gate_20260907'
VAL=P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6'
TEST=P/'work/rerun_kvasir_sam3base_test_20260906/quality_root'
BASE=P/'work/rerun_kvasir_sam3base_test_20260906/final_masks_no_student'
MODES=['anchor_conditioned_target_pooling','anchor_conditioned_patch_correspondence']
FEATURES=['tp_cycle','cycle_difference','log_area_change_difference','centroid_change_difference','empty_count_difference','log_component_difference','bridge_difference','tp_score_margin','pc_score_margin','tp_mask_consensus','pc_mask_consensus','between_mask_dice','log_area_ratio','tp_sam_score','sam_score_difference']
PARAMS=[(lam,t) for lam in [10.,100.] for t in [.02,.05,.10]]
CACHE={};DATA={};ROUTER=None;AUDIT=[]

def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def jsonl(p,rr):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rr))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(r):return {k:v for k,v in r.items() if not k.startswith('gt_') and 'evaluation_only' not in k}
def mask(p):
    p=Path(p);p=p if p.is_absolute() else P/p
    x=np.asarray(Image.open(p).convert('L'))>127
    assert x.shape==(256,256)
    return x
def dice(a,b):
    n=int(a.sum())+int(b.sum())
    return 2*int(np.logical_and(a,b).sum())/n if n else 1.
def split_folds(ids,n,seed):
    order=np.random.default_rng(seed).permutation(sorted(ids))
    return [sorted(str(x) for x in order[k::n]) for k in range(n)]
def linear_fit(x,y,lam):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    mu=x.mean(0);sd=np.maximum(x.std(0),1e-8)
    z=np.column_stack([np.ones(len(x)),(x-mu)/sd]);reg=np.eye(z.shape[1])*lam;reg[0,0]=0
    w=np.linalg.solve(z.T@z+reg,z.T@y)
    return {'means':mu.tolist(),'stds':sd.tolist(),'weights':w.tolist()}
def predict(model,x):
    z=(np.asarray(x)-model['means'])/model['stds']
    return float(model['weights'][0]+z@np.asarray(model['weights'][1:]))

def load(split):
    if split in DATA:return DATA[split]
    out={};inputs={}
    for mode in MODES:
        path=(VAL if split=='validation' else TEST)/('sam3enc_'+mode)/f'propagation_quality_{split}/propagation_quality.jsonl'
        rr=read(path);inputs[str(path)]=sha(path)
        assert len(rr)==700 and all(r['status']=='success' for r in rr)
        by=ROUTER.grouped([{**r,'feature_mode':mode} for r in rr]);assert len(by)==100
        for target,group in by.items():
            group.sort(key=lambda r:r['bridge_count']);assert [r['bridge_count'] for r in group]==list(range(7))
            masks=[mask(r['forward_mask_path']) for r in group]
            digests=[hashlib.sha256(m.tobytes()).hexdigest() for m in masks]
            representatives=list(dict.fromkeys(digests))
            for i,r in enumerate(group):
                others=[digests.index(h) for h in representatives if h!=digests[i]]
                r['_consensus']=float(np.mean([dice(masks[i],masks[j]) for j in others])) if others else 1.
                r['_pred_area']=int(masks[i].sum());r['_mask']=masks[i]
            out.setdefault(target,{})[mode]=group
    protocol=read(P/'work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl')
    expected={r['merged_id'] for r in protocol if r['split']==split}
    assert set(out)==expected and all(len(v)==2 for v in out.values())
    DATA[split]=out;save(R/f'{split}_input_sha256.json',inputs)
    return out

def expert_models(train):
    result={}
    for mode in MODES:
        rr=[r for i in sorted(train) for r in DATA['validation'][i][mode]]
        fitted=linear_fit([ROUTER.feature_vector(clean(r),False) for r in rr],[r['gt_dice_evaluation_only'] for r in rr],1.)
        result[mode]=ROUTER.Ridge(**fitted,include_mode=False)
    return result

def choice(model,rr):
    # Model receives only the historical, GT-free feature vector.
    scored=[(model.score(clean(r)),-r['bridge_count'],r['route_id'],i) for i,r in enumerate(rr)]
    ranked=sorted(scored,reverse=True);i=ranked[0][-1]
    return rr[i],float(ranked[0][0]-ranked[1][0])

def pair_features(t,p,tm,pm):
    def val(r,k):return float(r.get(k) or 0.)
    f=[val(t,'q_cycle'),val(p,'q_cycle')-val(t,'q_cycle'),
       np.log1p(val(p,'trace_area_max_rel_delta'))-np.log1p(val(t,'trace_area_max_rel_delta')),
       val(p,'trace_centroid_max_step')-val(t,'trace_centroid_max_step'),
       val(p,'trace_empty_count')-val(t,'trace_empty_count'),
       np.log1p(val(p,'trace_component_final'))-np.log1p(val(t,'trace_component_final')),
       val(p,'bridge_count')-val(t,'bridge_count'),tm,pm,t['_consensus'],p['_consensus'],dice(t['_mask'],p['_mask']),
       np.log((p['_pred_area']+1)/(t['_pred_area']+1)),val(t,'final_sam_score'),val(p,'final_sam_score')-val(t,'final_sam_score')]
    assert len(f)==len(FEATURES) and np.isfinite(f).all()
    return [float(x) for x in f]

def examples(train,held,split='validation'):
    assert set(train).isdisjoint(held)
    key=(tuple(sorted(train)),tuple(sorted(held)),split)
    if key in CACHE:return CACHE[key]
    models=expert_models(train);rr=[]
    for target in sorted(held):
        t,tm=choice(models[MODES[0]],DATA[split][target][MODES[0]])
        p,pm=choice(models[MODES[1]],DATA[split][target][MODES[1]])
        f=pair_features(clean(t),clean(p),tm,pm)
        # Serialize GT-free selection evidence before attaching quality labels.
        e={'target_id':target,'features':f,'tp_route_id':t['route_id'],'pc_route_id':p['route_id'],'tp_mask_path':t['forward_mask_path'],'pc_mask_path':p['forward_mask_path']}
        if split=='validation':e.update(tp_dice=t['gt_dice_evaluation_only'],pc_dice=p['gt_dice_evaluation_only'],gain=p['gt_dice_evaluation_only']-t['gt_dice_evaluation_only'])
        rr.append(e)
    AUDIT.append({'expert_train_ids':sorted(train),'held_ids':sorted(held),'split':split})
    CACHE[key]=rr;return rr

def crossfit(ids,n,seed):
    rows=[]
    for held in split_folds(ids,n,seed):rows+=examples(sorted(set(ids)-set(held)),held)
    assert len(rows)==len(ids) and {r['target_id'] for r in rows}==set(ids)
    return sorted(rows,key=lambda r:r['target_id'])

def gate_fit(rr,lam):return linear_fit([r['features'] for r in rr],[r['gain'] for r in rr],lam)
def decisions(model,threshold,rr):
    result=[]
    for r in rr:
        pred=predict(model,r['features']) if model is not None else None
        use_pc=bool(pred is not None and pred>threshold)
        e={**r,'predicted_gain':pred,'use_pc':use_pc,'selected_route_id':r['pc_route_id'] if use_pc else r['tp_route_id'],'selected_mask_path':r['pc_mask_path'] if use_pc else r['tp_mask_path']}
        if 'gain' in r:e.update(dice=r['pc_dice'] if use_pc else r['tp_dice'],delta=r['gain'] if use_pc else 0.)
        result.append(e)
    return result

def train_policy(ids,tag):
    # Hyperparameter validation stays entirely inside the current outer train set.
    tested={p:[] for p in PARAMS}
    inner_folds=split_folds(ids,4,2027)
    for fold,held in enumerate(inner_folds):
        tr=sorted(set(ids)-set(held));train_rows=crossfit(tr,3,2028);held_rows=examples(tr,held)
        for lam in [10.,100.]:
            gate=gate_fit(train_rows,lam)
            for threshold in [.02,.05,.10]:
                tested[lam,threshold]+= [{**r,'inner_fold':fold} for r in decisions(gate,threshold,held_rows)]
    scores=[]
    for (lam,t),rr in tested.items():
        scores.append({'ridge':lam,'threshold':t,'mean_delta':float(np.mean([r['delta'] for r in rr])),'switches':sum(r['use_pc'] for r in rr)})
    # Include exact TP fallback, prefer fewer switches/stronger shrinkage on ties.
    options=scores+[{'ridge':None,'threshold':None,'mean_delta':0.,'switches':0}]
    best=max(options,key=lambda x:(x['mean_delta'],-x['switches'],x['ridge'] or float('inf'),x['threshold'] or float('inf')))
    policy={**best,'model':gate_fit(crossfit(ids,4,2027),best['ridge']) if best['ridge'] is not None else None}
    save(R/'policies'/f'{tag}.json',{'training_target_ids':sorted(ids),'candidates':scores,'selected':policy})
    return policy

def stats(rr):
    d=np.array([r['delta'] for r in rr]);rng=np.random.default_rng(2026)
    ci=np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975])
    return {'count':len(rr),'tp_dice':float(np.mean([r['tp_dice'] for r in rr])),'pc_dice':float(np.mean([r['pc_dice'] for r in rr])),'gate_dice':float(np.mean([r['dice'] for r in rr])),'mean_delta':float(d.mean()),'paired_bootstrap_95ci':ci.tolist(),'switches':sum(r['use_pc'] for r in rr),'wins':int((d>1e-12).sum()),'losses':int((d< -1e-12).sum()),'positive_delta_sum':float(d[d>0].sum()),'negative_delta_sum':float(d[d<0].sum()),'two_selected_mask_oracle':float(np.mean([max(r['tp_dice'],r['pc_dice']) for r in rr]))}

def run():
    global ROUTER
    assert not R.exists();R.mkdir();(R/'code').mkdir();shutil.copy2(__file__,R/'code/run_gain_gate.py')
    source=P/'scripts/analyze_propagation_quality_router.py';shutil.copy2(source,R/'code/router.py')
    spec=importlib.util.spec_from_file_location('historical_router',source);ROUTER=importlib.util.module_from_spec(spec);spec.loader.exec_module(ROUTER)
    config={'created_at':datetime.datetime.now().astimezone().isoformat(),'student':False,'new_sam3_inference':False,'split':'unchanged 800 train / 100 validation / 100 test; original 8 labeled and 792 unlabeled train','features':FEATURES,'expert_ridge':1.,'gate_ridges':[10.,100.],'thresholds':[.02,.05,.10],'outer_folds':5,'outer_seed':2026,'policy_tuning_folds':4,'policy_tuning_seed':2027,'gate_training_expert_oof_folds':3,'gate_training_expert_oof_seed':2028,'selection':'inner mean Dice gain; exact TP fallback included; ties fewer switches then stronger regularization then larger threshold','test_gate':'nested outer mean Dice gain >= .003 AND paired bootstrap 95% lower endpoint > 0; otherwise do not evaluate a new policy on test','source_sha256':sha(source),'code_sha256':sha(Path(__file__))}
    save(R/'config.json',config)
    try:
        load('validation');ids=sorted(DATA['validation']);outer=split_folds(ids,5,2026);all_rows=[]
        for fold,held in enumerate(outer):
            train=sorted(set(ids)-set(held));policy=train_policy(train,f'outer_{fold}')
            rr=decisions(policy['model'],policy['threshold'],examples(train,held))
            for row in rr:row['outer_fold']=fold
            all_rows+=rr;jsonl(R/f'outer_{fold}_predictions.jsonl',rr)
            print('outer',fold,json.dumps(stats(rr)),flush=True)
        assert len(all_rows)==100 and len({r['target_id'] for r in all_rows})==100
        val=stats(all_rows);assert abs(val['tp_dice']-.8517552851672318)<1e-10
        val['passes_test_gate']=bool(val['mean_delta']>=.003 and val['paired_bootstrap_95ci'][0]>0)
        val['fold_delta']=[float(np.mean([r['delta'] for r in all_rows if r['outer_fold']==k])) for k in range(5)]
        jsonl(R/'validation_nested_oof.jsonl',sorted(all_rows,key=lambda r:r['target_id']));save(R/'validation_summary.json',val)
        save(R/'fold_provenance.json',AUDIT)
        test_result=None
        if val['passes_test_gate']:
            final=train_policy(ids,'final_validation');save(R/'frozen_final_policy.json',final)
            models=expert_models(ids);save(R/'frozen_experts.json',{k:vars(v) for k,v in models.items()});(R/'FROZEN_BEFORE_TEST').touch()
            load('test');rr=decisions(final['model'],final['threshold'],examples(ids,sorted(DATA['test']),'test'))
            dest=R/'final_test_masks';dest.mkdir()
            for row in rr:
                src=Path(row['selected_mask_path']);src=src if src.is_absolute() else P/src
                path=dest/(row['target_id'].replace('::','__')+'.png');shutil.copy2(src,path);row['final_mask_path']=str(path);row['mask_sha256']=sha(path)
            jsonl(R/'selected_test_masks.jsonl',rr)
            for row in rr:
                target=DATA['test'][row['target_id']]
                t=next(r for r in target[MODES[0]] if r['route_id']==row['tp_route_id']);p=next(r for r in target[MODES[1]] if r['route_id']==row['pc_route_id'])
                row.update(tp_dice=t['gt_dice_evaluation_only'],pc_dice=p['gt_dice_evaluation_only'])
                row['dice']=row['pc_dice'] if row['use_pc'] else row['tp_dice'];row['delta']=row['dice']-row['tp_dice']
                pred=mask(row['final_mask_path']);gt=np.asarray(Image.open(t['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
                assert abs(dice(pred,gt)-row['dice'])<1e-12
                inter=int(np.logical_and(pred,gt).sum());row['iou']=inter/max(int(pred.sum())+int(gt.sum())-inter,1)
            test_result=stats(rr);test_result['iou']=float(np.mean([r['iou'] for r in rr]))
            assert abs(test_result['tp_dice']-.8854326463411542)<1e-12 and abs(test_result['pc_dice']-.8616882728797673)<1e-12
            jsonl(R/'test_per_target_metrics.jsonl',rr);save(R/'test_summary.json',test_result)
        else:
            save(R/'TEST_NOT_RUN.json',{'reason':'Predeclared nested validation gain/stability criterion failed; retain original TP independent Router.','no_new_test_policy_evaluated':True})
        save(R/'results.json',{'validation':val,'test':test_result})
        lines=['# Kvasir：保留TP的PC增益门控实验','', '无学生、无新增SAM3传播。原两类b0-b6候选与独立Ridge Router保持；预测PC相对TP的Dice增益，仅超过阈值才替换。', '', '原800/100/100和8/792划分不变。外层5折评估完整流程；内层4折选择参数，每个内层gate训练样本由更内层3折OOF独立Router产生。所有划分按target图，所有训练/拟合/阈值选择排除当前外层held目标。', '', '| 验证指标 | 值 |','|---|---:|']
        for name in ['tp_dice','pc_dice','gate_dice','mean_delta','switches','wins','losses','two_selected_mask_oracle']:lines.append(f'| {name} | {val[name]} |')
        lines+=['',f"配对bootstrap 95%区间：{val['paired_bootstrap_95ci']}。各外层折Dice差：{val['fold_delta']}。",'', '预先固定test启用条件：外层平均Dice增益至少0.003，且配对bootstrap 95%区间下界大于0。此区间是当前样本的描述性稳定性检查，不是未来泛化保证。', '', '门控训练样本量约为目标图数，而不是候选数；不使用学生网络。候选mask一致性先按mask内容去重。Oracle只用于事后诊断。']
        if test_result:lines+=['','## 冻结后Test结果','',json.dumps(test_result,ensure_ascii=False,indent=2),'','100张最终mask均已保存并独立复算指标。']
        else:lines+=['','## 结论','', '本轮未通过预设验证门槛，因此没有用新门控评估test；保留原TP独立结果，历史test Dice为0.885433。不能把它写作新门控的test成绩。']
        lines+=['',f'完整产物：`{R}`。当前报告对应一次固定方案，不根据test调参。']
        report='\n'.join(lines)+'\n';(R/'report.md').write_text(report)
        with (P/'reproduction_reports/Kvasir_TP_PC_gain_gate_20260907.md').open('x') as f:f.write(report)
        (R/'COMPLETE').write_text(datetime.datetime.now().astimezone().isoformat());print(json.dumps({'validation':val,'test':test_result}),flush=True)
    except BaseException:
        (R/'FAILED').write_text(traceback.format_exc());raise

if __name__=='__main__':run()
