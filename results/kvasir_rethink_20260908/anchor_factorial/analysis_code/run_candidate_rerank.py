"""TP-preserving relative-gain ranking over all seven original PC candidates."""
import collections
import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import traceback
import numpy as np
from PIL import Image
import base_gate as g

R=g.P/'work/kvasir_tp_pc_candidate_rerank_20260907'
g.R=R
g.FEATURES[8]='pc_score_minus_best_other_pc_score'

def examples(train,held,split='validation'):
    assert set(train).isdisjoint(held)
    key=(tuple(sorted(train)),tuple(sorted(held)),split)
    if key in g.CACHE:return g.CACHE[key]
    models=g.expert_models(train);rr=[]
    for target in sorted(held):
        t,tm=g.choice(models[g.MODES[0]],g.DATA[split][target][g.MODES[0]])
        pc=g.DATA[split][target][g.MODES[1]]
        scores=[models[g.MODES[1]].score(g.clean(p)) for p in pc]
        order=sorted(range(7),key=lambda k:(scores[k],-pc[k]['bridge_count'],pc[k]['route_id']),reverse=True)
        for k,p in enumerate(pc):
            margin=scores[k]-max(scores[j] for j in range(7) if j!=k)
            f=g.pair_features(g.clean(t),g.clean(p),tm,margin)
            row={'target_id':target,'features':f,'tp_route_id':t['route_id'],'pc_route_id':p['route_id'],'tp_mask_path':t['forward_mask_path'],'pc_mask_path':p['forward_mask_path'],'pc_bridge_count':p['bridge_count'],'pc_original_rank':order.index(k)+1}
            if split=='validation':row.update(tp_dice=t['gt_dice_evaluation_only'],pc_dice=p['gt_dice_evaluation_only'],gain=p['gt_dice_evaluation_only']-t['gt_dice_evaluation_only'],pc_top1_dice=pc[order[0]]['gt_dice_evaluation_only'])
            rr.append(row)
    g.AUDIT.append({'expert_train_ids':sorted(train),'held_ids':sorted(held),'split':split})
    g.CACHE[key]=rr;return rr

def crossfit(ids,n,seed):
    rr=[]
    for held in g.split_folds(ids,n,seed):rr+=examples(sorted(set(ids)-set(held)),held)
    assert len(rr)==7*len(ids) and set(collections.Counter(r['target_id'] for r in rr).values())=={7}
    return sorted(rr,key=lambda r:(r['target_id'],r['pc_bridge_count']))

def fit_grouped(rr,lam):
    # Each target contributes mean squared loss across its seven candidates.
    # This preserves ridge strength per target instead of treating 7N rows as 7N images.
    counts=collections.Counter(r['target_id'] for r in rr)
    weights=np.array([1/counts[r['target_id']] for r in rr],np.float64)
    x=np.asarray([r['features'] for r in rr],np.float64);y=np.asarray([r['gain'] for r in rr],np.float64)
    mu=np.average(x,axis=0,weights=weights);sd=np.maximum(np.sqrt(np.average((x-mu)**2,axis=0,weights=weights)),1e-8)
    z=np.column_stack([np.ones(len(x)),(x-mu)/sd]);wroot=np.sqrt(weights)
    z=z*wroot[:,None];target=y*wroot;reg=np.eye(z.shape[1])*lam;reg[0,0]=0
    coefficients=np.linalg.solve(z.T@z+reg,z.T@target)
    return {'means':mu.tolist(),'stds':sd.tolist(),'weights':coefficients.tolist()}

def decisions(model,threshold,rr):
    groups=collections.defaultdict(list)
    for row in rr:groups[row['target_id']].append(row)
    result=[]
    for target,group in sorted(groups.items()):
        assert len(group)==7 and len({r['pc_route_id'] for r in group})==7
        assert len({r['tp_route_id'] for r in group})==1
        if model is not None:
            predictions=[g.predict(model,r['features']) for r in group]
            k=max(range(7),key=lambda j:(predictions[j],-group[j]['pc_bridge_count'],group[j]['pc_route_id']))
            pred=predictions[k];use_pc=bool(pred>threshold)
        else:
            k=min(range(7),key=lambda j:group[j]['pc_original_rank']);pred=None;use_pc=False
        row=group[k]
        chosen={**row,'predicted_gain':pred,'use_pc':use_pc,'selected_route_id':row['pc_route_id'] if use_pc else row['tp_route_id'],'selected_mask_path':row['pc_mask_path'] if use_pc else row['tp_mask_path']}
        if 'gain' in row:
            chosen.update(dice=row['pc_dice'] if use_pc else row['tp_dice'],delta=row['gain'] if use_pc else 0.,candidate_oracle=max([row['tp_dice']]+[r['pc_dice'] for r in group]))
        result.append(chosen)
    return result

def stats(rr):
    d=np.array([r['delta'] for r in rr]);rng=np.random.default_rng(2026)
    ci=np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975])
    return {'count':len(rr),'tp_dice':float(np.mean([r['tp_dice'] for r in rr])),'pc_original_top1_dice':float(np.mean([r['pc_top1_dice'] for r in rr])),'rerank_dice':float(np.mean([r['dice'] for r in rr])),'mean_delta':float(d.mean()),'paired_bootstrap_95ci':ci.tolist(),'switches':sum(r['use_pc'] for r in rr),'wins':int((d>1e-12).sum()),'losses':int((d< -1e-12).sum()),'positive_delta_sum':float(d[d>0].sum()),'negative_delta_sum':float(d[d<0].sum()),'candidate_oracle':float(np.mean([r['candidate_oracle'] for r in rr])),'switched_pc_original_rank_counts':dict(collections.Counter(r['pc_original_rank'] for r in rr if r['use_pc']))}

g.examples=examples;g.crossfit=crossfit;g.gate_fit=fit_grouped;g.decisions=decisions

def run():
    assert not R.exists();R.mkdir();(R/'code').mkdir()
    for name in ['run_candidate_rerank.py','base_gate.py','test_candidate_rerank.py']:shutil.copy2(Path(__file__).parent/name,R/'code'/name)
    source=g.P/'scripts/analyze_propagation_quality_router.py';shutil.copy2(source,R/'code/router.py')
    spec=importlib.util.spec_from_file_location('historical_router',source);g.ROUTER=importlib.util.module_from_spec(spec);spec.loader.exec_module(g.ROUTER)
    config={'created_at':datetime.datetime.now().astimezone().isoformat(),'student':False,'new_sam3_inference':False,'candidate_set':'one TP independently selected mask + all seven original PC masks','split':'original 800/100/100, labeled/unlabeled train 8/792 unchanged','features':g.FEATURES,'changed_feature_definition':'signed PC score margin relative to its best other PC candidate; all other 14 features unchanged from previous gate','loss':'per-target mean of candidate squared gain errors + ridge; unpenalized intercept','expert_ridge':1.,'ridges':[10.,100.],'thresholds':[.02,.05,.10],'fallback':'exact TP independent mask; no TP reranking','nested_validation':'outer 5 seed2026; inner policy tuning 4 seed2027; gate training experts crossfit 3 seed2028; by target','test_gate':'outer mean gain >= .003 and paired 95% bootstrap lower bound >0, same as previous experiment','code_sha256':{f.name:g.sha(f) for f in (R/'code').glob('*.py')}}
    g.save(R/'config.json',config)
    try:
        g.load('validation');ids=sorted(g.DATA['validation']);all_rows=[]
        for fold,held in enumerate(g.split_folds(ids,5,2026)):
            train=sorted(set(ids)-set(held));policy=g.train_policy(train,f'outer_{fold}')
            rr=decisions(policy['model'],policy['threshold'],examples(train,held))
            for row in rr:row['outer_fold']=fold
            all_rows+=rr;g.jsonl(R/f'outer_{fold}_predictions.jsonl',rr);print('outer',fold,json.dumps(stats(rr)),flush=True)
        assert len(all_rows)==100 and len({r['target_id'] for r in all_rows})==100
        val=stats(all_rows)
        assert abs(val['tp_dice']-.8517552851672318)<1e-10 and abs(val['candidate_oracle']-.8863397537782618)<1e-10
        previous={r['target_id']:r for r in g.read(g.P/'work/kvasir_tp_pc_gain_gate_20260907/validation_nested_oof.jsonl')}
        assert all(r['tp_route_id']==previous[r['target_id']]['tp_route_id'] for r in all_rows)
        val['passes_test_gate']=bool(val['mean_delta']>=.003 and val['paired_bootstrap_95ci'][0]>0)
        val['fold_delta']=[float(np.mean([r['delta'] for r in all_rows if r['outer_fold']==k])) for k in range(5)]
        g.jsonl(R/'validation_nested_oof.jsonl',sorted(all_rows,key=lambda r:r['target_id']));g.save(R/'validation_summary.json',val)
        test_result=None
        if val['passes_test_gate']:
            final=g.train_policy(ids,'final_validation');g.save(R/'frozen_final_policy.json',final)
            models=g.expert_models(ids);g.save(R/'frozen_experts.json',{k:vars(v) for k,v in models.items()});(R/'FROZEN_BEFORE_TEST').touch()
            g.load('test');rr=decisions(final['model'],final['threshold'],examples(ids,sorted(g.DATA['test']),'test'))
            dest=R/'final_test_masks';dest.mkdir()
            for row in rr:
                src=Path(row['selected_mask_path']);src=src if src.is_absolute() else g.P/src
                p=dest/(row['target_id'].replace('::','__')+'.png');shutil.copy2(src,p);row.update(final_mask_path=str(p),mask_sha256=g.sha(p))
            g.jsonl(R/'selected_test_masks.jsonl',rr) # GT-free choices saved first.
            for row in rr:
                target=g.DATA['test'][row['target_id']]
                tp=next(r for r in target[g.MODES[0]] if r['route_id']==row['tp_route_id'])
                pc=next(r for r in target[g.MODES[1]] if r['route_id']==row['pc_route_id'])
                top1=g.choice(models[g.MODES[1]],target[g.MODES[1]])[0]
                row.update(tp_dice=tp['gt_dice_evaluation_only'],pc_dice=pc['gt_dice_evaluation_only'],pc_top1_dice=top1['gt_dice_evaluation_only'],candidate_oracle=max([tp['gt_dice_evaluation_only']]+[r['gt_dice_evaluation_only'] for r in target[g.MODES[1]]]))
                row['dice']=row['pc_dice'] if row['use_pc'] else row['tp_dice'];row['delta']=row['dice']-row['tp_dice']
                pred=g.mask(row['final_mask_path']);gt=np.asarray(Image.open(tp['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
                assert abs(g.dice(pred,gt)-row['dice'])<1e-12
                inter=int(np.logical_and(pred,gt).sum());row['iou']=inter/max(int(pred.sum())+int(gt.sum())-inter,1)
            test_result=stats(rr);test_result['iou']=float(np.mean([r['iou'] for r in rr]))
            assert abs(test_result['tp_dice']-.8854326463411542)<1e-12 and abs(test_result['pc_original_top1_dice']-.8616882728797673)<1e-12
            g.jsonl(R/'test_per_target_metrics.jsonl',rr);g.save(R/'test_summary.json',test_result)
        else:g.save(R/'TEST_NOT_RUN.json',{'reason':'Predeclared nested validation gain/stability criterion failed; keep original TP.','new_test_policy_evaluated':False})
        g.save(R/'fold_provenance.json',g.AUDIT)
        assert all(not set(x['expert_train_ids'])&set(x['held_ids']) for x in g.AUDIT)
        audit={'partitions_checked':len(g.AUDIT),'train_held_disjoint':True,'tp_outer_choices_identical_to_prior_100':True,'new_test_read':bool(test_result is not None)}
        g.save(R/'audit.json',audit);g.save(R/'results.json',{'validation':val,'test':test_result})
        lines=['# Kvasir：固定TP结果 + 全7个PC候选的相对收益重排序','', '无学生，无新增SAM3传播；原800/100/100及8/792划分不变。每图固定一个独立TP结果，保留PC全部b0-b6七个候选；回归每个PC候选相对TP的Dice增益，最高预测增益超过阈值才替换。', '', '复用上一轮15个特征，其中PC分数间隔改为该候选相对其最佳其他PC候选的有符号间隔。使用按图平均的候选损失，避免把7条候选视为7张独立图而削弱正则。', '', '外层5折评估完整流程，内层4折调参数、更内层3折产生训练OOF专家输出；全部按目标图划分。正则10/100、阈值0.02/0.05/0.10，加原TP回退选项；只用内层验证平均收益选参数。', '', '| 验证指标 | 值 |','|---|---:|']
        for key in ['tp_dice','pc_original_top1_dice','rerank_dice','mean_delta','switches','wins','losses','candidate_oracle']:lines.append(f'| {key} | {val[key]} |')
        lines+=['',f"配对bootstrap 95%区间：{val['paired_bootstrap_95ci']}。各外层折Dice差：{val['fold_delta']}。",'',f"实际替换所选PC原排名分布：{val['switched_pc_original_rank_counts']}。",'', 'test启用条件保持上一轮预设：外层平均Dice增益至少0.003，且配对bootstrap 95%区间下界大于0。区间是当前样本的描述性检查，不保证未来泛化。Oracle只用于事后分析。']
        if test_result:lines+=['','## 冻结后Test','',json.dumps(test_result,indent=2,ensure_ascii=False),'','100张最终mask均已保存并独立复算。']
        else:lines+=['','## 结论','', '未通过预设验证门槛；本轮未读取test候选文件或用新模型评估test。继续保留原TP独立结果，其历史test Dice0.885433不是新模型成绩。']
        lines+=['',f'完整产物：`{R}`。']
        report='\n'.join(lines)+'\n';(R/'report.md').write_text(report)
        with (g.P/'reproduction_reports/Kvasir_TP_PC_candidate_rerank_20260907.md').open('x') as f:f.write(report)
        (R/'COMPLETE').write_text(datetime.datetime.now().astimezone().isoformat());print(json.dumps({'validation':val,'test':test_result}),flush=True)
    except BaseException:(R/'FAILED').write_text(traceback.format_exc());raise

if __name__=='__main__':run()
