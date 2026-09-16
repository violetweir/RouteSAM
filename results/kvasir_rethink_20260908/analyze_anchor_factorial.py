"""Target-grouped validation controls; no test evaluation or variant tuning."""
import collections
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from run_anchor_matrix import P,R as MATRIX,OLD,MODES,read,save,jsonl,module,sha,resolve

R=MATRIX.parent/'anchor_factorial'

def summary(rows, key='dice'):
    d=np.array([r[key]-r['tp_dice'] for r in rows]);rng=np.random.default_rng(2026)
    ci=np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975])
    return dict(dice=float(np.mean([r[key] for r in rows])),delta=float(d.mean()),ci95=ci.tolist(),
        wins=int((d>1e-12).sum()),losses=int((d< -1e-12).sum()),severe_losses_over_005=int((d<-.05).sum()),
        fold_delta=[float(np.mean([r[key]-r['tp_dice'] for r in rows if r['outer_fold']==f])) for f in range(5)])

def prepare_code():
    code=R/'analysis_code';code.mkdir(exist_ok=True)
    src=P/'work/kvasir_tp_pc_candidate_rerank_20260907/code'
    for name in ['base_gate.py','run_candidate_rerank.py']:
        dest=code/name
        if not dest.exists():shutil.copy2(src/name,dest)
    if not (code/'router.py').exists():shutil.copy2(P/'scripts/analyze_propagation_quality_router.py',code/'router.py')
    sys.path.insert(0,str(code))
    import run_candidate_rerank as rerank
    import base_gate as g
    g.ROUTER=module('factorial_router',code/'router.py')
    save(R/'analysis_config.json',dict(primary='Frozen TP independent Router + existing nested all-aux-candidate gain gate',
        primary_features=g.FEATURES,ridges=[10,100],thresholds=[.02,.05,.1],fallback='Exact TP',
        folds='5 outer seed2026 / 4 inner seed2027 / 3 training experts seed2028, grouped by target',
        secondary='Independent auxiliary and joint Ridge Router, five-fold OOF',
        comparisons=['C minus B: alternative anchor retrieval','D minus C: path score at fixed anchor'],
        no_test=True,no_variant_selected_from_outer_results=True,code_sha256={f.name:sha(f) for f in code.glob('*.py')}))
    return g,rerank

def load_variant(g,variant):
    g.CACHE={};g.DATA={};g.AUDIT=[];g.R=R/variant/'selection';g.R.mkdir(exist_ok=True)
    out={};audit=[]
    for mode,path in zip(g.MODES,[OLD/MODES[0]/'propagation_quality_validation/propagation_quality.jsonl',R/variant/'propagation_quality.jsonl']):
        rows=read(path);assert len(rows)==700
        by=g.ROUTER.grouped([{**r,'feature_mode':mode} for r in rows])
        for target,group in by.items():
            group.sort(key=lambda r:r['bridge_count']);assert [r['bridge_count'] for r in group]==list(range(7))
            masks=[g.mask(r['forward_mask_path']) for r in group]
            gt=np.asarray(Image.open(group[0]['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
            for r,m in zip(group,masks):
                assert sha(resolve(r['forward_mask_path']))==r['forward_mask_sha256']
                assert abs(g.dice(m,gt)-r['gt_dice_evaluation_only'])<1e-12
                assert r['q_cycle'] is not None
            digests=[__import__('hashlib').sha256(m.tobytes()).hexdigest() for m in masks];representatives=list(dict.fromkeys(digests))
            for i,r in enumerate(group):
                others=[digests.index(h) for h in representatives if h!=digests[i]]
                r['_consensus']=float(np.mean([g.dice(masks[i],masks[j]) for j in others])) if others else 1.
                r['_pred_area']=int(masks[i].sum());r['_mask']=masks[i]
            out.setdefault(target,{})[mode]=group
        audit.append(dict(path=str(path),sha256=sha(path),verified_masks=700))
    assert len(out)==100;g.DATA['validation']=out
    save(g.R/'input_audit.json',audit)

def run_variant(g,rerank,variant):
    load_variant(g,variant);ids=sorted(g.DATA['validation']);all_rows=[]
    frozen={r['target_id']:r for r in read(P/'work/kvasir_tp_pc_candidate_rerank_20260907/validation_nested_oof.jsonl')}
    for fold,held in enumerate(g.split_folds(ids,5,2026)):
        train=sorted(set(ids)-set(held));policy=g.train_policy(train,f'outer_{fold}')
        rows=rerank.decisions(policy['model'],policy['threshold'],rerank.examples(train,held))
        experts=g.expert_models(train)
        training=[r for t in train for mode in g.MODES for r in g.DATA['validation'][t][mode]]
        fitted=g.linear_fit([g.ROUTER.feature_vector(g.clean(r),True) for r in training],[r['gt_dice_evaluation_only'] for r in training],1.)
        joint=g.ROUTER.Ridge(**fitted,include_mode=True)
        for row in rows:
            target=row['target_id'];tp=g.DATA['validation'][target][g.MODES[0]];aux=g.DATA['validation'][target][g.MODES[1]]
            jchoice=g.choice(joint,tp+aux)[0];achoice=g.choice(experts[g.MODES[1]],aux)[0]
            row.update(outer_fold=fold,joint_dice=jchoice['gt_dice_evaluation_only'],joint_route_id=jchoice['route_id'],
                joint_feature_mode=jchoice['feature_mode'],aux_router_dice=achoice['gt_dice_evaluation_only'],
                tp_oracle=max(r['gt_dice_evaluation_only'] for r in tp),aux_oracle=max(r['gt_dice_evaluation_only'] for r in aux),
                union_oracle=max(r['gt_dice_evaluation_only'] for r in tp+aux))
            assert row['tp_route_id']==frozen[target]['tp_route_id']
            assert row['union_oracle']>=row['tp_oracle']-1e-12
        all_rows+=rows;jsonl(g.R/f'outer_{fold}.jsonl',rows)
        print(variant,'fold',fold,'gate',np.mean([r['dice'] for r in rows]),flush=True)
    all_rows.sort(key=lambda r:r['target_id'])
    assert abs(np.mean([r['tp_dice'] for r in all_rows])-.8517552851672318)<1e-12
    result=dict(primary_gate=summary(all_rows),secondary_joint_router=summary(all_rows,'joint_dice'),aux_router=summary(all_rows,'aux_router_dice'),
        oracle={key:float(np.mean([r[key] for r in all_rows])) for key in ['tp_oracle','aux_oracle','union_oracle','candidate_oracle']},
        switch_count=sum(r['use_pc'] for r in all_rows),
        fixed_bridge_aux_dice={str(b):float(np.mean([g.DATA['validation'][t][g.MODES[1]][b]['gt_dice_evaluation_only'] for t in ids])) for b in range(7)},
        audit=dict(expert_partitions=len(g.AUDIT),all_disjoint=all(not set(a['expert_train_ids'])&set(a['held_ids']) for a in g.AUDIT),
            frozen_tp_route_matches=100,oracle_monotonicity_checks=100))
    assert result['audit']['all_disjoint']
    save(g.R/'fold_provenance.json',g.AUDIT);jsonl(g.R/'validation_oof.jsonl',all_rows);save(g.R/'results.json',result)
    return result,all_rows

def main():
    assert (R/'INFERENCE_COMPLETE.json').exists()
    g,rerank=prepare_code();results={};rows={}
    for variant in ['B','C','D']:results[variant],rows[variant]=run_variant(g,rerank,variant)
    comparisons={}
    for new,old in [('C','B'),('D','C')]:
        comparisons[new+'-'+old]={}
        for key in ['dice','joint_dice','aux_router_dice','union_oracle']:
            pairs=[dict(dice=n[key],tp_dice=o[key],outer_fold=n['outer_fold']) for n,o in zip(rows[new],rows[old])]
            comparisons[new+'-'+old][key]=summary(pairs)
    save(R/'results.json',dict(split='validation',TP_OOF=.8517552851672318,variants=results,paired_comparisons=comparisons,test=None))
    save(R/'TEST_NOT_RUN.json',dict(reason='Mechanism diagnosis on validation; no test variant selected or run.'))
    lines=['# Kvasir：辅助参考图检索与传播路径固定预算对照','','原 TP 七候选完整保留，各组增加七个辅助候选，validation 100 张。原 8/792 训练身份不变，无学生，无新 test。','','B：TP 备用 anchor + TP 路径；C：PC 备用 anchor + TP 路径；D：与 C 相同 anchor + PC 路径。','','| 方案 | 冻结 TP + nested gain gate | 辅助独立 Router | 联合 Router | 联合 Oracle |','|---|---:|---:|---:|---:|']
    lines.append('| TP 原基线 | 0.851755 | — | 0.851755 | 0.869544 |')
    for v,r in results.items():lines.append(f"| {v} | {r['primary_gate']['dice']:.6f} | {r['aux_router']['dice']:.6f} | {r['secondary_joint_router']['dice']:.6f} | {r['oracle']['union_oracle']:.6f} |")
    lines+=['','主要选择器复用既有 15 特征、ridge 10/100、阈值 0.02/0.05/0.10 与 TP 回退，按图嵌套交叉验证；辅助候选的 PC 字段名仅为复用接口，不表示 B/C 使用 PC 路径。联合 Router 为次要比较，允许其改变 TP 内部选择。','', 'C−B 检验辅助 anchor 检索，D−C 检验固定 anchor 后的路径评分。Oracle 仅用目标 GT 事后计算，不参与推理。三个变体均完整报告，未用外层结果选择 test 赢家。','','配对差异与置信区间见 results.json。全部输入 mask 校验哈希与 GT Dice；每组 100 个 TP 默认选择与历史折外输出完全一致；全部联合 Oracle 均不低于原 TP。','']
    (R/'report.md').write_text('\n'.join(lines));(R/'COMPLETE').write_text('validation analysis complete\n')
    print((R/'results.json').read_text(),flush=True)

if __name__=='__main__':main()
