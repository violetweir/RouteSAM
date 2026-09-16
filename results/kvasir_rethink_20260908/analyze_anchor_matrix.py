"""Posthoc GT diagnostic; no target GT is used to generate or rank inference."""
import collections
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from run_anchor_matrix import R, OLD, MODES, read, save, jsonl, sha, resolve

def dice(a,b):
    return float(2*np.logical_and(a,b).sum()/max(int(a.sum())+int(b.sum()),1))

def stats(v, base=None):
    v=np.asarray(v,dtype=float)
    result={'mean':float(v.mean()),'count':len(v)}
    if base is not None:
        d=v-np.asarray(base);rng=np.random.default_rng(2026)
        ci=np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975])
        result.update(delta=float(d.mean()),ci95=ci.tolist(),wins=int((d>1e-12).sum()),
            losses=int((d< -1e-12).sum()),severe_losses_over_005=int((d<-.05).sum()),
            worst_delta=float(d.min()),positive_sum=float(d[d>0].sum()),negative_sum=float(d[d<0].sum()))
    return result

def main():
    assert (R/'INFERENCE_COMPLETE.json').exists()
    rr=read(R/'quality/propagation_quality.jsonl')
    groups=collections.defaultdict(list)
    for r in rr:groups[r['target_id']].append(r)
    assert len(groups)==100 and all(len(v)==8 for v in groups.values())
    ids=sorted(groups);per=[];all_y=[];all_tp=[];all_pc=[];all_cons=[];rank_corr={'TP':[],'PC':[],'consensus':[]}
    selected=collections.defaultdict(list);oracles=collections.defaultdict(list)
    old_b0={m:{r['target_id']:r for r in read(OLD/m/'propagation_quality_validation/propagation_quality.jsonl') if r['bridge_count']==0} for m in MODES}
    masks_checked=0
    for target in ids:
        rows=sorted(groups[target],key=lambda r:r['anchor_index'])
        y=np.array([r['gt_dice_evaluation_only'] for r in rows])
        tp=np.array([r['tp_retrieval_score'] for r in rows]);pc=np.array([r['pc_retrieval_score'] for r in rows])
        tp_order=sorted(range(8),key=lambda i:(tp[i],rows[i]['anchor_id']),reverse=True)
        pc_order=sorted(range(8),key=lambda i:(pc[i],rows[i]['anchor_id']),reverse=True)
        assert rows[tp_order[0]]['route_id']==old_b0[MODES[0]][target]['route_id']
        assert rows[pc_order[0]]['route_id']==old_b0[MODES[1]][target]['route_id']
        ms=[];gt=np.asarray(Image.open(rows[0]['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
        for i,r in enumerate(rows):
            path=resolve(r['forward_mask_path']);assert sha(path)==r['forward_mask_sha256']
            mask=np.asarray(Image.open(path).convert('L'))>127
            assert mask.shape==(256,256) and abs(dice(mask,gt)-y[i])<1e-12
            ms.append(mask);masks_checked+=1
        similarity=np.array([[dice(a,b) for b in ms] for a in ms]);np.fill_diagonal(similarity,0)
        agreement=similarity.sum(1)/7
        medoid=max(range(8),key=lambda i:(agreement[i],tp[i],rows[i]['anchor_id']))
        confidence=max(range(8),key=lambda i:(rows[i]['final_sam_score'],tp[i],rows[i]['anchor_id']))
        alternate=next(i for i in pc_order if i!=tp_order[0])
        choices={'TP_top1':tp_order[0],'PC_top1':pc_order[0],'eight_anchor_mask_medoid':medoid,'eight_anchor_sam_confidence':confidence}
        for k,i in choices.items():selected[k].append(y[i])
        poolsets={'all8':range(8),'TP_top2':tp_order[:2],'TP_top3':tp_order[:3],'PC_top2':pc_order[:2],
            'TP_top1_PC_alternate':[tp_order[0],alternate]}
        for k,inds in poolsets.items():oracles[k].append(float(y[list(inds)].max()))
        for key,scores in [('TP',tp),('PC',pc),('consensus',agreement)]:
            c=float(spearmanr(scores,y).statistic)
            if np.isfinite(c):rank_corr[key].append(c)
        per.append(dict(target_id=target,anchor_ids=[r['anchor_id'] for r in rows],dice=y.tolist(),tp_scores=tp.tolist(),pc_scores=pc.tolist(),
            agreement=agreement.tolist(),tp_rank=tp_order,pc_rank=pc_order,choices=choices,
            oracle={k:oracles[k][-1] for k in poolsets},mask_pairwise_dice=similarity.tolist()))
        all_y.append(y);all_tp.append(tp);all_pc.append(pc);all_cons.append(agreement)
    all_y=np.array(all_y);folds=json.loads((R.parent.parent/'kvasir_pc_adaptive_spatial_20260906/validation_folds.json').read_text())
    # Diagnostic OOF fixed-anchor reference: select one anchor using training-fold mean Dice only.
    oof_fixed=np.zeros(100);fixed_choice={}
    for fold in range(5):
        held=np.array([folds[t]==fold for t in ids]);train=~held
        anchor=int(np.argmax(all_y[train].mean(0)));oof_fixed[held]=all_y[held,anchor];fixed_choice[fold]=anchor
    selected['OOF_fixed_anchor_training_mean']=oof_fixed.tolist()
    baseline=selected['TP_top1']
    score_audit=[]
    for ai in range(8):
        ar=[r for r in rr if r['anchor_index']==ai]
        nonempty=[r['final_sam_score'] for r in ar if r['final_candidate_count']>0]
        score_audit.append(dict(anchor_index=ai,nonempty_targets=len(nonempty),
            unique_nonempty_final_sam_scores=sorted(set(nonempty))))
    res=dict(split='validation only',targets=100,anchors=8,student=False,
        actual={k:stats(v,baseline) for k,v in selected.items()},
        oracle={k:stats(v,baseline) for k,v in oracles.items()},
        within_target_spearman={k:stats(v) for k,v in rank_corr.items()},
        anchor_mean_dice=all_y.mean(0).tolist(),anchor_dice_below_05=(all_y<.5).sum(0).tolist(),
        error_correlation=np.corrcoef((1-all_y).T).tolist(),oof_fixed_anchor_by_fold=fixed_choice,
        final_sam_score_by_anchor=score_audit,
        audit=dict(mask_hashes_and_gt_dice_checked=masks_checked,original_TP_PC_b0_choices_reproduced=200,
            protocol_manifest_sha256=sha(R/'protocol/merged_manifest.jsonl'),test_run=False),
        interpretation='Oracle top-k uses target GT posthoc. Eight-anchor medoid/confidence spend eight candidates. OOF fixed-anchor uses training-fold validation labels. These are diagnostics, not claims of seven-route superiority.')
    save(R/'results.json',res);jsonl(R/'per_target_diagnostic.jsonl',per)
    np.savez_compressed(R/'transfer_matrix.npz',target_ids=ids,anchor_ids=per[0]['anchor_ids'],dice=all_y,tp=np.array(all_tp),pc=np.array(all_pc),consensus=np.array(all_cons))
    lines=['# Kvasir：8-anchor b0 直接传播诊断','','仅 validation 100 张；沿用原 8 个训练 anchor。无学生，无新 test 评估。','','## 实际选择结果','','| 方法 | Dice | 相对 TP b0 |','|---|---:|---:|']
    for k,v in res['actual'].items():lines.append(f"| {k} | {v['mean']:.6f} | {v['delta']:+.6f} |")
    lines+=['','## 候选上界（GT 事后选择）','','| 候选 | Oracle Dice |','|---|---:|']
    for k,v in res['oracle'].items():lines.append(f"| {k} | {v['mean']:.6f} |")
    lines+=['','上述 b0 结果不能与 b0–b6 Router 直接当作相同预算的机制比较。全 8 anchor 结果用于定位检索瓶颈；top-k Oracle 不是实际可用选择器。', '', '目标内部相似度与真实传播 Dice 的平均 Spearman：', '',json.dumps(res['within_target_spearman'],ensure_ascii=False,indent=2),'','全部 800 张 mask 已校验哈希并独立重算 Dice；TP/PC 的 200 个原始 b0 检索选择已复现。', '', '## 置信度字段的新诊断', '', '本轮每个 anchor 在所有非空 target 输出上的 final_sam_score 都只有一个取值；空输出时为 0。它在这里基本区分参考对象，不能解释为目标 mask 的连续质量估计。SAM3 的后处理另算了 out_tracker_probs，但当前返回接口没有导出它；旧脚本 select_top 读取的是 out_probs。后续若研究质量选择，应分清对象分数、逐帧跟踪存在分数与 mask 准确度，不能直接把其中任何一个视为 Dice。当前固定预算对照继续保持旧特征定义，以免把候选机制与选择证据同时改动。', '', '## 当前解释', '', 'TP 前两名 anchor 的 b0 Oracle 为 0.860772，高于 TP 首选加 PC 备用的 0.816058；TP/PC 排序与实际传播质量相关性均弱。这个结果支持继续验证多参考图与检索适配性，尚不能替代 b0–b6 的固定预算对照。', '']
    (R/'report.md').write_text('\n'.join(lines))
    print(json.dumps(res,indent=2),flush=True)

if __name__=='__main__':main()
