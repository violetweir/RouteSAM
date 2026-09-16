"""Post-hoc diagnostics only: never changes predictions or tunes parameters."""
import json
from pathlib import Path
import numpy as np

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'work/kvasir_tp_guided_pc_joint_20260907'
OLD=P/'work/rerun_kvasir_sam3base_test_20260906/final_masks_no_student'
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def idx(rr):return {r['target_id']:r for r in rr}
def delta_stats(d):
    d=np.asarray(d,dtype=float)
    return {'count':len(d),'sum_delta':float(d.sum()),'mean_delta':float(d.mean()) if len(d) else None,'wins':int((d>1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),'losses':int((d< -1e-12).sum()),'positive_delta_sum':float(d[d>0].sum()),'negative_delta_sum':float(d[d<0].sum())}

out={'interpretation':'Post-hoc test diagnostics; oracle choices use GT and are not deployable results. No new selector fitted.'}
scopes=['target_pooling','patch_correspondence','combined']
sel={s:idx(read(OLD/'b0_b6'/s/'per_target_metrics.jsonl')) for s in scopes}
ids=sorted(sel['target_pooling']);tp=sel['target_pooling'];pc=sel['patch_correspondence'];comb=sel['combined']
assert len(ids)==100 and all(set(sel[s])==set(ids) for s in scopes)
out['old_two_route_system']={
    'pc_selected_vs_tp_selected':delta_stats([pc[i]['dice']-tp[i]['dice'] for i in ids]),
    'oracle_between_two_independently_selected_masks':float(np.mean([max(tp[i]['dice'],pc[i]['dice']) for i in ids])),
    'joint_router_minus_tp':delta_stats([comb[i]['dice']-tp[i]['dice'] for i in ids])}
for group,chosen in [('combined_chooses_PC',[i for i in ids if comb[i]['route_mode'].endswith('patch_correspondence')]),('combined_chooses_TP',[i for i in ids if comb[i]['route_mode'].endswith('target_pooling')])]:
    out['old_two_route_system'][group]=delta_stats([comb[i]['dice']-tp[i]['dice'] for i in chosen])
    out['old_two_route_system'][group]['different_route_from_tp_baseline']=sum(comb[i]['route_id']!=tp[i]['route_id'] for i in chosen)

baseline=read(R/'tp_baseline/tp_baseline/propagation_quality_test/propagation_quality.jsonl')
base={(r['target_id'],r['bridge_count']):r for r in baseline}
out['new_joint_anchor_diagnostics']={}
for v in ['joint_v1','joint_v2']:
    rr=read(R/v/v/'propagation_quality_test/propagation_quality.jsonl')
    d={}
    for b in [0,'all']:
        subset=[r for r in rr if b=='all' or r['bridge_count']==b]
        for changed in [False,True]:
            group=[r for r in subset if (r['anchor_id']!=base[r['target_id'],r['bridge_count']]['anchor_id'])==changed]
            d[f'b{b}_anchor_changed_{changed}']=delta_stats([r['gt_dice_evaluation_only']-base[r['target_id'],r['bridge_count']]['gt_dice_evaluation_only'] for r in group])
    out['new_joint_anchor_diagnostics'][v]=d

cache=np.load(P/'work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features/sam3_base_s256_features.npz')
local=np.load(R/'guided_local_scores.npy')
records=read(R/'protocol/merged_manifest.jsonl');index={r['merged_id']:i for i,r in enumerate(records)}
train=[i for i,r in enumerate(records) if r['split']=='train']
out['train_only_score_correlations']=[]
for a,anchor in enumerate(cache['anchor_ids'].tolist()):
    ii=[i for i in train if i!=index[anchor]]
    out['train_only_score_correlations'].append({'anchor_id':anchor,'tp_vs_guided_local_pearson':float(np.corrcoef(cache['cond_target'][a,ii],local[a,ii])[0,1]),'tp_vs_old_pc_pearson':float(np.corrcoef(cache['cond_target'][a,ii],cache['cond_correspondence'][a,ii])[0,1])})
(R/'route_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
