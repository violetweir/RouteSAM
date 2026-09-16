"""Equal-budget support retrieval/path ablation; GT-free route construction."""
from pathlib import Path
import numpy as np
from run_anchor_matrix import P,R as MATRIX,OLD,MODES,CACHE,read,save,jsonl,module,sha

R=MATRIX.parent/'anchor_factorial'

def all_anchor_routes(stage, records, support, patch_mean, cond_scores):
    index={r['merged_id']:i for i,r in enumerate(records)}
    train=np.array([i for i,r in enumerate(records) if r['split']=='train'])
    ids=np.array([records[i]['merged_id'] for i in train])
    sim=(patch_mean@patch_mean.T).astype(np.float64)
    cond=np.asarray(cond_scores,dtype=np.float64)
    anchors=stage.t21.human_pool(support,512)
    targets=sorted([r for r in records if r['split']=='validation'],key=lambda r:r['merged_id'])
    result={}
    for ai,anchor in enumerate(anchors):
        ranks=[train[np.lexsort((ids,cond[ai,train],sim[train,tail]))[::-1]] for tail in range(len(records))]
        for target in targets:
            ti=index[target['merged_id']];forbidden={ti,index[anchor['anchor_id']]}
            paths=np.empty((1,0),dtype=np.int64)
            for depth in range(7):
                if depth:
                    expanded=[]
                    for path in paths:
                        used=forbidden|set(path.tolist());tail=ti if len(path)==0 else int(path[0])
                        ranked=[int(i) for i in ranks[tail] if int(i) not in used][:32]
                        expanded.extend([[i,*path] for i in ranked])
                    paths=np.asarray(expanded,dtype=np.int64)
                nodes=np.concatenate([paths,np.full((len(paths),1),ti,dtype=np.int64)],axis=1)
                values=np.concatenate([cond[ai,nodes],sim[nodes[:,:-1],nodes[:,1:]]],axis=1)
                mins,means=values.min(1),values.mean(1)
                order=np.lexsort((-means,-mins))[:32];paths=paths[order]
                score=(float(mins[order[0]]),float(means[order[0]]))
                result[target['merged_id'],ai,depth]=stage.make_route(target,depth,anchor,paths[0].tolist(),score,records)
        print('anchor routes',ai+1,'/8',flush=True)
    return result

def main():
    assert not (R/'GENERATION_COMPLETE.json').exists()
    R.mkdir(parents=True,exist_ok=True)
    stage=module('factorial_stage',MATRIX/'code/stage1_feature_knn_routes.py')
    records=read(MATRIX/'protocol/merged_manifest.jsonl');support=read(MATRIX/'protocol/support_manifest.jsonl')
    cache=np.load(CACHE);index={r['merged_id']:i for i,r in enumerate(records)}
    anchors=cache['anchor_ids'].tolist();targets=sorted(r['merged_id'] for r in records if r['split']=='validation')
    computed={};audit={}
    for mode,key in zip(MODES,['cond_target','cond_correspondence']):
        computed[mode]=all_anchor_routes(stage,records,support,cache['patch_mean'],cache[key])
        old=read(OLD/mode/'validation_pool0_stage1/routes.jsonl')
        score_errors=[]
        for oldrow in old:
            candidates=[computed[mode][oldrow['target_id'],ai,oldrow['bridge_count']] for ai in range(8)]
            selected=max(candidates,key=lambda r:(r['path_bottleneck_similarity'],r['path_mean_similarity'],r['anchor_id']))
            assert selected['route_id']==oldrow['route_id'],(mode,oldrow['route_id'],selected['route_id'])
            score_errors.extend(abs(selected[k]-oldrow[k]) for k in ['path_bottleneck_similarity','path_mean_similarity'])
        assert len(old)==700 and max(score_errors)<1e-6
        audit[mode]={'identical_route_ids':len(old),'max_score_error':max(score_errors)}
    save(R/'historical_route_identity_audit.json',audit)
    groups={'B':[],'C':[],'D':[]};decisions=[]
    for target in targets:
        ti=index[target]
        tp_order=sorted(range(8),key=lambda a:(float(cache['cond_target'][a,ti]),anchors[a]),reverse=True)
        pc_order=sorted(range(8),key=lambda a:(float(cache['cond_correspondence'][a,ti]),anchors[a]),reverse=True)
        primary=tp_order[0];b=tp_order[1];c=next(a for a in pc_order if a!=primary)
        decisions.append(dict(target_id=target,primary_anchor=anchors[primary],B_anchor=anchors[b],C_D_anchor=anchors[c],
            C_anchor_TP_rank=tp_order.index(c)+1, B_anchor_PC_rank=pc_order.index(b)+1))
        for variant,ai,mode in [('B',b,MODES[0]),('C',c,MODES[0]),('D',c,MODES[1])]:
            for depth in range(7):
                r=computed[mode][target,ai,depth]
                groups[variant].append({**r,'experiment_variant':variant,'primary_b0_anchor':anchors[primary]})
    for variant,rows in groups.items():
        assert len(rows)==700 and all(r['anchor_id']!=r['primary_b0_anchor'] for r in rows)
        jsonl(R/variant/'routes.jsonl',rows)
    assert all(c['anchor_id']==d['anchor_id'] for c,d in zip(groups['C'],groups['D']))
    assert all(c['route_id']==d['route_id'] for c,d in zip(groups['C'],groups['D']) if c['bridge_count']==0)
    jsonl(R/'anchor_choices.jsonl',decisions)
    save(R/'config.json',dict(split='validation',targets=100,primary='Original TP 7 unchanged',auxiliary_count=7,total_count=14,
        B='TP alternate anchor, TP path',C='PC alternate anchor, TP path',D='Same anchor as C, PC path',
        bridge_counts=list(range(7)),beam_width=32,knn_feature='patch_mean',feature_size=256,canvas=256,internal_size=1008,
        target_gt_used_for_search_or_inference=False,train_support_count=8,unlabeled_train_count=792,
        primary_reference='Frozen TP independent Router; shared nested gain selector protocol to follow',
        original_router='Secondary identical five-fold Ridge protocol for candidate comparison',
        no_test_at_this_stage=True))
    save(R/'GENERATION_COMPLETE.json',dict(aux_routes_per_group=700,unique_aux_routes=len({r['route_id'] for rr in groups.values() for r in rr}),
        B_C_same_anchor_targets=sum(r['B_anchor']==r['C_D_anchor'] for r in decisions),
        C_D_different_routes=sum(c['route_id']!=d['route_id'] for c,d in zip(groups['C'],groups['D'])),
        code_sha256=sha(__file__)))
    print((R/'GENERATION_COMPLETE.json').read_text(),flush=True)

if __name__=='__main__':main()
