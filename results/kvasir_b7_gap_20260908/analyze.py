from pathlib import Path
import json, collections, hashlib, shutil
import numpy as np
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
OLD=P/'work/rerun_c0_256_sam3knn_s256_base'
NEW=P/'work/kvasir_tp_student_mainline_20260907'
OUT=P/'work/kvasir_b7_gap_20260908'
MODES=['sam3enc_anchor_conditioned_target_pooling','sam3enc_anchor_conditioned_patch_correspondence']
def read(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def path(p):
    p=Path(p); return p if p.is_absolute() else P/p
def mask(p):
    a=np.asarray(Image.open(path(p)).convert('L'))>127
    assert a.shape==(256,256)
    return a
def dice(a,b):
    den=a.sum()+b.sum(); return float(2*np.logical_and(a,b).sum()/den) if den else 1.
def score(qr,qm,qs): return np.maximum(qr,1e-6)**.2*np.maximum(qm,1e-6)**.4*np.maximum(qs,1e-6)**.4
def pick(s,qm,qr,rows): return max(range(len(rows)),key=lambda i:(s[i],qm[i],qr[i],rows[i]['route_id']))
def ci(delta):
    d=np.array(delta); rng=np.random.default_rng(2026)
    boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
    return [float(x) for x in np.quantile(boot,[.025,.975])]

OUT.mkdir(parents=True,exist_ok=True)
save('protocol.json',{'purpose':'Fixed 2x2 student/candidate-pool controls and validation-only scoring probes.',
 'matrix':['old_X3_TP','old_X3_dual','new_X3_TP','new_X3_dual'],
 'score':'q_return^.2 * q_multi^.4 * q_model^.4; clamp each at 1e-6; q_multi excludes self',
 'validation_probes':['soft_model','no_multi','cluster_multi'],
 'cluster_multi':'Greedy route-id-ordered mask clustering, Dice>=0.95; mean within each peer cluster, then average clusters. Exclude candidate itself.',
 'soft_model':'Soft Dice between binary candidate and uint16-decoded frozen X3 probability.',
 'no_multi':'q_return^(1/3) * q_model^(2/3)',
 'policy':'No test tuning. Test only fixed historical-formula matrix. Validation probes exploratory: X3 checkpoint and pseudo-pool router have already used validation. No claim of untouched validation/OOF.',
 'student_checkpoints':'Previously frozen ordinary validation-best, no retraining',
 'splits':'Unchanged original 100 validation / 100 test; same eight train anchors',
 'gt':'Selection completed before evaluation; oracle and term decomposition diagnostic only'})
all_results={}; all_records=[]; identity={}
for split in ['validation','test']:
    groups=collections.defaultdict(list)
    for mode in MODES:
        for r in read(OLD/f'stage1_feature_knn_b0_b6/{mode}/propagation_quality_{split}/propagation_quality.jsonl'):
            r['mode']='TP' if mode==MODES[0] else 'PC'; groups[r['target_id']].append(r)
    assert len(groups)==100
    students={v:{r['merged_id']:r for r in read(root/f'predictions/X3_best/student_predictions_{split}.jsonl')} for v,root in [('old',OLD),('new',NEW)]}
    new_tp={r['route_id']:r for r in read(NEW/f'quality/anchor_conditioned_target_pooling/propagation_quality_{split}/propagation_quality.jsonl')}
    identity[split]={'tp_route_count':len(new_tp),'identical_masks':0,'identical_q_cycle':0}
    rows_results=collections.defaultdict(list); candidate_records=[]
    for target,rr in sorted(groups.items()):
        rr.sort(key=lambda r:(r['mode']!='TP',r['bridge_count'],r['route_id']))
        assert len(rr)==14 and [r['bridge_count'] for r in rr[:7]]==list(range(7))
        mm=[mask(r['forward_mask_path']) for r in rr]
        for r,m in zip(rr[:7],mm[:7]):
            n=new_tp[r['route_id']]; assert np.array_equal(m,mask(n['forward_mask_path']))
            assert r['q_cycle']==n['q_cycle']
            identity[split]['identical_masks']+=1;identity[split]['identical_q_cycle']+=1
        pair=np.array([[dice(a,b) for b in mm] for a in mm])
        binaries={v:mask(students[v][target]['student_binary_mask']) for v in students}
        qs={v:np.array([dice(m,binaries[v]) for m in mm]) for v in students}
        pending=[]
        for pool,n in [('TP',7),('dual',14)]:
            sub=rr[:n]; qr=np.array([r['q_cycle'] for r in sub]); qm=(pair[:n,:n].sum(1)-1)/(n-1)
            for version in ['old','new']:
                ss=score(qr,qm,qs[version][:n]); j=pick(ss,qm,qr,sub)
                pending.append((f'{version}_X3_{pool}',j,n,qr,qm,qs[version][:n],ss))
            if split=='validation':
                im=Image.open(path(students['new'][target]['student_probability_map'])); raw=np.asarray(im)
                assert raw.max()>255 and raw.max()<=65535
                prob=raw.astype(np.float64)/65535.
                soft=np.array([float(2*(m*prob).sum()/max(m.sum()+prob.sum(),1e-12)) for m in mm[:n]])
                clusters=[]
                for k in sorted(range(n),key=lambda k:sub[k]['route_id']):
                    match=next((c for c in clusters if pair[k,c[0]]>=.95),None)
                    if match is None: clusters.append([k])
                    else: match.append(k)
                cm=[]
                for k in range(n):
                    peer_clusters=[[i for i in c if i!=k] for c in clusters]
                    cm.append(np.mean([np.mean(pair[k,c]) for c in peer_clusters if c]))
                cm=np.array(cm)
                variants={'soft_model':(score(qr,qm,soft),qm,soft),
                  'no_multi':(np.maximum(qr,1e-6)**(1/3)*np.maximum(qs['new'][:n],1e-6)**(2/3),qm,qs['new'][:n]),
                  'cluster_multi':(score(qr,cm,qs['new'][:n]),cm,qs['new'][:n])}
                for name,(ss,multi,model) in variants.items():
                    pending.append((f'probe_{name}_{pool}',pick(ss,multi,qr,sub),n,qr,multi,model,ss))
        # Evaluation starts only after all decisions above. Never feed GT to score/pick.
        gt=np.asarray(Image.open(rr[0]['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
        gd=np.array([dice(m,gt) for m in mm])
        for r,d in zip(rr,gd): assert abs(d-r['gt_dice_evaluation_only'])<1e-12
        for name,j,n,qr,qm,model,ss in pending:
            best_d=gd[:n].max()
            oracle=max((k for k in range(n) if abs(gd[k]-best_d)<1e-12),key=lambda k:ss[k])
            inter=int((mm[j]&gt).sum()); union=int((mm[j]|gt).sum())
            wr,wm,ws=(1/3,0.,2/3) if name.startswith('probe_no_multi') else (.2,.4,.4)
            row={'split':split,'target_id':target,'method':name,'dice':float(gd[j]),'iou':inter/union if union else 1.,'oracle':float(gd[oracle]),
              'regret':float(gd[oracle]-gd[j]),'index':j,'route_id':rr[j]['route_id'],'mode':rr[j]['mode'],'bridge':rr[j]['bridge_count'],
              'oracle_index':oracle,'oracle_mode':rr[oracle]['mode'],'oracle_rank':int(np.sum(ss>ss[oracle]))+1,
              'q_return':float(qr[j]),'q_multi':float(qm[j]),'q_model':float(model[j]),'score':float(ss[j]),
              'source_mask_path':str(path(rr[j]['forward_mask_path'])),
              'log_score_advantage_selected_over_oracle':{'return':float(wr*np.log(max(qr[j],1e-6)/max(qr[oracle],1e-6))),
                  'multi':float(wm*np.log(max(qm[j],1e-6)/max(qm[oracle],1e-6))),
                  'model':float(ws*np.log(max(model[j],1e-6)/max(model[oracle],1e-6)))}}
            assert abs(sum(row['log_score_advantage_selected_over_oracle'].values())-np.log(ss[j]/ss[oracle]))<1e-10
            if name=='new_X3_dual':
                dest=OUT/f'new_X3_dual_{split}/masks';dest.mkdir(parents=True,exist_ok=True)
                final=dest/(target.replace('::','__')+'.png');shutil.copy2(row['source_mask_path'],final)
                row['final_mask_path']=str(final);row['mask_sha256']=hashlib.sha256(final.read_bytes()).hexdigest()
                assert np.array_equal(mask(final),mm[j])
            rows_results[name].append(row);all_records.append(row)
        candidate_records.append({'target_id':target,'tp_oracle':float(gd[:7].max()),'pc_oracle':float(gd[7:].max()),'dual_oracle':float(gd.max()),
          'candidates':[{'route_id':r['route_id'],'mode':r['mode'],'bridge':r['bridge_count'],'dice_diagnostic_only':float(gd[k]),
           'q_return':r['q_cycle'],'q_model_old':float(qs['old'][k]),'q_model_new':float(qs['new'][k]),'q_multi_dual':float((pair[k].sum()-1)/13)} for k,r in enumerate(rr)]})
    result={}
    for name,rows in rows_results.items():
        baseline=rows_results['new_X3_TP'];delta=np.array([r['dice']-b['dice'] for r,b in zip(rows,baseline)])
        regret=np.array([r['regret'] for r in rows]); worst=sorted(rows,key=lambda r:-r['regret'])
        positive=[r for r in rows if r['regret']>1e-9]
        result[name]={'count':len(rows),'dice':float(np.mean([r['dice'] for r in rows])), 'iou':float(np.mean([r['iou'] for r in rows])),
          'oracle':float(np.mean([r['oracle'] for r in rows])), 'gap':float(regret.mean()),
          'delta_vs_new_TP':float(delta.mean()),'paired_ci95_vs_new_TP':ci(delta),'win_tie_loss_vs_new_TP':[int((delta>1e-9).sum()),int((abs(delta)<=1e-9).sum()),int((delta< -1e-9).sum())],
          'selected_PC':sum(r['mode']=='PC' for r in rows),'oracle_hits':int((regret<1e-9).sum()),'regret_gt_005':int((regret>.05).sum()),
          'top10_gap_share':float(sum(r['regret'] for r in worst[:10])/max(regret.sum(),1e-12)),
          'oracle_in_score_top3':sum(r['oracle_rank']<=3 for r in rows),
          'largest_term_against_oracle_on_misses':dict(collections.Counter(max(r['log_score_advantage_selected_over_oracle'],key=r['log_score_advantage_selected_over_oracle'].get) for r in positive)),
          'worst_10':[{k:r[k] for k in ['target_id','dice','oracle','regret','mode','oracle_mode']} for r in worst[:10]]}
    pc_gain=np.array([c['dual_oracle']-c['tp_oracle'] for c in candidate_records])
    result['candidate_diagnostics']={'pc_improves_oracle_count':int((pc_gain>1e-9).sum()),'pc_gain_mean_all100':float(pc_gain.mean()),'pc_gain_top10_share':float(np.sort(pc_gain)[-10:].sum()/max(pc_gain.sum(),1e-12))}
    if split=='test':
        assert abs(result['old_X3_dual']['dice']-.8954317676013283)<1e-12
        assert abs(result['new_X3_TP']['dice']-.8926393880148344)<1e-12
    all_results[split]=result
    (OUT/f'new_X3_dual_{split}/selected_masks.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows_results['new_X3_dual']))
    (OUT/f'candidate_diagnostics_{split}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in candidate_records))
    print(split,json.dumps({k:{a:v[a] for a in ['dice','oracle','gap','delta_vs_new_TP']} for k,v in result.items() if 'dice' in v}),flush=True)
save('candidate_identity.json',identity);save('results.json',all_results)
(OUT/'per_target.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in all_records))
(OUT/'COMPLETE').write_text('Fixed historical-formula controls and validation-only probes complete.\n')
