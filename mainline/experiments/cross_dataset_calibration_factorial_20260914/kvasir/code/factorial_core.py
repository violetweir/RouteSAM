"""BUSI 2x2 消融：原始/中心化目标TP分数 x top1/top2，同一legacy Router。"""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import sys, json, hashlib, shutil, time, collections, importlib.util, traceback
import numpy as np
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E=P/'new_project/experiments'
B=E/'busi_auto5_tp_1pct_20260913'
S=E/'busi_calibrated_multi_anchor_20260913'
R=E/'busi_calibration_factorial_20260914'
MODE='sam3enc_anchor_conditioned_target_pooling'
GROUPS=['raw_top1','centered_top1','raw_top2','centered_top2','original_per_bridge']
PAIRS=[('centered_top1','raw_top1'),('centered_top2','raw_top2'),
       ('raw_top2','raw_top1'),('centered_top2','centered_top1'),
       ('centered_top2','original_per_bridge')]

def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def jl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def status(stage,**kw): save(R/'status.json',dict(stage=stage,time=time.time(),**kw));print(stage,kw,flush=True)
def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def prev():
    m=mod('frozen_previous',R/'code/previous_pipeline.py');m.R=R;return m
def qp(root,split):return root/f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl'
def rp(split):return R/f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl'
def bitmap(p):return np.asarray(Image.open(p).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
def metric(a,b):
    i=int((a&b).sum());s=int(a.sum())+int(b.sum());return dict(dice=2*i/s if s else 1.,iou=i/(s-i) if s-i else 1.)

def prepare():
    R.mkdir(exist_ok=False)
    for n in ['code','protocol','logs']:(R/n).mkdir()
    shutil.copy2(__file__,R/'pipeline.py')
    shutil.copy2(S/'pipeline.py',R/'code/previous_pipeline.py')
    for n in ['router.py','stage1_feature_knn_routes.py','eval_route_propagation_quality.py','run_t21_dynamic_pseudovideo.py']:
        shutil.copy2(S/'code'/n,R/'code'/n)
    for n in ['merged_manifest.jsonl','support_manifest.jsonl']:shutil.copy2(S/'protocol'/n,R/'protocol'/n)
    shutil.copy2(S/'calibration_frozen.json',R/'calibration_frozen.json')
    shutil.copy2(S/'folds_frozen.json',R/'folds_frozen.json')
    save(R/'predefined_policy.json',dict(groups=GROUPS,train=517,validation=64,test=66,anchors=5,
      raw='rank anchors by raw TP(anchor,target), descending; anchor ID ascending breaks ties',
      centered='rank anchors by TP(anchor,target) minus mean TP(anchor,517 train), same as previous frozen calibration',
      topk='top1 or top2 anchors fixed for the target across all bridge lengths; retain all b0-b6 for each',
      original='additional historical control: independently choose the best anchor/path for each bridge length',
      invariant='same 5 automatic anchors, same split, SAM3-base, box prompt, no text, canvas256, per-anchor raw path objective, beam32',
      router='all five pools refit legacy28 Ridge(alpha=1), same validation image-grouped folds; no peer or calibration features; no new hyperparameter search',
      objective='sum squared residuals + alpha times squared weights, as historical Ridge',
      ordering='rank then bridge within raw/centered pool; original bridge order; argmax first candidate on tied Router score',
      calibration_train_GT_read=False,validation_labels_extra_to_1percent=True,prior_test_seen=True,
      test_policy='evaluate all four prespecified arms and original control; freeze all five models and choices before reading test GT; no test-driven winner selection',
      statistics='10000 image-paired bootstrap samples seed2026; intervals exploratory, no multiplicity correction',
      timing='historical row seconds include forward/return and serialization; sum per pool is descriptive, not a controlled end-to-end speed benchmark'))
    audits={}; frozen={}; inputs={}
    for split,n in [('validation',64),('test',66)]:
        allroutes=read(S/f'{split}_all_anchor_routes.jsonl')
        bytarget=collections.defaultdict(list)
        for row in allroutes:bytarget[row['target_id']].append(row)
        assert len(bytarget)==n
        old=read(B/f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl')
        old_by=collections.defaultdict(list)
        for row in old:old_by[row['target_id']].append(row)
        members={g:{} for g in GROUPS};needed={}
        for tid,rows in sorted(bytarget.items()):
            anchors={r['anchor_id']:r for r in rows};assert len(anchors)==5 and len(rows)==35
            raw=sorted(anchors,key=lambda a:(-anchors[a]['anchor_target_raw'],a))
            centered=sorted(anchors,key=lambda a:(-anchors[a]['anchor_target_centered'],a))
            assert all(anchors[a]['calibrated_anchor_rank']==i for i,a in enumerate(centered))
            for kind,ranking in [('raw',raw),('centered',centered)]:
                for k in [1,2]:
                    rs=sorted([r for r in rows if r['anchor_id'] in ranking[:k]],key=lambda r:(ranking.index(r['anchor_id']),r['bridge_count']))
                    assert len(rs)==7*k;members[f'{kind}_top{k}'][tid]=[r['route_id'] for r in rs]
                    for row in rs:needed[row['route_id']]=row
            rs=sorted(old_by[tid],key=lambda r:r['bridge_count']);assert len(rs)==7
            members['original_per_bridge'][tid]=[r['route_id'] for r in rs]
            all_by={r['route_id']:r for r in rows}
            for row in rs:needed[row['route_id']]=all_by[row['route_id']]
        oldq={r['route_id']:r for r in read(qp(S,split))}
        reused=[]
        for rid,row in sorted(needed.items()):
            if rid in oldq:
                q=oldq[rid];assert q['status']=='success' and not q['target_gt_used_for_search_or_inference']
                assert sha(q['forward_mask_path'])==q['forward_mask_sha256']
                for key in ['anchor_id','bridge_ids','anchor_mask_sha256','anchor_box_xywh_normalized']:
                    assert q[key]==row[key],(key,rid)
                reused.append({**{k:v for k,v in q.items() if not k.startswith('gt_')},**row})
        jl(rp(split),list(needed.values()));jl(qp(R,split),reused)
        frozen[split]=members
        audits[split]=dict(targets=n,candidate_union=len(needed),reused=len(reused),missing=len(needed)-len(reused))
        for f in [S/f'{split}_all_anchor_routes.jsonl',qp(S,split),B/f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl']:
            inputs[str(f)]=sha(f)
    save(R/'pool_membership_frozen.json',dict(time=time.time(),groups=frozen,test_GT_read=False))
    save(R/'reuse_audit.json',audits)
    for f in list((R/'code').glob('*.py'))+list((R/'protocol').glob('*'))+[R/'pipeline.py',R/'calibration_frozen.json',R/'folds_frozen.json']:
        inputs[str(f)]=sha(f)
    save(R/'input_hashes.json',inputs)
    assert audits['validation']['missing']==0
    status('prepared',audit=audits)

def extract(split,with_gt):
    feature=mod('legacy_feature',R/'code/router.py')
    rows={r['route_id']:r for r in read(qp(R,split))}
    members=json.loads((R/'pool_membership_frozen.json').read_text())['groups'][split]
    allids=sorted(members[GROUPS[0]])
    for r in rows.values():
        assert r['status']=='success' and not r['target_gt_used_for_search_or_inference']
        assert sha(r['forward_mask_path'])==r['forward_mask_sha256']
    values={};gt={}
    if with_gt:
        for r in rows.values():
            tid=r['target_id']
            if tid not in gt:gt[tid]=bitmap(r['target_mask_path_evaluation_only']);assert gt[tid].any()
            values[r['route_id']]=metric(bitmap(r['forward_mask_path']),gt[tid])
    data={}
    for group in GROUPS:
        rs=[[rows[rid] for rid in members[group][tid]] for tid in allids]
        x=np.array([[feature.feature_vector(r,False) for r in rr] for rr in rs],dtype=float)
        assert x.shape[2]==28
        data[group]=dict(ids=allids,rows=rs,raw=x,
             y=np.array([[values[r['route_id']]['dice'] for r in rr] for rr in rs]) if with_gt else None)
    return data,values

def paired(vectors):
    rng=np.random.default_rng(2026);n=len(next(iter(vectors.values())));indices=rng.integers(0,n,(10000,n));out={}
    for a,b in PAIRS:
        delta=np.asarray(vectors[a])-np.asarray(vectors[b]);boot=delta[indices].mean(1)
        out[a+'_minus_'+b]=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),improved=int((delta>1e-12).sum()),worsened=int((delta < -1e-12).sum()))
    d=(np.asarray(vectors['centered_top2'])-np.asarray(vectors['raw_top2']))-(np.asarray(vectors['centered_top1'])-np.asarray(vectors['raw_top1']))
    out['interaction']=dict(mean=float(d.mean()),ci95=np.quantile(d[indices].mean(1),[.025,.975]).tolist())
    return out

def validate():
    if (R/'models_frozen.json').exists():return
    status('fitting_validation_routers_cpu')
    m=prev();data,values=extract('validation',True);ids=data[GROUPS[0]]['ids'];assert len(ids)==64
    folds=json.loads((R/'folds_frozen.json').read_text());f=np.array([folds[t] for t in ids]);ix=np.arange(64)
    models={};summary={};vectors={};selections={}
    for group,d in data.items():
        k=d['raw'].shape[1]//7;config=(k,'legacy',1.)
        scores=m.cv(d,ix,f,config);choice=scores.argmax(1);y=d['y'][ix,choice];vectors[group]=y.tolist()
        oracle=float(d['y'].max(1).mean());models[group]=m.fit(d,ix,config)
        selected=[d['rows'][i][j] for i,j in enumerate(choice)]
        selections[group]=[dict(target_id=r['target_id'],route_id=r['route_id'],anchor_id=r['anchor_id'],bridge_count=r['bridge_count'],dice=values[r['route_id']]['dice']) for r in selected]
        summary[group]=dict(candidates_per_target=d['raw'].shape[1],oof_dice=float(y.mean()),oracle_dice=oracle,oracle_gap=oracle-float(y.mean()),
             selected_anchors=dict(collections.Counter(r['anchor_id'] for r in selected)),
             candidate_anchors=dict(collections.Counter(r['anchor_id'] for rr in d['rows'] for r in rr)),
             fixed_rank1_b0_b6=d['y'][:,:7].mean(0).tolist())
    expected={'original_per_bridge':.6176338160406722,'centered_top1':.728772240305812,'centered_top2':.7389582880918044}
    for group,value in expected.items():assert abs(summary[group]['oof_dice']-value)<1e-10,(group,summary[group])
    save(R/'validation_results.json',dict(n=64,groups=summary,paired_oof=paired(vectors),per_target_oof=selections,
          note='All arms prespecified, same alpha1, no hyperparameter search; OOF uses validation GT only.'))
    save(R/'models_frozen.json',dict(time=time.time(),models=models,folds_sha256=sha(R/'folds_frozen.json'),
          membership_sha256=sha(R/'pool_membership_frozen.json'),validation_quality_sha256=sha(qp(R,'validation')),test_GT_read=False))
    status('validation_complete',results=summary)

def evaluate():
    status('freezing_test_choices_without_GT')
    m=prev();data,_=extract('test',False);n=66;ix=np.arange(n)
    frozen=json.loads((R/'models_frozen.json').read_text());choices={}
    for group,d in data.items():
        assert len(d['ids'])==n
        js=m.predict(d,ix,frozen['models'][group]).argmax(1)
        choices[group]=[d['rows'][i][j]['route_id'] for i,j in enumerate(js)]
    save(R/'test_choices_frozen.json',dict(time=time.time(),models_sha256=sha(R/'models_frozen.json'),choices=choices,test_GT_read=False))
    status('scoring_frozen_test_choices')
    _,values=extract('test',True);metrics={};vectors={};per={}
    for group,d in data.items():
        byid={r['route_id']:r for rr in d['rows'] for r in rr};rs=[byid[rid] for rid in choices[group]]
        per[group]=[dict(target_id=r['target_id'],route_id=r['route_id'],anchor_id=r['anchor_id'],bridge_count=r['bridge_count'],mask_path=r['forward_mask_path'],**values[r['route_id']]) for r in rs]
        y=[values[r['route_id']]['dice'] for r in rs];vectors[group]=y
        oracle=float(np.mean([max(values[r['route_id']]['dice'] for r in rr) for rr in d['rows']]))
        times=[sum(float(r['seconds']) for r in rr) for rr in d['rows']]
        metrics[group]=dict(candidates_per_target=d['raw'].shape[1],dice=float(np.mean(y)),iou=float(np.mean([values[r['route_id']]['iou'] for r in rs])),
          oracle_dice=oracle,oracle_gap=oracle-float(np.mean(y)),
          selected_anchors=dict(collections.Counter(r['anchor_id'] for r in rs)),selected_bridges=dict(collections.Counter(r['bridge_count'] for r in rs)),
          candidate_anchors=dict(collections.Counter(r['anchor_id'] for rr in d['rows'] for r in rr)),
          historical_candidate_seconds_per_target_mean=float(np.mean(times)),
          fixed_rank1_b0_b6=[float(np.mean([values[rr[b]['route_id']]['dice'] for rr in d['rows']])) for b in range(7)])
        dest=R/group/'masks';dest.mkdir(parents=True,exist_ok=True)
        for r in rs:shutil.copy2(r['forward_mask_path'],dest/(r['target_id'].replace('::','__')+'.png'))
    assert abs(metrics['original_per_bridge']['dice']-.5668083894947364)<1e-10
    assert abs(metrics['centered_top2']['dice']-.7521975757993846)<1e-10
    validation=json.loads((R/'validation_results.json').read_text())
    out=dict(validation={k:v for k,v in validation.items() if k!='per_target_oof'},test=metrics,paired_test=paired(vectors),reuse=json.loads((R/'reuse_audit.json').read_text()))
    save(R/'results.json',out);save(R/'test_per_target.json',per);save(R/'test_candidate_metrics.json',values)
    for f,h in json.loads((R/'input_hashes.json').read_text()).items():assert sha(f)==h,(f,'changed')
    save(R/'completion_audit.json',dict(all_test_66=True,prespecified_four_arms_plus_original=True,same_alpha_and_folds=True,
         test_choices_frozen_before_GT_read=True,original_and_centered_top2_reproduced=True,inputs_unchanged=True))
    lines=['# BUSI：分数校准 × 参考数量四组消融','',
      '固定原5张自动参考图，train517/val64/test66，SAM3-base，GT box，无文本，256。每组独立拟合相同legacy28 Ridge(alpha=1)，沿用同一validation图像级5折。','',
      '| 组别 | 每图候选 | val OOF Dice | test Dice | test Oracle | Oracle差距 |',
      '|---|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        x=metrics[group];v=validation['groups'][group]
        lines.append(f"| {group} | {x['candidates_per_target']} | {v['oof_dice']:.6f} | {x['dice']:.6f} | {x['oracle_dice']:.6f} | {x['oracle_gap']:.6f} |")
    lines+=['','## 配对比较','', '| 对比 | test Dice增量 | 95% bootstrap区间 |','|---|---:|---|']
    for name,x in out['paired_test'].items():lines.append(f"| {name} | {x['mean']:.6f} | [{x['ci95'][0]:.6f}, {x['ci95'][1]:.6f}] |")
    lines+=['','原始top1按目标原始TP分数选定一张参考并固定用于全部b；历史original_per_bridge每个b独立选参考，两者不能混为同一配置。',
      'top2共有14个候选。固定rank1 b0–b6只是对应分数排名第一参考的候选，不代表每个b单独训练了Router。',
      '四组预先固定，没有根据test选择新的超参数。test此前已用于诊断，本轮为已使用基准上的消融。95%区间为逐图配对探索性区间，未做多重比较校正。',
      '计时仅将各候选历史seconds相加，不是控制硬件负载、缓存和并行方式的端到端测速；候选数量7→14意味着候选传播预算翻倍。',
      '详细每图结果、源参考使用占比、冻结模型和候选成员见同目录JSON。']
    (R/'report.md').write_text('\n'.join(lines)+'\n');(R/'COMPLETE').touch();status('complete',test=metrics)

def run():
    validate()
    m=prev();m.propagate('test')
    evaluate()

if __name__=='__main__':
    try:
        action=sys.argv[1]
        if action=='prepare':prepare()
        elif action=='run':run()
        elif action=='validate':validate()
        elif action=='evaluate':evaluate()
        else:raise ValueError(action)
    except BaseException:
        if R.exists():status('failed',error=traceback.format_exc())
        raise
