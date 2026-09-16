"""Frozen B pool: read-only tracker evidence and nested target-level selection."""
import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
ROOT=P/'work/kvasir_rethink_20260908'
R=ROOT/'b_target_quality'
FACTOR=ROOT/'anchor_factorial'
MATRIX=ROOT/'anchor_matrix'
TP=P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6/sam3enc_anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl'
AUX=FACTOR/'B/propagation_quality.jsonl'
CKPT=Path('/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt')

def read(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []
def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def jsonl(p,rr):
    Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
def append(p,r):
    with Path(p).open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def resolve(p):
    p=Path(p);return p if p.is_absolute() else P/p
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def status(state,**kw):
    save(R/'pipeline_status.json',dict(state=state,time=datetime.datetime.now().astimezone().isoformat(),**kw))

def setup():
    assert not (R/'config.json').exists()
    (R/'code').mkdir(parents=True,exist_ok=True)
    for f in ['base_gate.py','run_candidate_rerank.py','router.py']:
        shutil.copy2(FACTOR/'analysis_code'/f,R/'code'/f)
    shutil.copy2(__file__,R/'code/b_target_quality.py')
    rr=read(TP)+read(AUX);assert len(rr)==1400
    sources={}
    for row in rr:
        assert row['target_split']=='validation' and row['trace_candidate_count_max']==1
        assert sha(resolve(row['forward_mask_path']))==row['forward_mask_sha256']
        if row['route_id'] in sources:assert sources[row['route_id']]['forward_mask_sha256']==row['forward_mask_sha256']
        sources[row['route_id']]=row
    rows=sorted(sources.values(),key=lambda r:(r['target_id'],r['bridge_count'],r['route_id']))
    jsonl(R/'frozen_routes.jsonl',rows)
    sourcefiles=[TP,AUX,MATRIX/'code/eval_route_propagation_quality.py',MATRIX/'code/run_t21_dynamic_pseudovideo.py',
        Path('/Data_8TB/lht/sam3/sam3/model/sam3_tracker_base.py'),Path('/Data_8TB/lht/sam3/sam3/model/sam3_video_inference.py')]
    config=dict(split='validation',targets=100,primary='target_quality',
        variants={'legacy':'Existing 15 gain features, exact replay',
            'remove_object_scores':'First 13 legacy features, remove old object score and difference',
            'target_quality':'First 13 legacy + target predicted IoU, presence sigmoid, low-res mask stability; each represented by TP absolute and auxiliary minus TP'},
        feature_counts={'legacy':15,'remove_object_scores':13,'target_quality':19},
        model='Same target-weighted Ridge gain regression',ridges=[10.,100.],thresholds=[.02,.05,.10],fallback='Exact independent TP mask',
        folds='5 outer seed2026; 4 inner seed2027; 3 crossfit experts seed2028; grouped by target',
        experts='Original independent Router fitted identically; new features only enter gain gate',
        new_features=dict(predicted_iou='max tracker mask-head IoU estimate, not true IoU',
            presence='sigmoid tracker object_score_logits on target frame, not mask accuracy',
            stability='count(low_res_logits > +1)/count(low_res_logits > -1); zero if denominator empty',
            missing='zero for all three if target mask empty or no head output; keep explicit extraction flags'),
        candidate_slots=1400,unique_routes=len(rows),canvas=256,internal_size=1008,checkpoint=str(CKPT),
        checkpoint_sha256=sha(CKPT),mask_predictions='Must match every frozen mask pixel exactly; no new masks selected during extraction',
        gpu=1,allocator_fraction=.25,new_test=False,students=False,target_gt_in_extraction=False,
        comparison='Primary target_quality vs both exact legacy B and original TP; controls not used to select an outer-fold winner',
        future_test_gate='Primary gains >= .003 vs legacy B and TP, paired 95% lower bounds >0 for both, at least 3 positive outer folds; no automatic test in this run',
        source_sha256={str(p):sha(p) for p in sourcefiles},code_sha256={p.name:sha(p) for p in (R/'code').glob('*.py')})
    assert config['checkpoint_sha256']=='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
    save(R/'config.json',config);status('prepared',unique_routes=len(rows));print(json.dumps(config,indent=2),flush=True)

def extract(limit=0):
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_video_model
    os.chdir(P);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    config=json.loads((R/'config.json').read_text())
    for p,digest in config['source_sha256'].items():assert sha(p)==digest,p
    rows=read(R/'frozen_routes.jsonl');done={r['route_id']:r for r in read(R/'evidence.jsonl')}
    pending=[r for r in rows if r['route_id'] not in done]
    if limit:pending=pending[:limit]
    ev=module('b_quality_ev',MATRIX/'code/eval_route_propagation_quality.py')
    model=build_sam3_video_model(checkpoint_path=str(CKPT),load_from_HF=False,device='cuda',compile=False);model.eval()
    assert model.image_size==1008
    original_step=model.tracker.track_step;original_heads=model.tracker._forward_sam_heads
    collector={'frame':None,'head_calls':[],'step_calls':[]}
    def step(*args,**kwargs):
        old=collector['frame'];frame=kwargs.get('frame_idx',args[0] if args else None)
        collector['frame']=int(frame)
        collector['step_calls'].append(int(frame))
        try:return original_step(*args,**kwargs)
        finally:collector['frame']=old
    def heads(*args,**kwargs):
        out=original_heads(*args,**kwargs)
        assert collector['frame'] is not None
        assert out[2].shape[0]==1 and out[6].numel()==1, 'Unexpected multi-object tracker batch'
        logits=out[3].detach().float();denom=int((logits>-1).sum().item())
        collector['head_calls'].append(dict(frame=collector['frame'],predicted_iou=float(out[2].detach().float().max().item()),
            presence=float(out[6].detach().float().sigmoid().item()),stability=float((logits>1).sum().item()/denom) if denom else 0.,
            raw_object_logit=float(out[6].detach().float().item()),low_res_positive_pixels=int((logits>0).sum().item())))
        return out
    model.tracker.track_step=step;model.tracker._forward_sam_heads=heads
    for i,row in enumerate(pending,1):
        collector['head_calls']=[];collector['step_calls']=[];started=time.time()
        paths=[row['anchor_image_path'],*row['bridge_image_paths'],row['target_image_path']]
        trace=ev.propagate_with_trace(model,paths,row['anchor_box_xywh_normalized'],256)
        old=np.asarray(Image.open(resolve(row['forward_mask_path'])).convert('L'))>127
        differing=int(np.count_nonzero(old!=trace['final_mask']))
        if differing:
            save(R/'REPLAY_MISMATCH.json',dict(route_id=row['route_id'],differing_pixels=differing));raise RuntimeError('Read-only evidence replay changed frozen mask')
        target_frame=len(paths)-1;hits=[h for h in collector['head_calls'] if h['frame']==target_frame]
        nonempty=bool(old.any())
        assert len(hits)<=1, 'Ambiguous target mask head; inspect instead of guessing'
        if nonempty:assert len(hits)==1, 'Missing target quality for nonempty prediction'
        q={k:float(hits[0][k]) if nonempty and hits else 0. for k in ['predicted_iou','presence','stability']}
        result=dict(route_id=row['route_id'],target_id=row['target_id'],anchor_id=row['anchor_id'],bridge_count=row['bridge_count'],
            quality=q,target_head_observed=bool(hits),target_mask_nonempty=nonempty,head_calls=collector['head_calls'],step_frames=collector['step_calls'],
            frozen_mask_sha256=row['forward_mask_sha256'],differing_pixels=0,seconds=time.time()-started,
            extraction_target_gt_used=False)
        assert all(np.isfinite(v) for v in q.values())
        append(R/'evidence.jsonl',result);done[row['route_id']]=result
        if i==1 or i%10==0:
            status('extracting',complete=len(done),total=len(rows));print('EVIDENCE',len(done),'/',len(rows),'quality',q,flush=True)
    if len(done)==len(rows):
        save(R/'EXTRACTION_COMPLETE.json',dict(unique_routes=len(rows),all_masks_pixel_identical=True,max_cuda_allocated=torch.cuda.max_memory_allocated()))
    else:save(R/'SMOKE_COMPLETE.json',dict(complete=len(done),all_masks_pixel_identical=True))

def load_gate():
    import numpy as np
    sys.path.insert(0,str(R/'code'))
    import base_gate as g
    import run_candidate_rerank as rerank
    g.ROUTER=module('b_quality_router',R/'code/router.py')
    out={}
    for mode,path in zip(g.MODES,[TP,AUX]):
        grouped=g.ROUTER.grouped([{**r,'feature_mode':mode} for r in read(path)])
        for target,group in grouped.items():
            group.sort(key=lambda r:r['bridge_count']);assert [r['bridge_count'] for r in group]==list(range(7))
            masks=[g.mask(r['forward_mask_path']) for r in group]
            digests=[hashlib.sha256(m.tobytes()).hexdigest() for m in masks];unique=list(dict.fromkeys(digests))
            for i,r in enumerate(group):
                others=[digests.index(h) for h in unique if h!=digests[i]]
                r['_consensus']=float(np.mean([g.dice(masks[i],masks[j]) for j in others])) if others else 1.
                r['_pred_area']=int(masks[i].sum());r['_mask']=masks[i]
            out.setdefault(target,{})[mode]=group
    assert len(out)==100;g.DATA={'validation':out}
    return g,rerank

def analyze(controls_only=False):
    import numpy as np
    if not controls_only:assert (R/'EXTRACTION_COMPLETE.json').exists()
    g,rerank=load_gate();old_pair=g.pair_features;old_features=list(g.FEATURES)
    evidence={r['route_id']:r for r in read(R/'evidence.jsonl')}
    baseline={r['target_id']:r for r in read(FACTOR/'B/selection/validation_oof.jsonl')}
    def target_pair(t,p,tm,pm):
        f=old_pair(t,p,tm,pm)[:13]
        for key in ['predicted_iou','presence','stability']:
            a=evidence[t['route_id']]['quality'][key];b=evidence[p['route_id']]['quality'][key]
            f.extend([a,b-a])
        return f
    # The historical pair builder asserts a length against g.FEATURES; leave its
    # legacy schema intact and record actual new schemas separately in config.
    variants=['legacy','remove_object_scores'] if controls_only else ['legacy','remove_object_scores','target_quality']
    results={};all_predictions={}
    for variant in variants:
        g.CACHE={};g.AUDIT=[];g.R=R/variant;g.R.mkdir(exist_ok=True)
        if variant=='legacy':g.pair_features=old_pair
        elif variant=='remove_object_scores':g.pair_features=lambda t,p,tm,pm:old_pair(t,p,tm,pm)[:13]
        else:g.pair_features=target_pair
        ids=sorted(g.DATA['validation']);rows=[]
        for fold,held in enumerate(g.split_folds(ids,5,2026)):
            train=sorted(set(ids)-set(held));policy=g.train_policy(train,f'outer_{fold}')
            pred=rerank.decisions(policy['model'],policy['threshold'],rerank.examples(train,held))
            for r in pred:
                r['outer_fold']=fold;r['legacy_dice']=baseline[r['target_id']]['dice']
                assert r['tp_route_id']==baseline[r['target_id']]['tp_route_id']
                assert abs(r['candidate_oracle']-baseline[r['target_id']]['candidate_oracle'])<1e-12
                if variant=='legacy':assert r['selected_route_id']==baseline[r['target_id']]['selected_route_id']
            rows+=pred;print('ANALYZE',variant,fold,np.mean([x['dice'] for x in pred]),flush=True)
        rows.sort(key=lambda r:r['target_id']);s=rerank.stats(rows)
        delta=np.array([r['dice']-r['legacy_dice'] for r in rows]);rng=np.random.default_rng(2026)
        ci=np.quantile(delta[rng.integers(0,100,(10000,100))].mean(1),[.025,.975])
        s['vs_legacy']=dict(delta=float(delta.mean()),ci95=ci.tolist(),wins=int((delta>1e-12).sum()),losses=int((delta< -1e-12).sum()),
            severe_losses_over_005=int((delta<-.05).sum()),fold_delta=[float(np.mean([r['dice']-r['legacy_dice'] for r in rows if r['outer_fold']==f])) for f in range(5)])
        s['fold_delta_vs_tp']=[float(np.mean([r['delta'] for r in rows if r['outer_fold']==f])) for f in range(5)]
        assert abs(s['tp_dice']-.8517552851672318)<1e-12
        if variant=='legacy':assert abs(s['rerank_dice']-.857936922072554)<1e-12
        assert all(not set(a['expert_train_ids'])&set(a['held_ids']) for a in g.AUDIT)
        save(g.R/'fold_provenance.json',g.AUDIT);jsonl(g.R/'validation_oof.jsonl',rows);save(g.R/'results.json',s)
        results[variant]=s;all_predictions[variant]=rows
    if controls_only:
        save(R/'controls_results.json',results);return
    primary=results['target_quality']
    eligible=bool(primary['mean_delta']>=.003 and primary['paired_bootstrap_95ci'][0]>0 and primary['vs_legacy']['delta']>=.003 and primary['vs_legacy']['ci95'][0]>0 and sum(x>0 for x in primary['vs_legacy']['fold_delta'])>=3)
    result=dict(split='validation',variants=results,primary='target_quality',passes_predeclared_future_test_gate=eligible,test=None,
        audit=dict(unique_evidence_routes=len(evidence),all_frozen_masks_pixel_identical=True,frozen_TP_and_candidate_oracles_identical=True,legacy_100_choices_reproduced=True))
    save(R/'results.json',result);save(R/'TEST_NOT_RUN.json',dict(reason='This run is validation-only; no test variant evaluated.',primary_passes_future_gate=eligible))
    lines=['# Kvasir：冻结 B 候选池的目标端质量证据实验','','原始 validation 100 张，原 8/792 训练划分不变。无学生，无新 test。TP 七候选与 B 辅助七候选保持，默认 TP 独立 Router 的折外输出逐图不变。','','| 方案 | 实际 Dice | 相对 TP | 相对原 B 选择器 |','|---|---:|---:|---:|']
    for v,s in results.items():lines.append(f"| {v} | {s['rerank_dice']:.6f} | {s['mean_delta']:+.6f} | {s['vs_legacy']['delta']:+.6f} |")
    lines+=['','主实验为 target_quality：保留前 13 个旧特征，用目标帧预测 IoU、对象存在概率和 logit 稳定性替代旧对象分数，每项使用 TP 值与辅助相对差。remove_object_scores 仅删旧分数，用于区分去噪与新证据的作用。','',
        '新评分通过只读包装 tracker 方法记录，返回的模型张量完全不变。每条传播都与冻结 mask 逐像素核对；提取过程没有读取目标 GT。预测 IoU、对象存在分数及稳定性都不是实际 Dice。','',
        '选择器与之前相同：按目标加权的 Ridge 增益回归，正则 10/100，阈值 0.02/0.05/0.10，包含 TP 回退；5 外层、4 内层、3 层专家交叉拟合。主实验预先固定，不按外层结果挑控制组作为赢家。','',
        f"主实验相对原 B 的增益区间：{primary['vs_legacy']['ci95']}；各折增益：{primary['vs_legacy']['fold_delta']}。",'',
        f"是否通过预先设定的后续 test 门槛：{eligible}。本轮均不自动评估 test。",'',
        '冻结 TP + 辅助七候选的 Oracle 保持 0.893326；完整 TP 七候选 + 辅助七候选 Oracle 为 0.898534。主选择器只访问前一个集合。']
    (R/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(R/'report.md',P/'reproduction_reports/Kvasir_B_target_quality_validation_20260908.md')
    status('complete',results=str(R/'results.json'));(R/'COMPLETE').touch();print(json.dumps(result,indent=2),flush=True)

def pipeline():
    try:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (R/'extraction.log').open('a') as log:
            subprocess.run(['/home/violet/anaconda3/envs/sam3/bin/python','-u',__file__,'extract'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('analyzing')
        with (R/'analysis.log').open('a') as log:
            subprocess.run(['/home/violet/anaconda3/envs/mkunet_mamba/bin/python','-u',__file__,'analyze'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    except BaseException:
        error=traceback.format_exc();(R/'FAILED.txt').write_text(error);status('failed',error=error);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['setup','extract','controls','analyze','pipeline']);parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    if args.action=='setup':setup()
    elif args.action=='extract':extract(args.limit)
    elif args.action=='controls':analyze(True)
    elif args.action=='analyze':analyze()
    else:pipeline()
