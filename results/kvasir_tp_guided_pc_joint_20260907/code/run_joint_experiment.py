"""Unified TP/PC route construction and no-student test evaluation."""
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
import traceback

import numpy as np
from PIL import Image
from beam_backend import fast_routes
from joint_scores import fit_anchor_calibration,fit_transition_calibration,standardize,guided_reference,self_test

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'work/kvasir_tp_guided_pc_joint_20260907'
OLD=P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6'
TEST=P/'work/rerun_kvasir_sam3base_test_20260906/quality_root'
PROTO=P/'work/kvasir_1pct_anchors/protocol'
CACHE=P/'work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features/sam3_base_s256_features.npz'
CKPT=Path('/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt')
TP='sam3enc_anchor_conditioned_target_pooling'
PC='sam3enc_anchor_conditioned_patch_correspondence'
VARIANTS=['tp_baseline','joint_v1','joint_v2']

def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def jsonl(path,rr):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rr))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def data():return read(R/'protocol/merged_manifest.jsonl'),read(R/'protocol/support_manifest.jsonl')
def quality_path(v,split):return R/v/v/f'propagation_quality_{split}/propagation_quality.jsonl'

def setup():
    self_test();assert not R.exists(),R
    assert sha(PROTO/'merged_manifest.jsonl')=='eb74d0ea534450d7b97ae93319260b7cbbd78e5b74b2b615c777b1263eb91322'
    assert sha(CACHE)=='ab8ca194f9c789ad9f6b641d382ed848751f29b23ac0bdb29a9c6722fd599907'
    R.mkdir();(R/'code').mkdir();(R/'logs').mkdir();shutil.copytree(PROTO,R/'protocol')
    for f in ['run_joint_experiment.py','joint_scores.py','beam_backend.py']:shutil.copy2(Path(__file__).parent/f,R/'code'/f)
    for f in ['stage1_feature_knn_routes.py','eval_route_propagation_quality.py','analyze_propagation_quality_router.py']:shutil.copy2(P/'scripts'/f,R/'code'/f)
    records,support=data()
    assert collections.Counter(r['split'] for r in records)=={'train':800,'validation':100,'test':100}
    assert len({r['merged_id'] for r in records})==1000 and len(support)==8
    assert {r['image_path'] for r in support}==set((PROTO/'frozen_labeled_images.txt').read_text().splitlines())
    cfg={'created_at':datetime.datetime.now().astimezone().isoformat(),'checkpoint':str(CKPT),'checkpoint_sha256':sha(CKPT),'feature_cache':str(CACHE),'feature_cache_sha256':sha(CACHE),'split_counts':{'train':800,'validation':100,'test':100},'labeled_train':8,'unlabeled_train':792,'feature_size':256,'propagation_canvas':256,'beam_width':32,'bridge_range':[0,6],'candidates_per_target':7,'student':False,'train_propagation':False,
     'fixed_parameters':{'tau':10.0,'local_top_r':3,'local_r_rule':'min(3, anchor foreground patch count)','joint_v1_alpha':.75,'joint_v2_alpha':.70,'mad_epsilon':1e-6},
     'definitions':{'tp':'original L2-normalized attention-pooled descriptor cosine with anchor prototype','pc':'original prototype-to-target Top8 mean','local':'TP-attention weighted mean of per-target-patch top-r foreground-anchor-token similarities','joint_v1':'.75*z(TP)+.25*z(original_PC)','joint_v2':'.70*z(TP)+.30*z(TP_guided_local_PC)','calibration':'median/MAD separately per anchor/component on original train RGB, excluding the anchor itself; freeze before route evaluation','transition_calibration':'global median/MAD of off-diagonal unordered train/train patch_mean cosine pairs; dimensionless transitions for both Joint variants','knn_proposal':'original patch_mean nearest-neighbor proposal ordering, with new conditional score as tie-break; joint score enters anchor and full beam path objective','path_objective':'lexicographic max(bottleneck, mean) over conditional node scores and consecutive transition scores','router':'historical ridge=1.0, fit independently per family on validation; no mode bits, no student','validation':'5-fold target-level OOF, seed 2026, diagnostic; both Joint variants proceed to test by user request','test':'100 fixed test targets, fit final Router on all original 100 validation targets before test inference, no confidence filtering or test tuning'},
     'code_sha256':{f.name:sha(f) for f in (R/'code').glob('*.py')}}
    save(R/'config.json',cfg);print(json.dumps(cfg,indent=2),flush=True)

def extract():
    import torch
    from sam3.model_builder import build_sam3_video_model
    stage=module('joint_stage',R/'code/stage1_feature_knn_routes.py')
    records,support=data();base=np.load(CACHE)
    assert base['anchor_ids'].tolist()==[r['merged_id'] for r in support]
    model=build_sam3_video_model(checkpoint_path=str(CKPT),load_from_HF=False,device='cuda',compile=False).eval()
    trunk=model.detector.backbone.vision_backbone.trunk;stage.prepare_sam3_trunk(trunk,256)
    def tokens(x):return torch.nn.functional.normalize(trunk(x)[0].flatten(2).permute(0,2,1),dim=-1)
    anchor_tokens=[];proto=torch.from_numpy(base['anchor_prototypes']).cuda();proto_error=[]
    local_all=np.zeros((8,len(records)),np.float32);top8_error=[];tp_error=[];probe_error=[]
    with torch.no_grad():
        for ai,anchor in enumerate(support):
            t=tokens(((stage.load_rgb_tensor(anchor['image_path'],256)-.5)/.5)[None].cuda())[0]
            fg=stage.load_mask_grid(anchor['frozen_mask_path'],18).reshape(-1)
            assert fg.any(),'No foreground tokens in anchor'
            selected=t[fg];anchor_tokens.append(selected)
            normalized=torch.nn.functional.normalize(selected.mean(0),dim=0)
            proto_error.append(float((normalized-proto[ai]).abs().max()))
        assert max(proto_error)<=1e-5,proto_error
        for start in range(0,len(records),8):
            chunk=records[start:start+8]
            x=torch.stack([(stage.load_rgb_tensor(r['image_path'],256)-.5)/.5 for r in chunk]).cuda()
            t=tokens(x)
            sims=torch.einsum('ad,bpd->abp',proto,t)
            weights=torch.exp(10*(sims-sims.amax(2,keepdim=True)))
            weights=weights/weights.sum(2,keepdim=True).clamp_min(1e-12)
            pooled=torch.nn.functional.normalize(torch.einsum('abp,bpd->abd',weights,t),dim=-1)
            tp=torch.einsum('abd,ad->ab',pooled,proto).float().cpu().numpy()
            pc=torch.topk(sims,k=8,dim=2).values.mean(2).float().cpu().numpy()
            tp_error.append(float(np.max(np.abs(tp-base['cond_target'][:,start:start+len(chunk)]))))
            top8_error.append(float(np.max(np.abs(pc-base['cond_correspondence'][:,start:start+len(chunk)]))))
            # SAM3 globally enables TF32. Compute the new correspondence head
            # in float64 so it agrees with the independent mathematical reference;
            # leave historical trunk/TP/Top8 arithmetic unchanged for exact replay.
            precise_tokens=t.double()
            local_weights=torch.softmax(10*torch.einsum('ad,bpd->abp',proto.double(),precise_tokens),dim=2)
            for ai,a in enumerate(anchor_tokens):
                local=torch.einsum('md,bpd->bmp',a.double(),precise_tokens).topk(min(3,len(a)),dim=1).values.mean(1)
                scalar=(local_weights[ai]*local).sum(1).float().cpu().numpy()
                local_all[ai,start:start+len(chunk)]=scalar
                if start==0:
                    reference,_,_=guided_reference(t[0].float().cpu().numpy(),a.float().cpu().numpy(),proto[ai].float().cpu().numpy())
                    probe_error.append(abs(reference-float(scalar[0])))
            if start==0 or (start+8)%80==0:print('features',min(start+8,len(records)),flush=True)
    audit={'image_count':len(records),'tp_max_error':max(tp_error),'pc_top8_max_error':max(top8_error),'anchor_prototype_max_error':max(proto_error),'local_reference_max_error':max(probe_error),'anchor_foreground_token_counts':[len(a) for a in anchor_tokens]}
    save(R/'feature_replay_audit.json',audit)
    assert max(tp_error+top8_error+proto_error+probe_error)<=1e-5,audit
    np.savez_compressed(R/'anchor_tokens.npz',**{f'anchor_{i}':a.float().cpu().numpy() for i,a in enumerate(anchor_tokens)})
    np.save(R/'guided_local_scores.npy',local_all)
    print(json.dumps(audit),flush=True)

def build():
    self_test();stage=module('joint_stage',R/'code/stage1_feature_knn_routes.py')
    records,support=data();base=np.load(CACHE);local=np.load(R/'guided_local_scores.npy')
    index={r['merged_id']:i for i,r in enumerate(records)}
    train=[i for i,r in enumerate(records) if r['split']=='train'];anchors=[index[r['merged_id']] for r in support]
    raw={'region':base['cond_target'],'pc':base['cond_correspondence'],'local':local}
    calibration={k:fit_anchor_calibration(s,train,anchors) for k,s in raw.items()}
    sim=base['patch_mean']@base['patch_mean'].T
    calibration['transitions']=fit_transition_calibration(sim,train)
    # Freeze train-only medians/MADs before any route quality is evaluated.
    save(R/'calibration.json',calibration)
    z={k:standardize(s,calibration[k]) for k,s in raw.items()}
    c=calibration['transitions'];edge=(sim.astype(np.float64)-c['median'])/(c['mad']+c['epsilon'])
    scores={'tp_baseline':base['cond_target'],'joint_v1':.75*z['region']+.25*z['pc'],'joint_v2':.70*z['region']+.30*z['local']}
    np.savez_compressed(R/'joint_score_tables.npz',**scores)
    tp_reference={s:{(r['target_id'],r['bridge_count']):r for r in read(OLD/TP/f'{s}_pool0_stage1/routes.jsonl')} for s in ['validation','test']}
    for variant in VARIANTS:
        for split in ['validation','test']:
            rr=fast_routes(stage,records,support,base['patch_mean'],scores[variant],split,transition_sim=None if variant=='tp_baseline' else edge)
            assert len(rr)==700 and len({(r['target_id'],r['bridge_count']) for r in rr})==700
            expected={r['merged_id'] for r in records if r['split']==split}
            assert {r['target_id'] for r in rr}==expected
            for row in rr:
                assert set(row['bridge_ids'])<={records[i]['merged_id'] for i in train}
                assert row['anchor_id'] in {x['merged_id'] for x in support}
                row['route_family']=variant
            old=tp_reference[split]
            changes=sum(r['route_id']!=old[r['target_id'],r['bridge_count']]['route_id'] for r in rr)
            score_err=max(abs(r[k]-old[r['target_id'],r['bridge_count']][k]) for r in rr for k in ['path_mean_similarity','path_bottleneck_similarity']) if variant=='tp_baseline' else None
            if variant=='tp_baseline':assert changes==0 and score_err<1e-6,(changes,score_err)
            audit={'count':700,'changed_routes_vs_tp':changes,'changed_anchors_vs_tp':sum(r['anchor_id']!=old[r['target_id'],r['bridge_count']]['anchor_id'] for r in rr),'anchor_counts':dict(collections.Counter(r['anchor_id'] for r in rr)),'baseline_replay_max_path_score_error':score_err}
            save(R/variant/f'route_audit_{split}.json',audit)
            jsonl(R/variant/variant/f'{split}_pool0_stage1/routes.jsonl',rr)
            print('routes complete',variant,split,flush=True)

def seed(variant,split):
    path=quality_path(variant,split)
    if path.exists():return
    sources={}
    for mode in [TP,PC]:
        root=OLD if split=='validation' else TEST
        for row in read(root/mode/f'propagation_quality_{split}/propagation_quality.jsonl'):
            if row.get('status')=='success':sources[row['route_id']]=row
    for previous in ['v1','v2']:
        path_old=P/'work/kvasir_pc_adaptive_spatial_20260906'/previous/PC/f'propagation_quality_{split}/propagation_quality.jsonl'
        if path_old.exists():
            for row in read(path_old):
                if row.get('status')=='success':sources[row['route_id']]=row
    rr=read(R/variant/variant/f'{split}_pool0_stage1/routes.jsonl');seeded=[]
    for row in rr:
        old=sources.get(row['route_id'])
        if old is None:continue
        for key in ['anchor_id','anchor_mask_sha256','anchor_box_xywh_normalized','bridge_ids','target_id','anchor_image_path','bridge_image_paths','target_image_path']:assert row[key]==old[key]
        assert Path(old['forward_mask_path']).is_file()
        seeded.append({**old,**row})
    jsonl(path,seeded)
    save(R/variant/f'reuse_audit_{split}.json',{'identical_routes_reused':len(seeded),'new_inference':700-len(seeded),'checkpoint':str(CKPT),'new_path_scores_preserved':True})

def infer(variant,split):
    seed(variant,split)
    if len(read(quality_path(variant,split)))<700:
        cmd=[sys.executable,'-u',str(R/'code/eval_route_propagation_quality.py'),'--checkpoint',str(CKPT),'--mode',variant,'--root',str(R/variant),'--split',split,'--canvas','256','--resume']
        print('INFERENCE',variant,split,flush=True)
        with (R/'logs'/f'{variant}_{split}_propagation.log').open('a') as log:subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True)
    rr=read(quality_path(variant,split))
    assert len(rr)==700 and len({r['route_id'] for r in rr})==700
    assert all(r.get('status')=='success' and Path(r['forward_mask_path']).is_file() for r in rr)

def quality(variant,split):return [{**r,'feature_mode':'anchor_conditioned_target_pooling'} for r in read(quality_path(variant,split))]
def fit(router,rr):
    raw=np.asarray([router.feature_vector(r,False) for r in rr],np.float64)
    means=raw.mean(0);stds=np.maximum(raw.std(0),1e-8)
    x=np.column_stack([np.ones(len(rr)),(raw-means)/stds]);reg=np.eye(x.shape[1]);reg[0,0]=0
    weights=np.linalg.solve(x.T@x+reg,x.T@np.asarray([r['gt_dice_evaluation_only'] for r in rr]))
    return router.Ridge(means.tolist(),stds.tolist(),weights.tolist(),False)
def select(router,scorer,rr):
    out=[]
    for target,group in sorted(router.grouped(rr).items()):
        clean=[{k:v for k,v in r.items() if not k.startswith('gt_') and 'evaluation_only' not in k} for r in group]
        c=max(clean,key=lambda r:(scorer.score(r),-int(r['bridge_count']),r['route_id']))
        out.append({'target_id':target,'route_id':c['route_id'],'bridge_count':c['bridge_count'],'router_score':scorer.score(c),'source_mask_path':c['forward_mask_path'],'anchor_id':c['anchor_id']})
    return out
def evaluate(choices,rr):
    index={r['route_id']:r for r in rr};by_target=collections.defaultdict(list)
    for row in rr:by_target[row['target_id']].append(row)
    for c in choices:
        row=index[c['route_id']]
        c['dice']=float(row['gt_dice_evaluation_only']);c['oracle']=max(float(r['gt_dice_evaluation_only']) for r in by_target[c['target_id']])
    return choices
def summary(choices):return {'count':len(choices),'selected_dice':float(np.mean([r['dice'] for r in choices])),'oracle_dice':float(np.mean([r['oracle'] for r in choices]))}

def prepare_router(variant):
    router=module('joint_router',R/'code/analyze_propagation_quality_router.py');rr=quality(variant,'validation')
    ids=sorted({r['target_id'] for r in rr});order=np.random.default_rng(2026).permutation(ids);folds={str(t):i%5 for i,t in enumerate(order)}
    save(R/variant/'validation_folds.json',folds);oof=[]
    for fold in range(5):
        train=[r for r in rr if folds[r['target_id']]!=fold];held=[r for r in rr if folds[r['target_id']]==fold]
        oof.extend(evaluate(select(router,fit(router,train),held),held))
    oof.sort(key=lambda r:r['target_id']);jsonl(R/variant/'validation_oof.jsonl',oof)
    s=summary(oof);s['bridges']={str(b):float(np.mean([r['gt_dice_evaluation_only'] for r in rr if r['bridge_count']==b])) for b in range(7)}
    save(R/variant/'validation_summary.json',s)
    scorer=fit(router,rr)
    if variant=='tp_baseline':
        original=router.Ridge.fit(rr,ridge=1.0,include_mode=False)
        error=max(abs(scorer.score(r)-original.score(r)) for r in rr)
        assert error<1e-7,error
        # Compare actual choices separately because floating score values can differ.
        assert [r['route_id'] for r in select(router,scorer,rr)]==[r['route_id'] for r in select(router,original,rr)]
        save(R/'ridge_implementation_audit.json',{'max_score_error':error,'route_changes':0})
    save(R/variant/'final_router.json',vars(scorer))
    (R/variant/'ROUTER_FROZEN').touch()

def evaluate_test(variant):
    assert (R/variant/'ROUTER_FROZEN').exists()
    router=module('joint_router',R/'code/analyze_propagation_quality_router.py')
    scorer=router.Ridge(**json.loads((R/variant/'final_router.json').read_text()))
    rr=quality(variant,'test');choices=select(router,scorer,rr)
    dest=R/variant/'final_test_masks';dest.mkdir(exist_ok=True)
    for c in choices:
        p=dest/(c['target_id'].replace('::','__')+'.png');shutil.copy2(c['source_mask_path'],p);c['final_mask_path']=str(p);c['mask_sha256']=sha(p)
    jsonl(R/variant/'selected_test_masks.jsonl',choices) # Commit choices before GT evaluation.
    evaluated=evaluate(choices,rr);index={r['route_id']:r for r in rr}
    for c in evaluated:
        pred=np.asarray(Image.open(c['final_mask_path']).convert('L'))>127
        gt=np.asarray(Image.open(index[c['route_id']]['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
        inter=int(np.logical_and(pred,gt).sum());a=int(pred.sum());b=int(gt.sum())
        assert pred.shape==(256,256) and abs(c['dice']-2*inter/max(a+b,1))<1e-12
        c['iou']=inter/max(a+b-inter,1)
    jsonl(R/variant/'test_per_target_metrics.jsonl',evaluated)
    s=summary(evaluated);s.update({'iou':float(np.mean([r['iou'] for r in evaluated])),'bridge_counts':dict(collections.Counter(r['bridge_count'] for r in evaluated)),'anchor_counts':dict(collections.Counter(r['anchor_id'] for r in evaluated)),'bridges':{str(b):float(np.mean([r['gt_dice_evaluation_only'] for r in rr if r['bridge_count']==b])) for b in range(7)}})
    assert len(evaluated)==100
    if variant=='tp_baseline':assert abs(s['selected_dice']-.8854326463411542)<1e-12
    save(R/variant/'test_summary.json',s)

def worker(variant):
    try:
        infer(variant,'validation');prepare_router(variant)
        infer(variant,'test');evaluate_test(variant)
        (R/variant/'COMPLETE').write_text(datetime.datetime.now().astimezone().isoformat())
    except BaseException:
        save(R/variant/'FAILED.json',{'traceback':traceback.format_exc()});raise

def final_report():
    result={v:{s:json.loads((R/v/f'{s}_summary.json').read_text()) for s in ['validation','test']} for v in VARIANTS}
    save(R/'results.json',result)
    lines=['# Kvasir：TP-guided Patch Correspondence 统一选路实验','', 'SAM3-base@256；原800/100/100 split和8个supports；无学生。每版本单一route family，b0-b6每图7个候选，测试覆盖全部100张。','', 'Joint-v1=.75 z(TP)+.25 z(原Top8 PC)。Joint-v2=.70 z(TP)+.30 z(TP-attention加权的局部多patch对应)。tau=10，local top-r=min(3,前景anchor patch数)。区域TP使用原实现的归一化pooled descriptor。', '', 'z=(s-median)/(MAD+1e-6)，每anchor/分数组件仅由训练图像拟合，排除anchor自身。两Joint版本的转移余弦也按训练图像对median/MAD标准化，避免混合量纲。原patch-mean KNN提案排序保持，联合分数进入anchor与完整beam路径评分。', '', 'Validation报告5折target-level OOF Dice。最终Router使用全部100张validation拟合并冻结，再评test；两个Joint均按用户要求测试，不按test选择权重。', '', '| 方法 | Val OOF Dice | Test Dice | Test IoU | Test Oracle |','|---|---:|---:|---:|---:|']
    for v,d in result.items():lines.append(f"| {v} | {d['validation']['selected_dice']:.6f} | {d['test']['selected_dice']:.6f} | {d['test']['iou']:.6f} | {d['test']['oracle_dice']:.6f} |")
    lines+=['','## Test 固定桥长','', '| Bridge | TP baseline | Joint-v1 | Joint-v2 |','|---|---:|---:|---:|']
    for b in range(7):lines.append(f'| b{b} | '+' | '.join(f"{result[v]['test']['bridges'][str(b)]:.6f}" for v in VARIANTS)+' |')
    lines+=['', 'Oracle只用GT事后分析，不能作为实际预测性能。最终mask为单一路线候选中Router最高分mask，未合并独立TP/PC候选池。全部保存mask已按原GT重新计算并核对Dice；本轮未做train传播。','', 'Anchor patch数量差异仍可能影响局部相似度分布；训练分布median/MAD校准是受测方案，不预设已解决anchor偏向。这个三组对照没有单独隔离“仅校准TP”的效果，不能把Joint对TP的全部差值都归于局部对应。', '', f'完整产物：`{R}`。各版本final_test_masks、selected_test_masks.jsonl、test_per_target_metrics.jsonl、final_router.json均保留。']
    (R/'report.md').write_text('\n'.join(lines)+'\n')
    target=P/'reproduction_reports/Kvasir_TP_guided_PC_joint_20260907.md'
    with target.open('x') as f:f.write('\n'.join(lines)+'\n')

def run():
    with (R/'STARTED').open('x') as f:f.write(datetime.datetime.now().astimezone().isoformat())
    try:
        for cmd in ['extract','build']:subprocess.run([sys.executable,'-u',__file__,cmd],cwd=P,check=True)
        worker('tp_baseline')
        processes=[]
        for gpu,v in enumerate(['joint_v1','joint_v2']):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            with (R/'logs'/f'{v}_worker.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-u',__file__,'worker',v],cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(R/v/'process.json',{'pid':proc.pid,'gpu':gpu});processes.append((v,proc))
        errors=[]
        for v,proc in processes:
            rc=proc.wait();save(R/v/'exit.json',{'exit_code':rc})
            if rc:errors.append((v,rc))
        assert not errors,errors
        final_report();(R/'COMPLETE').write_text(datetime.datetime.now().astimezone().isoformat())
    except BaseException:
        (R/'FAILED').write_text(traceback.format_exc());raise

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='setup':setup()
    elif cmd=='run':run()
    elif cmd=='extract':extract()
    elif cmd=='build':build()
    elif cmd=='worker':worker(sys.argv[2])
    else:raise ValueError(cmd)
