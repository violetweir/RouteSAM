"""Validation-first, no-student PC ablation. Original experiment files are read-only."""
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

import numpy as np
from PIL import Image
from pc_adaptive_core import adaptive_k, fast_routes, score_maps, self_test

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'work/kvasir_pc_adaptive_spatial_20260906'
OLD=P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6'
PROTO=P/'work/kvasir_1pct_anchors/protocol'
CACHE=P/'work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features/sam3_base_s256_features.npz'
PC='sam3enc_anchor_conditioned_patch_correspondence'
TP='sam3enc_anchor_conditioned_target_pooling'
CKPT=Path('/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt')

def read(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def save(p,d):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')

def jsonl(p,rr):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rr))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def data():
    return read(R/'protocol/merged_manifest.jsonl'),read(R/'protocol/support_manifest.jsonl')

def setup():
    self_test()
    assert not R.exists(),R
    records,support=read(PROTO/'merged_manifest.jsonl'),read(PROTO/'support_manifest.jsonl')
    assert collections.Counter(r['split'] for r in records)=={'train':800,'validation':100,'test':100}
    assert len({r['merged_id'] for r in records})==1000
    assert {r['merged_id'] for r in support}<={r['merged_id'] for r in records if r['split']=='train'}
    assert len(support)==8
    assert sha(PROTO/'merged_manifest.jsonl')=='eb74d0ea534450d7b97ae93319260b7cbbd78e5b74b2b615c777b1263eb91322'
    assert sha(CACHE)=='ab8ca194f9c789ad9f6b641d382ed848751f29b23ac0bdb29a9c6722fd599907'
    R.mkdir();shutil.copytree(PROTO,R/'protocol');(R/'code').mkdir();(R/'logs').mkdir()
    for f in ['pc_adaptive_core.py','run_pc_ablation.py']:
        shutil.copy2(Path(__file__).parent/f,R/'code'/f)
    for f in ['stage1_feature_knn_routes.py','eval_route_propagation_quality.py','analyze_propagation_quality_router.py']:
        shutil.copy2(P/'scripts'/f,R/'code'/f)
    fractions=[float((np.asarray(Image.open(r['frozen_mask_path']).convert('L'))>127).mean()) for r in support]
    config={'created_at':datetime.datetime.now().astimezone().isoformat(),'checkpoint':str(CKPT),'checkpoint_sha256':sha(CKPT),'feature_cache':str(CACHE),'feature_cache_sha256':sha(CACHE),'feature_size':256,'grid':18,'patch_count':324,'beam_width':32,'canvas':256,'bridges':[0,6],'student':False,'new_train_propagation':False,'gpu':1,
      'variants':{'v0':'original cached Top8 mean','v1':'K=clamp(round(anchor original GT fraction*324),1,324); TopK mean','v2':'v1 - 0.05*(1 - K/bounding_rectangle_area(topK))'},
      'anchor_fractions':fractions,'anchor_k':[adaptive_k(f,324) for f in fractions],
      'validation_protocol':{'folds':5,'fold_unit':'target_id','seed':2026,'ridge':1.0,'primary':'PC-only b0-b6 out-of-fold selected Dice','secondary':['mean fixed b3-b6 Dice','candidate oracle','TP+PC OOF selected Dice'],
      'test_gate':'Best new variant must improve PC-only OOF Dice by >=0.003 and mean fixed b3-b6 Dice must be >= v0 minus 0.005. Choose higher PC OOF; exact tie prefers v1. Otherwise stop at validation.',
      'test_if_passed':'Freeze winner; run only winner test routes; fit final Ridge on all original 100 validation targets per setting. Report v0/winner PC-only and TP+PC at b0-b6. No test-based tuning.'},
      'code_sha256':{f.name:sha(f) for f in (R/'code').glob('*.py')}}
    save(R/'config.json',config)
    print(json.dumps(config,indent=2),flush=True)

def extract(split):
    import torch
    from sam3.model_builder import build_sam3_video_model
    records,support=data();base=np.load(CACHE)
    assert base['anchor_ids'].tolist()==[r['merged_id'] for r in support]
    stage=module('pc_stage',R/'code/stage1_feature_knn_routes.py')
    model=build_sam3_video_model(checkpoint_path=str(CKPT),load_from_HF=False,device='cuda',compile=False)
    model.eval();trunk=model.detector.backbone.vision_backbone.trunk
    stage.prepare_sam3_trunk(trunk,256)
    proto=torch.from_numpy(base['anchor_prototypes']).cuda()
    maps=np.full((8,len(records),324),np.nan,dtype=np.float32)
    wanted={'train','validation'} if split=='validation' else {'test'}
    errors=[];done=0
    with torch.no_grad():
        for start in range(0,len(records),8):
            chunk=records[start:start+8]
            if not any(r['split'] in wanted for r in chunk):continue
            # Retain original batch slots while never opening test RGB before its gate.
            batch=torch.stack([(stage.load_rgb_tensor(r['image_path'],256)-.5)/.5 if r['split'] in wanted else torch.zeros(3,256,256) for r in chunk]).cuda()
            features=trunk(batch)[0]
            tokens=torch.nn.functional.normalize(features.flatten(2).permute(0,2,1),dim=-1)
            sims=torch.einsum('ad,bpd->abp',proto,tokens)
            top8=torch.topk(sims,k=8,dim=2).values.mean(dim=2).float().cpu().numpy()
            sim_np=sims.float().cpu().numpy()
            for j,rr in enumerate(chunk):
                if rr['split'] not in wanted:continue
                maps[:,start+j]=sim_np[:,j]
                errors.append(float(np.max(np.abs(top8[:,j]-base['cond_correspondence'][:,start+j]))))
                done+=1
            if done%80<8:print('extracted',done,flush=True)
    audit={'split':split,'count':done,'top8_replay_max_abs_error':max(errors),'top8_replay_mean_abs_max_error':float(np.mean(errors))}
    save(R/f'feature_replay_{split}.json',audit)
    assert max(errors)<=1e-5, audit
    np.save(R/f'patch_similarity_maps_{split}.npy',maps)
    print(json.dumps(audit),flush=True)

def generate(split):
    self_test();records,support=data();base=np.load(CACHE)
    stage=module('pc_stage',R/'code/stage1_feature_knn_routes.py')
    if split=='validation':
        original=fast_routes(stage,records,support,base['patch_mean'],base['cond_correspondence'],'validation')
        old=read(OLD/PC/'validation_pool0_stage1/routes.jsonl')
        by={(r['target_id'],r['bridge_count']):r for r in old}
        mismatches=[r['route_id'] for r in original if r['route_id']!=by[r['target_id'],r['bridge_count']]['route_id']]
        max_score_error=max(abs(r[k]-by[r['target_id'],r['bridge_count']][k]) for r in original for k in ['path_mean_similarity','path_bottleneck_similarity'])
        save(R/'v0_route_identity_audit.json',{'count':len(original),'changed_route_count':len(mismatches),'max_path_score_error':max_score_error})
        # Historical float32 BLAS path similarities can differ by a few ULPs.
        # Require exact route identity and bound score drift to 1e-6.
        assert len(original)==700 and not mismatches and max_score_error<1e-6
        variants=['v1','v2']
    else:
        gate=json.loads((R/'validation_decision.json').read_text());assert gate['test_authorized_by_gate']
        variants=[gate['winner']]
    maps=np.load(R/'patch_similarity_maps_validation.npy')
    if split=='test':
        testmaps=np.load(R/'patch_similarity_maps_test.npy')
        take=[i for i,r in enumerate(records) if r['split']=='test'];maps[:,take]=testmaps[:,take]
    config=json.loads((R/'config.json').read_text())
    wanted=[i for i,r in enumerate(records) if r['split'] in ({'train','validation'} if split=='validation' else {'train','test'})]
    computed=score_maps(maps[:,wanted],config['anchor_fractions'])
    old=read(OLD/PC/f'{split}_pool0_stage1/routes.jsonl');by={(r['target_id'],r['bridge_count']):r for r in old}
    for variant in variants:
        scores=base['cond_correspondence'].copy();scores[:,wanted]=computed[variant]
        rr=fast_routes(stage,records,support,base['patch_mean'],scores,split)
        for row in rr:
            assert row['target_split']==split
            assert set(row['bridge_ids'])<={x['merged_id'] for x in records if x['split']=='train'}
        jsonl(R/variant/PC/f'{split}_pool0_stage1/routes.jsonl',rr)
        save(R/variant/f'route_audit_{split}.json',{'routes':len(rr),'changed_routes':sum(r['route_id']!=by[r['target_id'],r['bridge_count']]['route_id'] for r in rr),'changed_anchors':sum(r['anchor_id']!=by[r['target_id'],r['bridge_count']]['anchor_id'] for r in rr),'anchor_counts':dict(collections.Counter(r['anchor_id'] for r in rr))})
        np.savez_compressed(R/variant/f'conditioned_scores_{split}.npz',scores=scores,anchor_k=computed['anchor_k'],compactness=computed['compactness'],record_indices=wanted)
        print('generated',variant,split,flush=True)

def infer(variant,split):
    root=R/variant;routes=read(root/PC/f'{split}_pool0_stage1/routes.jsonl')
    quality=root/PC/f'propagation_quality_{split}';quality.mkdir(parents=True,exist_ok=True)
    path=quality/'propagation_quality.jsonl'
    if not path.exists():
        sources={}
        for mode in [PC,TP]:
            for row in read(OLD/mode/f'propagation_quality_{split}/propagation_quality.jsonl'):
                if row.get('status')=='success':sources[row['route_id']]=row
        if variant=='v2' and split=='validation':
            for row in read(R/'v1'/PC/'propagation_quality_validation/propagation_quality.jsonl'):
                if row.get('status')=='success':sources[row['route_id']]=row
        seeded=[]
        for route in routes:
            src=sources.get(route['route_id'])
            if src is None:continue
            for key in ['anchor_id','anchor_mask_sha256','anchor_box_xywh_normalized','bridge_ids','target_id']:
                assert route[key]==src[key]
            seeded.append({**src,**route}) # Preserve new route scores even when reusing inference.
        jsonl(path,seeded)
        save(root/f'reuse_audit_{split}.json',{'reused_identical_routes':len(seeded),'new_routes':len(routes)-len(seeded),'teacher':str(CKPT)})
    cmd=[sys.executable,'-u',str(R/'code/eval_route_propagation_quality.py'),'--checkpoint',str(CKPT),'--mode',PC,'--root',str(root),'--split',split,'--canvas','256','--resume']
    print('INFERENCE',variant,split,flush=True)
    with (R/'logs'/f'{variant}_{split}_propagation.log').open('a') as log:
        subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True)
    done=read(path);assert len(done)==700 and len({r['route_id'] for r in done})==700
    assert all(r.get('status')=='success' and Path(r['forward_mask_path']).is_file() for r in done)

def quality(variant,split):
    root=OLD if variant=='v0' else R/variant
    rr=read(root/PC/f'propagation_quality_{split}/propagation_quality.jsonl')
    return [{**r,'feature_mode':'anchor_conditioned_patch_correspondence'} for r in rr]

def fit_fast(router,rows,include_mode):
    # Same standardized ridge objective, solved with NumPy for CV efficiency.
    raw=np.asarray([router.feature_vector(r,include_mode) for r in rows],dtype=np.float64)
    means=raw.mean(0);stds=np.maximum(raw.std(0),1e-8)
    x=np.column_stack([np.ones(len(rows)),(raw-means)/stds]);y=np.asarray([r['gt_dice_evaluation_only'] for r in rows])
    reg=np.eye(x.shape[1]);reg[0,0]=0
    w=np.linalg.solve(x.T@x+reg,x.T@y)
    return router.Ridge(means.tolist(),stds.tolist(),w.tolist(),include_mode)

def select(router,scorer,rows):
    selected=[]
    for target,rr in sorted(router.grouped(rows).items()):
        # GT values cannot enter scorer inputs or the selection key.
        clean=[{k:v for k,v in r.items() if not k.startswith('gt_') and 'evaluation_only' not in k} for r in rr]
        chosen=max(clean,key=lambda r:(scorer.score(r),-int(r['bridge_count']),r['route_id']))
        original=next(r for r in rr if r['route_id']==chosen['route_id'] and r['feature_mode']==chosen['feature_mode'])
        selected.append({'target_id':target,'route_id':chosen['route_id'],'mode':chosen['feature_mode'],'bridge':chosen['bridge_count'],'score':scorer.score(chosen),'mask_path':chosen['forward_mask_path'],'dice':float(original['gt_dice_evaluation_only']),'oracle_dice':max(float(r['gt_dice_evaluation_only']) for r in rr)})
    return selected

def summarize(rr):
    return {'selected_dice':float(np.mean([r['dice'] for r in rr])),'oracle':float(np.mean([r['oracle_dice'] for r in rr])),'count':len(rr)}

def analyze_validation():
    router=module('pc_router',R/'code/analyze_propagation_quality_router.py')
    variants=['v0','v1','v2'];pool={v:quality(v,'validation') for v in variants}
    tp=[{**r,'feature_mode':'anchor_conditioned_target_pooling'} for r in read(OLD/TP/'propagation_quality_validation/propagation_quality.jsonl')]
    ids=sorted({r['target_id'] for r in pool['v0']});rng=np.random.default_rng(2026);order=rng.permutation(ids)
    folds={str(t):int(i%5) for i,t in enumerate(order)};save(R/'validation_folds.json',folds)
    report={}
    for v in variants:
        report[v]={'bridges':{str(b):float(np.mean([r['gt_dice_evaluation_only'] for r in pool[v] if r['bridge_count']==b])) for b in range(7)}}
        report[v]['long_mean']=float(np.mean([report[v]['bridges'][str(b)] for b in range(3,7)]))
        for scope in ['pc_only','tp_plus_pc']:
            rr=pool[v]+(tp if scope=='tp_plus_pc' else [])
            oof=[]
            for fold in range(5):
                train=[r for r in rr if folds[r['target_id']]!=fold]
                held=[r for r in rr if folds[r['target_id']]==fold]
                scorer=fit_fast(router,train,scope=='tp_plus_pc')
                oof.extend(select(router,scorer,held))
            oof.sort(key=lambda r:r['target_id']);jsonl(R/v/f'validation_{scope}_oof.jsonl',oof)
            report[v][scope]=summarize(oof)
    base_rows=read(R/'v0/validation_pc_only_oof.jsonl')
    for v in ['v1','v2']:
        rr=read(R/v/'validation_pc_only_oof.jsonl')
        delta=np.asarray([n['dice']-o['dice'] for n,o in zip(rr,base_rows)])
        boot=np.random.default_rng(2026).choice(delta,size=(10000,100),replace=True).mean(1)
        report[v]['paired_oof_delta']={'mean':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'improved':int((delta>1e-12).sum()),'degraded':int((delta< -1e-12).sum())}
    eligible=[v for v in ['v1','v2'] if report[v]['pc_only']['selected_dice']>=report['v0']['pc_only']['selected_dice']+.003 and report[v]['long_mean']>=report['v0']['long_mean']-.005]
    winner=max(eligible,key=lambda v:(report[v]['pc_only']['selected_dice'],v=='v1')) if eligible else None
    decision={'test_authorized_by_gate':winner is not None,'winner':winner,'eligible':eligible,'results':report}
    save(R/'validation_decision.json',decision)
    print(json.dumps(decision,indent=2),flush=True)
    return decision

def analyze_test(winner):
    router=module('pc_router',R/'code/analyze_propagation_quality_router.py')
    tp={s:[{**r,'feature_mode':'anchor_conditioned_target_pooling'} for r in read(OLD/TP/f'propagation_quality_{s}/propagation_quality.jsonl')] for s in ['validation','test']}
    report={}
    for v in ['v0',winner]:
        report[v]={}
        for scope in ['pc_only','tp_plus_pc']:
            val=quality(v,'validation')+(tp['validation'] if scope=='tp_plus_pc' else [])
            scorer=fit_fast(router,val,scope=='tp_plus_pc')
            save(R/v/f'final_router_{scope}.json',vars(scorer))
            rr=quality(v,'test')+(tp['test'] if scope=='tp_plus_pc' else [])
            chosen=select(router,scorer,rr)
            dest=R/v/f'final_test_{scope}';dest.mkdir(exist_ok=True)
            for row in chosen:
                new=dest/(row['target_id'].replace('::','__')+'.png');shutil.copy2(row['mask_path'],new);row['final_mask_path']=str(new)
            jsonl(R/v/f'test_{scope}_selected.jsonl',chosen)
            report[v][scope]=summarize(chosen)
        rr=quality(v,'test')
        report[v]['bridges']={str(b):float(np.mean([r['gt_dice_evaluation_only'] for r in rr if r['bridge_count']==b])) for b in range(7)}
    save(R/'test_results.json',report)

def report():
    config=json.loads((R/'config.json').read_text());decision=json.loads((R/'validation_decision.json').read_text())
    lines=['# Kvasir PC Adaptive Top-K + Spatial Compactness', '', 'SAM3-base@256, frozen original 800/100/100 split and 8 supports; no student. Only validation was used for variant selection.','', 'v0 = Top8 mean; v1 = anchor-area adaptive TopK mean; v2 = v1 - 0.05*(1 - topK bounding-box occupancy).','',f"Anchor K: {config['anchor_k']}", '', 'Validation selection uses 5-fold target-level out-of-fold Ridge predictions (seed 2026, ridge 1.0). No candidate from a held-out target is in its scorer training fold.', '', '| Variant | PC OOF Dice | TP+PC OOF Dice | PC oracle | Mean fixed b3-b6 |','|---|---:|---:|---:|---:|']
    for v,d in decision['results'].items():lines.append(f"| {v} | {d['pc_only']['selected_dice']:.6f} | {d['tp_plus_pc']['selected_dice']:.6f} | {d['pc_only']['oracle']:.6f} | {d['long_mean']:.6f} |")
    lines+=['',f"Predeclared test gate passed: {decision['test_authorized_by_gate']}. Winner: {decision['winner']}.", '', 'Gate: PC OOF improvement >=0.003 and fixed b3-b6 mean drop <=0.005. No test is run if no new variant passes. Test results are reported even if the validation-selected variant degrades.','', 'Anchor lesion area is a scale prior, not target lesion size ground truth. Bounding-box occupancy measures concentration, not anatomical correctness. Changes affect route generation; the original fixed candidate selection gap alone does not establish Top8 as its cause.']
    if (R/'test_results.json').exists():
        t=json.loads((R/'test_results.json').read_text());lines+=['','## Frozen test','', '| Variant | PC-only Dice | TP+PC Dice | PC oracle |','|---|---:|---:|---:|']
        for v,d in t.items():lines.append(f"| {v} | {d['pc_only']['selected_dice']:.6f} | {d['tp_plus_pc']['selected_dice']:.6f} | {d['pc_only']['oracle']:.6f} |")
    (R/'report.md').write_text('\n'.join(lines)+'\n')

def run(resume=False):
    if resume:
        assert (R/'FAILED').exists() and not (R/'COMPLETE').exists()
        (R/'FAILED').rename(R/f"FAILED_attempt_{time.time_ns()}")
        (R/'RESUMED').write_text(datetime.datetime.now().astimezone().isoformat())
    else:
        with (R/'STARTED').open('x') as f:f.write(datetime.datetime.now().astimezone().isoformat())
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    try:
        # Extraction has its own process so GPU memory is fully released afterwards.
        for command in ['extract-validation','generate-validation']:
            if resume and command=='extract-validation' and (R/'patch_similarity_maps_validation.npy').exists():
                audit=json.loads((R/'feature_replay_validation.json').read_text())
                assert audit['count']==900 and audit['top8_replay_max_abs_error']<=1e-5
                continue
            subprocess.run([sys.executable,'-u',__file__,command],cwd=P,check=True)
        for v in ['v1','v2']:infer(v,'validation')
        decision=analyze_validation();report()
        if decision['test_authorized_by_gate']:
            for command in ['extract-test','generate-test']:
                subprocess.run([sys.executable,'-u',__file__,command],cwd=P,check=True)
            infer(decision['winner'],'test');analyze_test(decision['winner']);report()
        (R/'COMPLETE').write_text(datetime.datetime.now().astimezone().isoformat())
    except BaseException:
        (R/'FAILED').write_text(traceback.format_exc());raise

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='setup':setup()
    elif cmd=='run':run()
    elif cmd=='resume':run(resume=True)
    elif cmd.startswith('extract-'):extract(cmd.split('-')[1])
    elif cmd.startswith('generate-'):generate(cmd.split('-')[1])
    else:raise ValueError(cmd)
