"""Reproduce frozen stage-one routers; no GPU inference or student network."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
HISTORY = P/'work/rerun_kvasir_sam3base_test_20260906'
VAL = P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6'
MODES = ['anchor_conditioned_target_pooling', 'anchor_conditioned_patch_correspondence']

def read(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def save(p, v):
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False)+'\n')

def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else P/p

def mask(p):
    return np.asarray(Image.open(resolve(p)).convert('L').resize((256,256), Image.Resampling.NEAREST)) > 127

def metric(a, b):
    i = int((a & b).sum()); n = int(a.sum())+int(b.sum())
    return 2*i/max(n,1), i/max(n-i,1)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=['old','new'], required=True)
    ap.add_argument('--quality-root', type=Path, default=HISTORY/'quality_root')
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    source = HISTORY/'final_masks_no_student/analyze_propagation_quality_router.py'
    spec = importlib.util.spec_from_file_location('frozen_router', source)
    router = importlib.util.module_from_spec(spec); spec.loader.exec_module(router)
    modes = MODES if args.variant == 'old' else MODES[:1]
    protocol = read(HISTORY/'protocol/merged_manifest.jsonl')
    ids = {s:{r['merged_id'] for r in protocol if r['split']==s} for s in ['validation','test']}
    assert len(ids['test'])==len(ids['validation'])==100 and not ids['test'] & ids['validation']
    rows, inputs = [], {}
    for m in modes:
        f = args.quality_root/('sam3enc_'+m)/'propagation_quality_test/propagation_quality.jsonl'
        rr = read(f)
        assert len(rr)==700 and {r['target_id'] for r in rr}==ids['test']
        assert len({(r['target_id'],r['bridge_count']) for r in rr})==700
        assert {r['bridge_count'] for r in rr}==set(range(7))
        assert all(r['status']=='success' for r in rr)
        rows.extend(dict(r,feature_mode=m) for r in rr)
        inputs[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'evaluate_stage1.py')
    scopes = [('target_pooling',MODES[:1])]
    if args.variant=='old':
        scopes += [('patch_correspondence',MODES[1:]),('combined',MODES)]
    decisions, summaries = {}, []
    for minb in [0,3]:
        for scope, mm in scopes:
            f = HISTORY/f'final_masks_no_student/router_b{minb}_b6_{scope}.json'
            scorer = router.Ridge(**json.loads(f.read_text()))
            dest = out/f'b{minb}_b6'/scope
            (dest/'masks').mkdir(parents=True)
            shutil.copy2(f,dest/'router.json')
            inputs[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
            groups = router.grouped([r for r in rows if r['feature_mode'] in mm and r['bridge_count']>=minb])
            chosen = []
            for target, group in sorted(groups.items()):
                clean = [{k:v for k,v in r.items() if not k.startswith('gt_') and 'evaluation_only' not in k} for r in group]
                r = max(clean,key=lambda r:(scorer.score(r),-int(r['bridge_count']),r['route_id']))
                output_mask = dest/'masks'/(target.replace('::','__')+'.png')
                shutil.copy2(resolve(r['forward_mask_path']),output_mask)
                chosen.append(dict(target_id=target,route_id=r['route_id'],mode=r['feature_mode'],bridge=r['bridge_count'],score=scorer.score(r),mask=str(output_mask)))
            save(dest/'selected_before_gt.json',chosen)
            decisions[(minb,scope)] = (mm,chosen)
    save(out/'INPUTS_AND_CHOICES_FROZEN.json',inputs)
    # No test GT is opened before this point.
    lookup, per_candidate = {}, []
    for r in rows:
        d,i = metric(mask(r['forward_mask_path']),mask(r['target_mask_path_evaluation_only']))
        lookup[(r['feature_mode'],r['target_id'],r['bridge_count'])]=(d,i)
        per_candidate.append(dict(mode=r['feature_mode'],target_id=r['target_id'],bridge=r['bridge_count'],dice=d,iou=i,q_cycle=r['q_cycle']))
    save(out/'per_candidate_metrics.json',per_candidate)
    fixed = []
    for m in modes + (['combined_mean'] if args.variant=='old' else []):
        for b in range(7):
            rr = [r for r in per_candidate if r['bridge']==b and (m=='combined_mean' or r['mode']==m)]
            fixed.append(dict(mode=m,bridge=b,n=len(rr),dice=float(np.mean([r['dice'] for r in rr])),iou=float(np.mean([r['iou'] for r in rr])),median=float(np.median([r['dice'] for r in rr])),q_cycle=float(np.mean([r['q_cycle'] for r in rr]))))
    for (minb,scope),(mm,chosen) in decisions.items():
        selected = [dict(r,dice=lookup[(r['mode'],r['target_id'],r['bridge'])][0],iou=lookup[(r['mode'],r['target_id'],r['bridge'])][1]) for r in chosen]
        oracle = float(np.mean([max(lookup[(m,t,b)][0] for m in mm for b in range(minb,7)) for t in sorted(ids['test'])]))
        d = float(np.mean([r['dice'] for r in selected]))
        result=dict(scope=scope,min_bridge=minb,max_bridge=6,n=100,dice=d,iou=float(np.mean([r['iou'] for r in selected])),oracle=oracle,gap=oracle-d)
        save(out/f'b{minb}_b6'/scope/'per_target_metrics.json',selected)
        summaries.append(result)
    save(out/'results.json',dict(variant=args.variant,fixed_bridge=fixed,routers=summaries,metric_resolution=256,aggregation='per-image macro mean',fresh_gpu_inference=False))
    (out/'COMPLETE').touch()
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    main()
