"""Read-only input audit; write a new, isolated stage-one reproduction report."""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
RUN = P/'work/rerun_kvasir_sam3base_test_20260906'
OLD = P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6'
SAVED = RUN/'final_masks_no_student'
OUT = P/'new_project/experiments/stage1_reproducibility_audit_20260912_py312'
MODES = ['anchor_conditioned_target_pooling', 'anchor_conditioned_patch_correspondence']

def read(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else P/p

def mask(p):
    return np.asarray(Image.open(resolve(p)).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127

def metrics(a, b):
    i = int((a & b).sum()); n = int(a.sum()) + int(b.sum())
    return 2*i/max(n, 1), i/max(n-i, 1)

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    source = SAVED/'analyze_propagation_quality_router.py'
    proto = json.loads((SAVED/'protocol.json').read_text())
    assert sha(source) == proto['source_sha256'] == sha(P/'scripts/analyze_propagation_quality_router.py')
    spec = importlib.util.spec_from_file_location('frozen_router', source)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    manifest = read(RUN/'protocol/merged_manifest.jsonl')
    ids = {s: {r['merged_id'] for r in manifest if r['split'] == s} for s in ['train', 'validation', 'test']}
    assert [len(ids[s]) for s in ['train', 'validation', 'test']] == [800, 100, 100]
    assert all(not ids[a] & ids[b] for a,b in [('train','test'),('train','validation'),('validation','test')])
    hashes = json.loads((SAVED/'input_sha256.json').read_text())
    for p,h in hashes.items():
        assert sha(p) == h, p
    val, test, previous = [], [], {}
    for m in MODES:
        v = read(OLD/('sam3enc_'+m)/'propagation_quality_validation/propagation_quality.jsonl')
        t = read(RUN/'quality_root'/('sam3enc_'+m)/'propagation_quality_test/propagation_quality.jsonl')
        old = read(OLD/('sam3enc_'+m)/'propagation_quality_test/propagation_quality.jsonl')
        for rows, split in [(v,'validation'),(t,'test'),(old,'test')]:
            assert len(rows) == 700 and {r['target_id'] for r in rows} == ids[split]
            assert len({(r['target_id'],r['bridge_count']) for r in rows}) == 700
            assert {r['bridge_count'] for r in rows} == set(range(7))
        val.extend(dict(r,feature_mode=m) for r in v)
        test.extend(dict(r,feature_mode=m) for r in t)
        previous.update({(m,r['target_id'],r['bridge_count']):r for r in old})
    choices, fit_deltas = {}, {}
    for scope, modes in [('target_pooling',MODES[:1]),('combined',MODES)]:
        scorer = mod.Ridge.fit([r for r in val if r['feature_mode'] in modes],1.0,scope=='combined')
        frozen = mod.Ridge(**json.loads((SAVED/f'router_b0_b6_{scope}.json').read_text()))
        delta = max(abs(a-b) for k in ['means','stds','weights'] for a,b in zip(getattr(scorer,k),getattr(frozen,k)))
        assert delta < 1e-12, (scope, delta)
        fit_deltas[scope] = delta
        selected = []
        for target, rows in sorted(mod.grouped([r for r in test if r['feature_mode'] in modes]).items()):
            clean = [{k:v for k,v in r.items() if not k.startswith('gt_') and 'evaluation_only' not in k} for r in rows]
            r = max(clean,key=lambda r:(scorer.score(r),-int(r['bridge_count']),r['route_id']))
            selected.append(dict(target_id=target, route_id=r['route_id'], feature_mode=r['feature_mode'], bridge_count=r['bridge_count'], score=scorer.score(r)))
        choices[scope] = selected
    (OUT/'choices_frozen_before_gt.json').write_text(json.dumps(choices,indent=2))
    # Read target GT only after both sets of choices have been committed.
    lookup, gt_hashes, identical_masks = {}, {}, 0
    for r in test:
        key = (r['feature_mode'],r['target_id'],r['bridge_count'])
        old = previous[key]
        gt = resolve(r['target_mask_path_evaluation_only'])
        gh = sha(gt)
        assert gh == sha(resolve(old['target_mask_path_evaluation_only']))
        assert r['target_id'] not in gt_hashes or gt_hashes[r['target_id']] == gh
        gt_hashes[r['target_id']] = gh
        a = mask(r['forward_mask_path'])
        assert np.array_equal(a,mask(old['forward_mask_path'])), key
        identical_masks += 1
        d,i = metrics(a,mask(gt))
        assert abs(d-r['gt_dice_evaluation_only']) < 1e-12
        lookup[key] = (d,i)
    results = []
    for scope,modes in [('target_pooling',MODES[:1]),('combined',MODES)]:
        saved_choices = {r['target_id']:r for r in read(SAVED/'b0_b6'/scope/'selected_masks.jsonl')}
        values = []
        for r in choices[scope]:
            old = saved_choices[r['target_id']]
            assert r['route_id'] == old['route_id'] and 'sam3enc_'+r['feature_mode'] == old['route_mode']
            path = resolve(old['final_mask_path'])
            assert sha(path) == old['mask_sha256']
            source = next(t for t in test if (t['feature_mode'],t['target_id'],t['bridge_count']) == (r['feature_mode'],r['target_id'],r['bridge_count']))
            assert np.array_equal(mask(path),mask(source['forward_mask_path']))
            values.append(lookup[(r['feature_mode'],r['target_id'],r['bridge_count'])])
        d,i = np.mean(values,axis=0)
        oracle = np.mean([max(lookup[(m,t,b)][0] for m in modes for b in range(7)) for t in sorted(ids['test'])])
        original = json.loads((SAVED/'b0_b6'/scope/'summary.json').read_text())
        assert abs(d-original['dice']) < 1e-12 and abs(i-original['iou']) < 1e-12
        assert abs(oracle-original['oracle_dice_evaluation_only']) < 1e-12
        results.append(dict(scope=scope,dice=float(d),iou=float(i),oracle=float(oracle),selection_matches=100,refit_parameter_max_delta=fit_deltas[scope]))
    report = dict(status='PASS',fresh_gpu_inference=False,candidate_masks_recomputed=1400,old_vs_rerun_pixel_identical=identical_masks,test_targets=100,metric_resolution=256,resize='NEAREST',binarization='uint8 > 127',aggregation='per-image macro mean',validation_gt_used_to_fit_router=True,test_gt_used_for_selection=False,input_hashes_verified=len(hashes),results=results)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'COMPLETE').touch()
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
