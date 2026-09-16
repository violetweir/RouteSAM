"""Fixed b0 anchor-transfer diagnostic; original validation/support identities."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R = P / 'work/kvasir_rethink_20260908/anchor_matrix'
OLD = P / 'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6'
PROTO = P / 'work/kvasir_1pct_anchors/protocol'
CACHE = P / 'work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features/sam3_base_s256_features.npz'
CKPT = Path('/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt')
MODES = ['sam3enc_anchor_conditioned_target_pooling', 'sam3enc_anchor_conditioned_patch_correspondence']

def read(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []

def save(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in rows))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else P / p

def setup():
    import numpy as np
    assert not (R/'config.json').exists(), 'Already set up'
    (R/'code').mkdir(parents=True, exist_ok=True)
    for name in ['stage1_feature_knn_routes.py', 'eval_route_propagation_quality.py', 'run_t21_dynamic_pseudovideo.py']:
        shutil.copy2(P/'scripts'/name, R/'code'/name)
    # The evaluator dynamically imports t21; pin its path to this snapshot.
    q = R/'code/eval_route_propagation_quality.py'
    q.write_text(q.read_text().replace('ROOT / "scripts/run_t21_dynamic_pseudovideo.py"', 'Path(__file__).parent / "run_t21_dynamic_pseudovideo.py"'))
    shutil.copytree(PROTO, R/'protocol', dirs_exist_ok=True)
    records = read(R/'protocol/merged_manifest.jsonl')
    support = read(R/'protocol/support_manifest.jsonl')
    assert len(support) == 8 and len(records) == 1000
    assert all(r['split']=='train' for r in support)
    stage = module('matrix_stage', R/'code/stage1_feature_knn_routes.py')
    anchors = stage.t21.human_pool(support, 512)
    cache = np.load(CACHE)
    assert cache['anchor_ids'].tolist() == [a['anchor_id'] for a in anchors]
    assert sha(CACHE) == 'ab8ca194f9c789ad9f6b641d382ed848751f29b23ac0bdb29a9c6722fd599907'
    assert sha(R/'protocol/merged_manifest.jsonl') == 'eb74d0ea534450d7b97ae93319260b7cbbd78e5b74b2b615c777b1263eb91322'
    assert sha(CKPT) == '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
    routes = []
    for ti, target in enumerate(records):
        if target['split'] != 'validation': continue
        for ai, a in enumerate(anchors):
            s = float(cache['cond_target'][ai, ti])
            r = stage.make_route(target, 0, a, [], (s,s), records)
            r.update(anchor_index=ai, tp_retrieval_score=s, pc_retrieval_score=float(cache['cond_correspondence'][ai,ti]))
            routes.append(r)
    routes.sort(key=lambda r:(r['target_id'],r['anchor_index']))
    assert len(routes)==800 and len({r['route_id'] for r in routes})==800
    sources = {}
    duplicate_checks = 0
    for mode in MODES:
        for r in read(OLD/mode/'propagation_quality_validation/propagation_quality.jsonl'):
            if r['bridge_count'] != 0: continue
            fp = resolve(r['forward_mask_path'])
            assert fp.exists() and sha(fp)==r['forward_mask_sha256']
            r['forward_mask_path'] = str(fp)
            if r['route_id'] in sources:
                duplicate_checks += 1
                assert sources[r['route_id']]['forward_mask_sha256']==r['forward_mask_sha256']
            sources[r['route_id']] = r
    reused = []
    for r in routes:
        if r['route_id'] not in sources: continue
        old = sources[r['route_id']]
        for k in ['anchor_id','anchor_mask_sha256','anchor_box_xywh_normalized','bridge_ids','target_id']:
            assert r[k]==old[k], k
        reused.append({**old, **r, 'reused_original':True})
    jsonl(R/'routes.jsonl', routes)
    jsonl(R/'reused.jsonl', reused)
    # Select replay cases by route ordering/anchor identity, never target Dice.
    replay=[]
    for ai in sorted({r['anchor_index'] for r in reused}):
        replay.append(next(r for r in reused if r['anchor_index']==ai))
    jsonl(R/'replay_routes.jsonl', replay)
    save(R/'config.json', dict(split='validation', targets=100, anchors=8, routes=800,
        canvas=256, internal_resolution=1008, frame_resize='PIL BILINEAR',
        prompt='Original anchor GT-derived box only, video propagation; no GT mask injection',
        checkpoint=str(CKPT), checkpoint_sha256=sha(CKPT), feature_cache_sha256=sha(CACHE),
        reused=len(reused), pending=800-len(reused), replay_count=len(replay),
        existing_duplicate_hash_checks=duplicate_checks, cycle='forward-only diagnostic; no cycle features used',
        gpu=1, allocator_fraction=.25, target_gt_used_for_search_or_inference=False,
        actual_metrics=['TP top1','PC top1','SAM confidence','mask agreement'],
        oracle_metrics=['all8','TP top2','PC top2','TP top1 + PC alternate'],
        note='Diagnostic anchor sweep, not seven-candidate-budget performance comparison',
        code_sha256={p.name:sha(p) for p in (R/'code').glob('*.py')}))
    print(json.dumps(json.loads((R/'config.json').read_text()), indent=2), flush=True)

def infer():
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_video_model
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    ev = module('matrix_eval', R/'code/eval_route_propagation_quality.py')
    model = build_sam3_video_model(checkpoint_path=str(CKPT), load_from_HF=False, device='cuda', compile=False)
    model.eval()
    assert model.image_size==1008, getattr(model,'image_size',None)
    replay_audit=[]
    for r in read(R/'replay_routes.jsonl'):
        tr = ev.propagate_with_trace(model,[r['anchor_image_path'], r['target_image_path']],r['anchor_box_xywh_normalized'],256)
        old = np.asarray(Image.open(r['forward_mask_path']).convert('L'))>127
        difference = int(np.count_nonzero(old != tr['final_mask']))
        replay_audit.append(dict(route_id=r['route_id'],anchor_id=r['anchor_id'],differing_pixels=difference,
            old_score=r['final_sam_score'],new_score=tr['final_sam_score']))
        save(R/'replay_audit.json', replay_audit)
        print('REPLAY',r['route_id'],difference,flush=True)
        assert difference==0, 'Replay changed predictions; do not mix cached and new inference'
    out = R/'quality'
    out.mkdir(exist_ok=True)
    if not (out/'propagation_quality.jsonl').exists():
        jsonl(out/'propagation_quality.jsonl',read(R/'reused.jsonl'))
    rows = ev.evaluate(model,read(R/'routes.jsonl'),out,256,True,True,no_cycle=True)
    assert len(rows)==800
    save(R/'INFERENCE_COMPLETE.json',dict(routes=len(rows),time=time.time(),max_cuda_allocated=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    os.chdir(P)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['setup','infer'])
    args=parser.parse_args()
    if args.action=='setup':setup()
    else:infer()
