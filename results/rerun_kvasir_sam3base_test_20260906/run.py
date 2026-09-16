"""Prepare and run only the frozen Kvasir test routes on two GPUs."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

PROJECT = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
RUN = PROJECT / 'work/rerun_kvasir_sam3base_test_20260906'
OLD = PROJECT / 'work/rerun_c0_256_sam3knn_s256_base'
PROTO = PROJECT / 'work/kvasir_1pct_anchors/protocol'
CHECKPOINT = Path('/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt')
MODES = ['sam3enc_anchor_conditioned_target_pooling', 'sam3enc_anchor_conditioned_patch_correspondence']
ROUTE_HASHES = ['89db1347a13699f139cad501b5a6aa0aaee7b1cd6ddb621407dd6e31515e6e6f', '15708488c9eb1cb7155e3157de00939e048d50b3958447546556b9a63fbe4e54']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def prepare():
    if RUN.exists():
        raise RuntimeError(f'Refusing to overwrite existing run: {RUN}')
    records = rows(PROTO / 'merged_manifest.jsonl')
    support = rows(PROTO / 'support_manifest.jsonl')
    split_ids = {s: {r['merged_id'] for r in records if r['split'] == s} for s in ['train', 'validation', 'test']}
    anchors = {r['merged_id'] for r in support}
    assert [len(split_ids[s]) for s in ['train', 'validation', 'test']] == [800, 100, 100]
    assert len(anchors) == 8 and anchors <= split_ids['train']
    assert len(set.union(*split_ids.values())) == 1000
    assert set((PROTO / 'frozen_labeled_images.txt').read_text().splitlines()) == {r['image_path'] for r in support}
    sources = []
    for mode, expected_hash in zip(MODES, ROUTE_HASHES):
        source = OLD / 'stage1_feature_knn_b0_b6' / mode / 'test_pool0_stage1/routes.jsonl'
        assert sha(source) == expected_hash
        rr = rows(source)
        assert len(rr) == 700 and len({r['route_id'] for r in rr}) == 700
        assert {r['target_id'] for r in rr} == split_ids['test']
        assert collections.Counter((r['target_id'], r['bridge_count']) for r in rr) == collections.Counter({(t,b): 1 for t in split_ids['test'] for b in range(7)})
        for r in rr:
            assert r['target_split'] == 'test' and r['anchor_id'] in anchors
            assert set(r['bridge_ids']) <= split_ids['train']
            assert r['target_gt_used_for_search_or_inference'] is False
            assert len(r['bridge_ids']) == len(r['bridge_image_paths']) == r['bridge_count']
            for p in [r['anchor_image_path'], r['anchor_mask_path'], r['target_image_path'], r['target_mask_path_evaluation_only'], *r['bridge_image_paths']]:
                assert Path(p).is_file(), p
        sources.append(source)
    assert CHECKPOINT.is_file()
    RUN.mkdir()
    (RUN / 'protocol').mkdir()
    (RUN / 'code').mkdir()
    (RUN / 'logs').mkdir()
    for p in PROTO.iterdir():
        if p.is_file():
            shutil.copy2(p, RUN / 'protocol' / p.name)
    for mode, source in zip(MODES, sources):
        dest = RUN / 'quality_root' / mode / 'test_pool0_stage1/routes.jsonl'
        dest.parent.mkdir(parents=True)
        shutil.copy2(source, dest)
    code_files = ['eval_route_propagation_quality.py', 'summarize_c0_256_bridge_metrics.py', 'run_t21_dynamic_pseudovideo.py']
    for name in code_files:
        shutil.copy2(PROJECT / 'scripts' / name, RUN / 'code' / name)
    shutil.copy2(Path(__file__), RUN / 'run.py')
    config = {
        'created_at': datetime.datetime.now().astimezone().isoformat(),
        'scope': 'test only; fresh forward and return-cycle inference; no X3/B7 or training',
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha(CHECKPOINT),
        'knn': 'frozen SAM3-base@256 patch_mean, beam_width=32',
        'canvas': 256, 'bridge_range': [0, 6], 'test_targets': 100,
        'split_counts': {s: len(v) for s,v in split_ids.items()},
        'labeled_train': 8, 'unlabeled_train': 792,
        'protocol_sha256': {p.name: sha(p) for p in (RUN / 'protocol').iterdir()},
        'route_sha256': dict(zip(MODES, ROUTE_HASHES)),
        'code_sha256': {name: sha(PROJECT / 'scripts' / name) for name in code_files},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        'python': sys.executable, 'gpu_mapping': dict(zip(MODES, [0, 1])),
    }
    save(RUN / 'config.json', config)
    (RUN / 'PREPARED').touch()
    print(json.dumps({'run': str(RUN), 'config': config}, indent=2))

def run():
    assert (RUN / 'PREPARED').is_file()
    lock = RUN / 'STARTED'
    with lock.open('x') as f:
        f.write(datetime.datetime.now().astimezone().isoformat())
    processes = []
    try:
        for gpu, mode in enumerate(MODES):
            cmd = [sys.executable, '-u', str(RUN / 'code/eval_route_propagation_quality.py'), '--checkpoint', str(CHECKPOINT), '--mode', mode, '--root', str(RUN / 'quality_root'), '--split', 'test', '--canvas', '256']
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['PYTHONUNBUFFERED'] = '1'
            with (RUN / 'logs' / (mode + '.log')).open('w') as log:
                p = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            processes.append((mode, p))
            save(RUN / (mode + '.process.json'), {'pid': p.pid, 'gpu': gpu, 'command': cmd})
            print(f'Started {mode} GPU={gpu} PID={p.pid}', flush=True)
        failures = []
        for mode, p in processes:
            rc = p.wait()
            save(RUN / (mode + '.exit.json'), {'pid': p.pid, 'exit_code': rc})
            if rc:
                failures.append((mode, rc))
        if failures:
            raise RuntimeError(f'Inference failed: {failures}')
        for mode in MODES:
            rr = rows(RUN / 'quality_root' / mode / 'test_pool0_stage1/routes.jsonl')
            results = rows(RUN / 'quality_root' / mode / 'propagation_quality_test/propagation_quality.jsonl')
            assert len(results) == 700
            assert {r['route_id'] for r in rr} == {r['route_id'] for r in results}
            assert all(r['status'] == 'success' and Path(r['forward_mask_path']).is_file() for r in results)
        subprocess.run([sys.executable, str(RUN / 'code/summarize_c0_256_bridge_metrics.py'), '--quality-root', str(RUN / 'quality_root'), '--split', 'test', '--modes', *MODES, '--min-bridge', '0', '--max-bridge', '6', '--output-json', str(RUN / 'bridge_b0_b6_test_metrics.json'), '--output-tsv', str(RUN / 'bridge_b0_b6_test_metrics.tsv')], cwd=PROJECT, check=True)
        new = json.loads((RUN / 'bridge_b0_b6_test_metrics.json').read_text())
        old = json.loads((OLD / 'current_base_test/bridge_b0_b6_test_metrics.json').read_text())
        lines = ['# Kvasir SAM3-base test rerun', '', 'Frozen original 800/100/100 split and 8 labeled anchors; test only, 100 targets per mode. SAM3-base@256 topology, base sam3.pt propagation, canvas 256, b0-b6. Fresh forward and return-cycle inference; no student or B7 selection.', '', '| Bridge | Target Dice | Old target | Delta | Patch Dice | Old patch | Delta |', '|---|---:|---:|---:|---:|---:|---:|']
        comparisons = []
        for b in range(7):
            key = f'bridge_{b}'
            vals = []
            for mode in MODES:
                n, o = new['modes'][mode][key]['dice'], old['modes'][mode][key]['dice']
                vals.extend([n, o, n-o])
                comparisons.append({'mode': mode, 'bridge': b, 'dice': n, 'old_dice': o, 'delta': n-o})
            lines.append(f'| b{b} | ' + ' | '.join(f'{v:.6f}' for v in vals) + ' |')
        save(RUN / 'comparison_with_previous.json', comparisons)
        (RUN / 'report.md').write_text('\n'.join(lines) + '\n')
        (RUN / 'COMPLETE').write_text(datetime.datetime.now().astimezone().isoformat())
        print('COMPLETE', flush=True)
    except BaseException:
        (RUN / 'FAILED').write_text(traceback.format_exc())
        raise

if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif sys.argv[1:] == ['run']:
        run()
    else:
        raise SystemExit('Usage: rerun_test.py prepare|run')
