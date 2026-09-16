"""Dry-run the TN3K validate/evaluate code path on already-propagated targets.

Builds a shadow experiment dir (R/dryrun) with a reduced pool membership and the
partial propagation rows, then calls the real pipeline's validate()/evaluate() with
R pointed at the shadow dir. Read-only w.r.t. the real experiment.
"""
from pathlib import Path
import json, shutil, importlib.util, sys, traceback

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R = P / 'new_project/experiments/tn3k_busi_factorial_20260915'
DR = R / 'dryrun'
MODE = 'sam3enc_anchor_conditioned_target_pooling'
GROUPS = ['raw_top1', 'centered_top1', 'raw_top2', 'centered_top2', 'original_per_bridge']
LIMIT = 120


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    if DR.exists(): shutil.rmtree(DR)
    (DR / 'code').mkdir(parents=True)
    for f in (R / 'code').glob('*.py'):
        shutil.copy2(f, DR / 'code' / f.name)
    shutil.copy2(R / 'calibration_frozen.json', DR / 'calibration_frozen.json')
    shutil.copy2(R / 'pool_audit.json', DR / 'pool_audit.json')
    shutil.copy2(R / 'pipeline.py', DR / 'pipeline.py')

    members = json.loads((R / 'pool_membership_frozen.json').read_text())['groups']
    folds = json.loads((R / 'folds_frozen.json').read_text())
    new_members = {}; counts = {}
    for split in ('validation', 'test'):
        src = R / f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl'
        rows = {r['route_id']: r for r in read(src)}
        g = members[split]
        ok = [tid for tid in sorted(g['raw_top1'])
              if all(rid in rows for grp in GROUPS for rid in g[grp][tid])][:LIMIT]
        assert len(ok) >= 20, (split, len(ok))
        new_members[split] = {grp: {tid: g[grp][tid] for tid in ok} for grp in GROUPS}
        counts[split] = len(ok)
        keep = set()
        for tid in ok:
            for grp in GROUPS: keep.update(g[grp][tid])
        out = DR / f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(''.join(json.dumps(rows[rid], ensure_ascii=False) + '\n' for rid in sorted(keep)))
        print(f'{split}: {len(ok)} targets, {len(keep)} routes')
    (DR / 'pool_membership_frozen.json').write_text(
        json.dumps({'time': 0, 'groups': new_members, 'test_GT_read': False}))
    val_ids = set(new_members['validation']['raw_top1'])
    (DR / 'folds_frozen.json').write_text(json.dumps({k: v for k, v in folds.items() if k in val_ids}))

    spec = importlib.util.spec_from_file_location('tn3k_pipe_dryrun', str(P / 'new_project/tn3k_busi_factorial.py'))
    m = importlib.util.module_from_spec(spec); sys.modules['tn3k_pipe_dryrun'] = m; spec.loader.exec_module(m)
    m.R = DR
    m.COUNTS = dict(train=2303, validation=counts['validation'], test=counts['test'])

    print('=== dry-run validate ===')
    m.validate()
    print('=== dry-run evaluate ===')
    # satisfy the input-hash gate with the files that exist inside the shadow dir
    files = [DR / 'pipeline.py', DR / 'folds_frozen.json', DR / 'pool_membership_frozen.json'] + list((DR / 'code').glob('*.py'))
    m.save(DR / 'input_hashes.json', {str(f): m.sha(f) for f in files})
    m.evaluate()
    print()
    print('=== DRY RUN OK ===')
    print((DR / 'report.md').read_text())


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
