"""Independent verification of the TN3K results.

Recomputes every headline number from scratch, deliberately NOT reusing
results.json / test_candidate_metrics.json:
  - GT loaded straight from /Data_8TB/lht/data/tn3k/test-mask/<idx>.jpg (path built
    from the target_id itself, not from the manifest)
  - forward masks loaded from the propagated rows' recorded mask paths
  - Dice/IoU recomputed with the frozen evaluation standard (gray, NEAREST 256, >127)
Also checks the freeze ordering: masks and choices must predate GT scoring.
"""
from pathlib import Path
import json, collections, hashlib
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R = P / 'new_project/experiments/tn3k_busi_factorial_20260915'
DATA = Path('/Data_8TB/lht/data/tn3k')
M = 'sam3enc_anchor_conditioned_target_pooling'
GROUPS = ['raw_top1', 'centered_top1', 'raw_top2', 'centered_top2', 'original_per_bridge']


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def bmp(p): return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127
def metric(a, b):
    i = int((a & b).sum()); s = int(a.sum()) + int(b.sum())
    return (2 * i / s if s else 1.0), (i / (s - i) if s - i else 1.0)


def gt_for(target_id):
    """Build the GT path independently from the target_id."""
    split, idx = target_id.split('::')[1], target_id.split('::')[2]
    sub = 'test-mask' if split == 'test' else 'trainval-mask'
    p = DATA / sub / f'{idx}.jpg'
    assert p.exists(), p
    return bmp(p)


def main():
    ok = True
    def check(label, got, want, tol=5e-7):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f'  [{"OK " if good else "FAIL"}] {label}: recomputed {got:.6f} vs report {want:.6f}')

    report = (R / 'report.md').read_text()
    res = json.loads((R / 'results.json').read_text())
    choices = json.loads((R / 'test_choices_frozen.json').read_text())
    members = json.loads((R / 'pool_membership_frozen.json').read_text())['groups']['test']
    qrows = {r['route_id']: r for r in read(R / f'quality_root/{M}/propagation_quality_test/propagation_quality.jsonl')}

    print(f'=== 0. coverage ===')
    print(f'  propagated test candidates: {len(qrows)}')
    print(f'  test targets: {len({r["target_id"] for r in qrows.values()})}')
    print(f'  choices frozen per group: ' + ', '.join(f'{g}={len(choices["choices"][g])}' for g in GROUPS))
    print(f'  test_choices_frozen.test_GT_read = {choices["test_GT_read"]}')

    # ---- GT cache, loaded independently
    tids = sorted(members['raw_top1'])
    gt = {t: gt_for(t) for t in tids}
    assert all(g.any() for g in gt.values()), 'empty GT found'

    print('\n=== 1. Router-selected test Dice / IoU (independent recompute) ===')
    for g in GROUPS:
        d = []; i = []
        for t, rid in zip(tids, choices['choices'][g]):
            m = bmp(qrows[rid]['forward_mask_path']); a, b = metric(m, gt[t])
            d.append(a); i.append(b)
        check(f'{g} Dice', float(np.mean(d)), res['test'][g]['dice'])
        check(f'{g} IoU ', float(np.mean(i)), res['test'][g]['iou'])

    print('\n=== 2. b0-b6 per-bridge table (independent recompute) ===')
    def per_b(group):
        out = collections.defaultdict(list)
        for t in tids:
            for b, rid in enumerate(members[group][t]):
                m = bmp(qrows[rid]['forward_mask_path'])
                out[b].append(metric(m, gt[t])[0])
        return {b: float(np.mean(v)) for b, v in out.items()}
    ob, cb = per_b('original_per_bridge'), per_b('centered_top1')
    rep_o = res['test']['original_per_bridge']['fixed_rank1_b0_b6']
    rep_c = res['test']['centered_top1']['fixed_rank1_b0_b6']
    for b in range(7):
        check(f'b{b} original ', ob[b], rep_o[b])
        check(f'b{b} calibrated', cb[b], rep_c[b])
    print(f'  best b: original=b{int(np.argmax([ob[b] for b in range(7)]))}, '
          f'calibrated=b{int(np.argmax([cb[b] for b in range(7)]))}')

    print('\n=== 3. candidate-pool Oracle (independent recompute) ===')
    for g in GROUPS:
        orc = []
        for t in tids:
            orc.append(max(metric(bmp(qrows[rid]['forward_mask_path']), gt[t])[0] for rid in members[g][t]))
        check(f'{g} Oracle', float(np.mean(orc)), res['test'][g]['oracle_dice'])

    print('\n=== 4. mask hashes still match what propagation recorded ===')
    bad = 0
    for rid, r in qrows.items():
        h = hashlib.sha256(Path(r['forward_mask_path']).read_bytes()).hexdigest()
        if h != r['forward_mask_sha256']: bad += 1
    print(f'  masks verified: {len(qrows)}, hash mismatches: {bad}')
    ok &= (bad == 0)

    print('\n=== 5. freeze ordering (masks/choices before GT scoring) ===')
    import os
    t_pred = (R / 'test_PREDICTIONS_FROZEN.json').stat().st_mtime
    t_choice = (R / 'test_choices_frozen.json').stat().st_mtime
    t_result = (R / 'results.json').stat().st_mtime
    print(f'  predictions frozen  {t_pred:.0f}  <= choices frozen {t_choice:.0f}  <= results {t_result:.0f}')
    print(f'  ordering ok: {t_pred <= t_choice <= t_result}')
    ok &= (t_pred <= t_choice <= t_result)

    print('\n=== 6. report.md numbers match results.json ===')
    for g in GROUPS:
        x = res['test'][g]
        line = [l for l in report.splitlines() if l.startswith(f'| {g} |')][0]
        cells = [c.strip() for c in line.strip('|').split('|')]
        good = (abs(float(cells[3]) - x['dice']) < 5e-7 and abs(float(cells[5]) - x['oracle_dice']) < 5e-7
                and int(cells[1]) == x['candidates_per_target'])
        ok &= good
        print(f'  [{"OK " if good else "FAIL"}] {g}: report row {cells[1:]}')
    print(f'\n=== {"ALL INDEPENDENT CHECKS PASSED" if ok else "SOME CHECKS FAILED"} ===')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
