"""Why is TN3K so much worse than BUSI? Same-protocol comparison of the two datasets.

Both use: SAM3-base, GT tight box, no text, canvas256, patch_mean KNN, beam32, b0-b6.
Compares the things that can actually explain the gap:
  A. candidate-pool ceiling (Oracle) and how many targets are simply unreachable
  B. whether "choosing the right reference" matters (per-target spread across anchors)
  C. object size (foreground fraction / pixel count at the 256 evaluation resolution)
"""
from pathlib import Path
import json, collections
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E = P / 'new_project/experiments'
M = 'sam3enc_anchor_conditioned_target_pooling'
SP = {'BUSI': E / 'busi_calibration_factorial_20260914', 'TN3K': E / 'tn3k_busi_factorial_20260915'}
GTDIR = {'BUSI': Path('/Data_8TB/lht/MK-UNet/BUSI/BUSI_split/test/masks'),
         'TN3K': Path('/Data_8TB/lht/data/tn3k/test-mask')}
GROUPS = ['raw_top1', 'centered_top1', 'raw_top2', 'centered_top2', 'original_per_bridge']


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def bmp(p): return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127


def gt_path(ds, tid):
    parts = tid.split('::')
    if ds == 'BUSI':
        return GTDIR[ds] / f'{parts[1]}_mask.png'
    return GTDIR[ds] / f'{parts[2]}.jpg'


def load(ds):
    root = SP[ds]
    metrics = json.loads((root / 'test_candidate_metrics.json').read_text())
    members = json.loads((root / 'pool_membership_frozen.json').read_text())['groups']['test']
    rows = {r['route_id']: r for r in read(root / f'quality_root/{M}/propagation_quality_test/propagation_quality.jsonl')}
    return metrics, members, rows


def main():
    print('%-6s %8s %10s %10s %10s %10s %10s %10s' % (
        'ds', 'targets', 'orc7_mean', 'orc7_med', 'orc14_mean', 'orc14_med', '<0.5', '<0.2'))
    summary = {}
    for ds in ('BUSI', 'TN3K'):
        metrics, members, rows = load(ds)
        tids = sorted(members['original_per_bridge'])
        o7 = np.array([max(metrics[r]['dice'] for r in members['original_per_bridge'][t]) for t in tids])
        o14 = np.array([max(metrics[r]['dice'] for r in members['centered_top2'][t]) for t in tids])
        summary[ds] = dict(tids=tids, metrics=metrics, members=members, rows=rows, o7=o7, o14=o14)
        print('%-6s %8d %10.4f %10.4f %10.4f %10.4f %9.1f%% %9.1f%%' % (
            ds, len(tids), o7.mean(), np.median(o7), o14.mean(), np.median(o14),
            100 * (o14 < 0.5).mean(), 100 * (o14 < 0.2).mean()))

    print('\n=== C. object size at the 256 evaluation resolution ===')
    print('%-6s %10s %10s %12s %12s %12s' % ('ds', 'frac_mean', 'frac_med', 'px_mean', 'px_med', 'px<300'))
    for ds in ('BUSI', 'TN3K'):
        s = summary[ds]
        fr = []; px = []
        for t in s['tids']:
            g = bmp(gt_path(ds, t)); fr.append(g.mean()); px.append(int(g.sum()))
        fr = np.array(fr); px = np.array(px)
        s['frac'] = fr; s['px'] = px
        print('%-6s %10.4f %10.4f %12.1f %12.1f %11.1f%%' % (
            ds, fr.mean(), np.median(fr), px.mean(), np.median(px), 100 * (px < 300).mean()))

    print('\n=== B. does choosing the RIGHT reference matter? (per-target spread of per-anchor mean Dice) ===')
    print('%-6s %10s %10s %10s %10s' % ('ds', 'anchors/t', 'spread_mean', 'spread_med', 'best-anchor'))
    for ds in ('BUSI', 'TN3K'):
        s = summary[ds]
        by_ta = collections.defaultdict(list)
        for rid, r in s['rows'].items():
            m = s['metrics'].get(rid)
            if m is not None:
                by_ta[(r['target_id'], r['anchor_id'])].append(m['dice'])
        spreads = []; nanc = []; bests = []
        for t in s['tids']:
            vals = [np.mean(v) for (tt, a), v in by_ta.items() if tt == t]
            if len(vals) < 2: continue
            spreads.append(max(vals) - min(vals)); nanc.append(len(vals)); bests.append(max(vals))
        print('%-6s %10.1f %10.4f %10.4f %10.4f' % (
            ds, np.mean(nanc), np.mean(spreads), np.median(spreads), np.mean(bests)))

    print('\n=== D. correlation: object size vs achievable Oracle (centered_top2 14 cand) ===')
    for ds in ('BUSI', 'TN3K'):
        s = summary[ds]
        r = float(np.corrcoef(s['px'], s['o14'])[0, 1])
        r2 = float(np.corrcoef(s['frac'], s['o14'])[0, 1])
        # oracle among small vs large objects
        small = s['o14'][s['px'] < np.median(s['px'])].mean()
        large = s['o14'][s['px'] >= np.median(s['px'])].mean()
        print(f'  {ds}: corr(px, oracle)={r:+.4f}  corr(frac, oracle)={r2:+.4f}  '
              f'small-half oracle={small:.4f}  large-half oracle={large:.4f}')

    print('\n=== E. per-bridge curve (original pool, both datasets) ===')
    for ds in ('BUSI', 'TN3K'):
        s = summary[ds]
        vals = []
        for b in range(7):
            v = [s['metrics'][s['members']['original_per_bridge'][t][b]]['dice'] for t in s['tids']]
            vals.append(np.mean(v))
        print(f'  {ds}: ' + ' '.join(f'b{b}={v:.3f}' for b, v in enumerate(vals)))


if __name__ == '__main__':
    main()
