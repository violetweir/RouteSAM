"""P2 事后分析：Kvasir 全 bank 的 Oracle(k) 曲线、校准增益随 k 的衰减、预算曲线，全部配对 bootstrap。"""
from pathlib import Path
import json, collections
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
G = P / 'new_project/reproduction_guides/automatic_selection_20260914'
ROOT = G / 'P2_fullbank_20260914'
MODE = 'sam3enc_anchor_conditioned_target_pooling'


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def mb(p):
    p = Path(p); p = p if p.is_absolute() else P / p
    return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127
def dice(a, b):
    n = int(a.sum()) + int(b.sum()); return 2 * int((a & b).sum()) / n if n else 1.
def boot(pair, n=10000, seed=2026):
    rng = np.random.default_rng(seed)
    bs = [pair[rng.integers(0, len(pair), len(pair))].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


cal = json.loads((ROOT / 'kvasir/calibration_frozen.json').read_text())
aids = [str(a) for a in cal['anchor_ids']]
order_greedy = [x['merged_id'] for x in read(ROOT / 'kvasir/protocol/support_manifest.jsonl')]
RES = {}
for split in ('validation', 'test'):
    rows = read(ROOT / f'kvasir/quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl')
    rts = {x['route_id']: x for x in read(ROOT / f'kvasir/quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl')}
    gt, per, meta = {}, collections.defaultdict(dict), collections.defaultdict(dict)
    for q in rows:
        t = q['target_id']; x = rts[q['route_id']]; a = x['anchor_id']
        if t not in gt:
            gt[t] = mb(q['target_mask_path_evaluation_only'])
        per[t][(a, int(x['bridge_count']))] = dice(mb(q['forward_mask_path']), gt[t])
        meta[t][a] = (x['anchor_target_raw'], x['anchor_target_centered'])
    targets = sorted(per)
    rank = {}
    for t in targets:
        ro = sorted(aids, key=lambda a: a, reverse=True); ro.sort(key=lambda a: -float(meta[t][a][0]))
        rank[t] = dict(raw=ro, cal=sorted(aids, key=lambda a: (-float(meta[t][a][1]), a)))

    def vec(mode, k, which):
        out = []
        for t in targets:
            pool = set(rank[t][which][:k])
            out.append(max(per[t][(a, 0)] for a in pool) if mode == 'b0'
                       else max(v for (a, b), v in per[t].items() if a in pool))
        return np.array(out)

    def bvec(K):
        seg = set(order_greedy[:K])
        return np.array([max(v for (a, b), v in per[t].items() if a in seg) for t in targets])

    r = dict(n_targets=len(targets), candidates_per_target=56)
    for mode in ('b0', 'full'):
        r[mode] = {}
        for k in (1, 2, 3, 5, 8):
            raw, cl = vec(mode, k, 'raw'), vec(mode, k, 'cal')
            pair = cl - raw
            lo, hi = boot(pair)
            r[mode][k] = dict(raw=float(raw.mean()), cal=float(cl.mean()), delta=float(pair.mean()),
                              ci=[lo, hi], win=float((pair > 0).mean()))
    # 宽度杠杆（同一条曲线内相邻 k）
    r['width_full'] = {}
    for a, b in ((1, 2), (2, 3), (3, 5), (5, 8)):
        pair = vec('full', b, 'raw') - vec('full', a, 'raw')
        lo, hi = boot(pair)
        r['width_full'][f'{a}->{b}'] = dict(delta=float(pair.mean()), ci=[lo, hi])
    r['width_b0'] = {}
    for a, b in ((1, 2), (2, 3), (3, 5), (5, 8)):
        pair = vec('b0', b, 'raw') - vec('b0', a, 'raw')
        lo, hi = boot(pair)
        r['width_b0'][f'{a}->{b}'] = dict(delta=float(pair.mean()), ci=[lo, hi])
    # 预算曲线（贪心前缀）
    r['budget'] = {}
    base = bvec(1)
    for K in (1, 2, 4, 8):
        v = bvec(K); pair = v - base
        r['budget'][K] = dict(oracle=float(v.mean()), gain_vs_K1=float(pair.mean()), ci=boot(pair))
    RES[split] = r

    print(f'===== kvasir / {split}（{len(targets)} 目标，56 候选/目标）=====')
    print('  全深度 b0-b6：')
    for k in (1, 2, 3, 5, 8):
        d = r['full'][k]
        print(f'    k={k}: raw={d["raw"]:.4f} cal={d["cal"]:.4f} Δcal={d["delta"]:+.4f} '
              f'CI[{d["ci"][0]:+.4f},{d["ci"][1]:+.4f}] win={d["win"]:.0%}')
    print('  仅 b0：')
    for k in (1, 2, 3, 5, 8):
        d = r['b0'][k]
        print(f'    k={k}: raw={d["raw"]:.4f} cal={d["cal"]:.4f} Δcal={d["delta"]:+.4f} '
              f'CI[{d["ci"][0]:+.4f},{d["ci"][1]:+.4f}]')
    print('  宽度杠杆（raw，全深度）：' + '  '.join(
        f'{kk}:{vv["delta"]:+.4f}[{vv["ci"][0]:+.3f},{vv["ci"][1]:+.3f}]' for kk, vv in r['width_full'].items()))
    print('  宽度杠杆（raw，仅 b0）：' + '  '.join(
        f'{kk}:{vv["delta"]:+.4f}' for kk, vv in r['width_b0'].items()))
    print('  预算曲线（贪心前缀，全深度）：' + '  '.join(
        f'K{K}:{vv["oracle"]:.4f}(+{vv["gain_vs_K1"]:.4f})' for K, vv in r['budget'].items()))

(ROOT / 'p2_analysis.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1) + '\n')
print('\n写入', ROOT / 'p2_analysis.json')
