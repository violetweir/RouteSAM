"""P1 事后分析：把 b0 上的校准收益拆到"目标类型"上，并做三数据集同口径对照。

只读 P1 输出与历史 mask，不跑 GPU。
"""
from pathlib import Path
import json, collections
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E = P / 'new_project/experiments'
G = P / 'new_project/reproduction_guides/automatic_selection_20260914'
ROOT = G / 'P1_pilot_20260914'
MODE = 'sam3enc_anchor_conditioned_target_pooling'
SPECS = {
    'kvasir': dict(old={'validation': E / 'auto8_tp_validation_20260911',
                        'test': E / 'automatic_anchor_tp_test_20260913/kvasir'},
                   feat=E / 'automatic_anchor_tp_test_20260913/kvasir/quality_root/features/sam3_base_s256_features.npz',
                   counts=(800, 100, 100), bias_ratio=0.68),
    'isic2018': dict(old={'validation': E / 'isic2018_auto21_tp_validation_20260911',
                          'test': E / 'automatic_anchor_tp_test_20260913/isic2018'},
                     feat=E / 'automatic_anchor_tp_test_20260913/isic2018/quality_root/features/sam3_base_s256_features.npz',
                     counts=(2075, 259, 260), bias_ratio=1.10),
}


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def mask(p):
    p = Path(p); p = p if p.is_absolute() else P / p
    return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127
def dice(a, b):
    n = int(a.sum()) + int(b.sum()); return 2 * int((a & b).sum()) / n if n else 1.
def boot(pair, seed=2026, n=10000):
    rng = np.random.default_rng(seed)
    bs = [pair[rng.integers(0, len(pair), len(pair))].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def p1_split(ds, split):
    r = ROOT / ds
    q = read(r / f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl')
    rt = {x['route_id']: x for x in read(r / f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl')}
    roles = json.loads((r / 'b0_roles.json').read_text())[split]
    gt, per = {}, collections.defaultdict(dict)
    for row in q:
        t = row['target_id']
        if t not in gt:
            gt[t] = mask(row['target_mask_path_evaluation_only'])
        per[t][rt[row['route_id']]['anchor_id']] = dice(mask(row['forward_mask_path']), gt[t])
    # 分数边距：用缓存里的 cond_target
    z = np.load(SPECS[ds]['feat'], allow_pickle=True)
    cond = z['cond_target'].astype(np.float64); aids = [str(a) for a in z['anchor_ids']]
    man = read(r / 'protocol/merged_manifest.jsonl')
    idx = {m['merged_id']: i for i, m in enumerate(man)}
    tr, va, te = SPECS[ds]['counts']
    mu = cond[:, :tr].mean(1)
    out = []
    for t in per:
        j = idx[t]; ro = roles[t]
        s = np.sort(cond[:, j])[::-1]
        out.append(dict(target=t, dice=per[t][ro['baseline']], dice_cal=per[t][ro['cal_top1']],
                        dice_cal2=per[t][ro['cal_top2']],
                        switched=ro['baseline'] != ro['cal_top1'],
                        margin=float(s[0] - s[1]),
                        centered_margin=float((cond[:, j] - mu).max() - np.sort(cond[:, j] - mu)[-2]),
                        baseline_anchor=ro['baseline'], cal_anchor=ro['cal_top1']))
    return out


print('=' * 78)
print('P1 事后分析')
print('=' * 78)
summary = {}
for ds in SPECS:
    for split in ('validation', 'test'):
        rows = p1_split(ds, split)
        base = np.array([r['dice'] for r in rows]); cal = np.array([r['dice_cal'] for r in rows])
        cal2 = np.array([r['dice_cal2'] for r in rows]); pair = cal - base
        switched = np.array([r['switched'] for r in rows])
        margin = np.array([r['margin'] for r in rows])
        lo, hi = boot(pair)
        d = dict(n=len(rows), baseline=float(base.mean()), cal_top1=float(cal.mean()),
                 delta=float(pair.mean()), ci=[lo, hi], win=float((pair > 0).mean()),
                 switch_rate=float(switched.mean()),
                 cal_top2_oracle=float(np.maximum(cal, cal2).mean()))
        # 子群：发生了改选 vs 没改选
        if switched.any() and (~switched).any():
            d['delta_switched'] = float(pair[switched].mean())
            d['delta_kept'] = float(pair[~switched].mean())
            d['n_switched'] = int(switched.sum())
        # 子群：按 raw 边距（top1-top2 原始分数）
        med = float(np.median(margin))
        lo_g = margin <= med; hi_g = margin > med
        d['delta_low_margin'] = float(pair[lo_g].mean()); d['n_low_margin'] = int(lo_g.sum())
        d['delta_high_margin'] = float(pair[hi_g].mean()); d['n_high_margin'] = int(hi_g.sum())
        # 子群：原始分数并列
        tie = margin < 1e-9
        d['n_tie'] = int(tie.sum())
        if tie.any():
            d['delta_tie'] = float(pair[tie].mean())
        # 逐 target 分位数
        d['pctl'] = [float(x) for x in np.percentile(pair, [10, 25, 50, 75, 90])]
        d['p_catastrophic_baseline'] = float((base < 0.3).mean())
        d['p_rescue'] = float(((base < 0.3) & (cal > base)).sum() / max(1, (base < 0.3).sum())) if (base < 0.3).any() else None
        summary[f'{ds}/{split}'] = d
        print(f"\n[{ds} / {split}]  n={d['n']}  改选率={d['switch_rate']:.0%}")
        print(f"  baseline b0   = {d['baseline']:.4f}")
        print(f"  cal_top1  b0  = {d['cal_top1']:.4f}   delta={d['delta']:+.4f}  "
              f"CI[{lo:+.4f},{hi:+.4f}]  胜率={d['win']:.0%}")
        print(f"  cal_top2 Oracle(b0) = {d['cal_top2_oracle']:.4f}  (vs baseline {d['cal_top2_oracle']-d['baseline']:+.4f}，"
              f"但这是 1 候选 vs 2 候选，非同预算)")
        if 'delta_switched' in d:
            print(f"  改选的目标 ({d['n_switched']}/{d['n']}): delta={d['delta_switched']:+.4f} ; "
                  f"未改选: {d['delta_kept']:+.4f}")
        print(f"  低边距目标 (n={d['n_low_margin']}): delta={d['delta_low_margin']:+.4f} ; "
              f"高边距 (n={d['n_high_margin']}): {d['delta_high_margin']:+.4f}")
        print(f"  原始并列目标 n={d['n_tie']}" + (f": delta={d.get('delta_tie'):+.4f}" if d['n_tie'] else ''))
        print(f"  逐目标 delta 分位 p10/25/50/75/90 = " + ' '.join(f'{x:+.3f}' for x in d['pctl']))
        print(f"  baseline 灾难率(Dice<0.3)={d['p_catastrophic_baseline']:.1%}"
              + (f"，被校准救回比例={d['p_rescue']:.0%}" if d['p_rescue'] is not None else ''))

# BUSI 同口径（完整 validation bank）
z = np.load(E / 'busi_auto5_tp_1pct_20260913/quality_root/features/sam3_base_s256_features.npz', allow_pickle=True)
cond = z['cond_target'].astype(np.float64); aids = [str(a) for a in z['anchor_ids']]
caldir = E / 'busi_calibrated_multi_anchor_20260913'
man = read(caldir / 'protocol/merged_manifest.jsonl')
idx = {m['merged_id']: i for i, m in enumerate(man)}
mu = cond[:, [i for i, m in enumerate(man) if m['split'] == 'train']].mean(1)
rows = read(caldir / f'quality_root/{MODE}/propagation_quality_validation/propagation_quality.jsonl')
gt, per = {}, collections.defaultdict(dict)
for r in rows:
    t = r['target_id']
    if t not in gt:
        gt[t] = mask(r['target_mask_path_evaluation_only'])
    per[t][r['anchor_id']] = mask(r['forward_mask_path'])
base, cal, cal2, margin = [], [], [], []
for t in per:
    j = idx[t]
    raw_pick = max(aids, key=lambda a: (cond[aids.index(a), j], a))
    order = sorted(range(len(aids)), key=lambda a: (-(cond[a, j] - mu[a]), aids[a]))
    cp, c2 = aids[order[0]], aids[order[1]]
    base.append(dice(per[t][raw_pick], gt[t])); cal.append(dice(per[t][cp], gt[t]))
    cal2.append(dice(per[t][c2], gt[t]))
    s = np.sort(cond[:, j])[::-1]; margin.append(float(s[0] - s[1]))
base, cal, cal2, margin = map(np.array, (base, cal, cal2, margin))
pair = cal - base; lo, hi = boot(pair)
med = float(np.median(margin))
print(f"\n[busi / validation]  n={len(base)}  （完整 bank，与 P1 同口径的 b0）")
print(f"  baseline b0 = {base.mean():.4f}  cal_top1 b0 = {cal.mean():.4f}  delta={pair.mean():+.4f} "
      f"CI[{lo:+.4f},{hi:+.4f}]  胜率={(pair>0).mean():.0%}")
print(f"  cal_top2 Oracle(b0) = {np.maximum(cal, cal2).mean():.4f} (vs baseline {np.maximum(cal,cal2).mean()-base.mean():+.4f})")
print(f"  低边距 (n={(margin<=med).sum()}): delta={pair[margin<=med].mean():+.4f} ; "
      f"高边距 (n={(margin>med).sum()}): {pair[margin>med].mean():+.4f}")
print(f"  逐目标 delta 分位 = " + ' '.join(f'{x:+.3f}' for x in np.percentile(pair, [10, 25, 50, 75, 90])))
print(f"  baseline 灾难率(Dice<0.3)={np.mean(base<0.3):.1%}，被校准救回比例="
      f"{((base<0.3)&(cal>base)).sum()/max(1,(base<0.3).sum()):.0%}")
summary['busi/validation'] = dict(n=len(base), bias_ratio=5.42, baseline=float(base.mean()),
                                  cal_top1=float(cal.mean()), delta=float(pair.mean()), ci=[lo, hi],
                                  win=float((pair > 0).mean()),
                                  cal_top2_oracle=float(np.maximum(cal, cal2).mean()),
                                  delta_low_margin=float(pair[margin <= med].mean()),
                                  delta_high_margin=float(pair[margin > med].mean()),
                                  pctl=[float(x) for x in np.percentile(pair, [10, 25, 50, 75, 90])],
                                  p_rescue=float(((base < 0.3) & (cal > base)).sum() / max(1, (base < 0.3).sum())))
biases = {'busi/validation': 5.42, 'isic2018/validation': 1.10, 'isic2018/test': 1.10,
          'kvasir/validation': 0.68, 'kvasir/test': 0.68}
print('\n' + '=' * 78)
print('三点规律：偏置比 vs b0 上的校准增益')
print('=' * 78)
print(f"{'数据集/split':<22}{'偏置比':>8}{'baseline':>10}{'cal_top1':>10}{'delta':>10}{'95%CI':>22}{'胜率':>8}")
for k, v in summary.items():
    print(f"{k:<22}{biases.get(k, float('nan')):>8.2f}{v['baseline']:>10.4f}{v['cal_top1']:>10.4f}"
          f"{v['delta']:>+10.4f}{f'[{v[chr(99)+chr(105)][0]:+.4f},{v[chr(99)+chr(105)][1]:+.4f}]':>22}{v['win']:>8.0%}")
(ROOT / 'p1_analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
print('\n写入', ROOT / 'p1_analysis.json')
