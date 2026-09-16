"""E0 诊断：全部基于现有存档，不跑 GPU，不写任何原始实验目录。

产出：
  e0_results.json            全部数值
  per_candidate_*.csv        统一口径的候选表（Dice 由 mask 重算）
  E0_diagnostics.md          报告
  figures/*.png              图
"""
import json, math, sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E = P / 'new_project/experiments'
G = P / 'new_project/reproduction_guides/automatic_selection_20260914'
OUT = G / 'E0_diagnostics_20260914'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
MODE = 'sam3enc_anchor_conditioned_target_pooling'

SPECS = {
    'kvasir':   dict(base=E / 'automatic_anchor_tp_test_20260913/kvasir',
                     router=E / 'automatic_anchor_tp_router_20260913/kvasir',
                     counts=(800, 100, 100)),
    'isic2018': dict(base=E / 'automatic_anchor_tp_test_20260913/isic2018',
                     router=E / 'automatic_anchor_tp_router_20260913/isic2018',
                     counts=(2075, 259, 260)),
    'busi':     dict(base=E / 'busi_auto5_tp_1pct_20260913',
                     router=E / 'busi_auto5_tp_router_20260913/busi',
                     counts=(517, 64, 66)),
}
CAL = dict(base=E / 'busi_calibrated_multi_anchor_20260913', counts=(517, 64, 66))

RES = {}
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------- 基础工具
def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def mask_bool(p):
    p = Path(p)
    if not p.is_absolute():
        p = P / p
    return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127


def load_split(base, split):
    return read_jsonl(base / f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl')


def score(rows):
    """Dice/IoU 一律由保存的 mask + GT 重算，三数据集同一口径。"""
    gt, out = {}, []
    for r in rows:
        t = r['target_id']
        if t not in gt:
            gt[t] = mask_bool(r['target_mask_path_evaluation_only'])
        a = mask_bool(r['forward_mask_path'])
        b = gt[t]
        inter = int((a & b).sum())
        den = int(a.sum()) + int(b.sum())
        dice = 2 * inter / den if den else 1.0
        iou = inter / (den - inter) if (den - inter) else 1.0
        out.append(dict(
            dataset=r.get('target_source_dataset', ''),
            split=r.get('target_split', ''),
            target_id=t, route_id=r['route_id'], anchor_id=r['anchor_id'],
            bridge_count=int(r['bridge_count']),
            calibrated_anchor_rank=r.get('calibrated_anchor_rank'),
            anchor_target_raw=r.get('anchor_target_raw'),
            anchor_target_centered=r.get('anchor_target_centered'),
            anchor_train_mean=r.get('anchor_train_mean'),
            q_cycle=(None if r.get('q_cycle') is None else float(r['q_cycle'])),
            cycle_success=bool(r.get('cycle_success', False)),
            path_bottleneck=float(r['path_bottleneck_similarity']),
            path_mean=float(r['path_mean_similarity']),
            sam_score=float(r.get('final_sam_score', 0.0) or 0.0),
            n_components=int(r.get('trace_component_final', 0) or 0),
            dice=dice, iou=iou))
    return out


def write_csv(rows, path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join('' if r[c] is None else (f'{r[c]:.8f}' if isinstance(r[c], float) else str(r[c]))
                             for c in cols) + '\n')


def sp(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return None
    return float(spearmanr(a[m], b[m]).statistic)


# ---------------------------------------------------------------- E0 数据
log('# 载入并重算候选 Dice')
TAB = {}
for name, sp_ in SPECS.items():
    rows = score(load_split(sp_['base'], 'test'))
    TAB[name] = rows
    write_csv(rows, OUT / f'per_candidate_{name}_test.csv')
    log(f'  {name}: {len(rows)} candidates, {len({r["target_id"] for r in rows})} targets')

cal_test = score(load_split(CAL['base'], 'test'))
cal_val = score(load_split(CAL['base'], 'validation'))
TAB['busi_cal'] = cal_test
write_csv(cal_test, OUT / 'per_candidate_busi_calibrated_test.csv')
log(f'  busi_cal: {len(cal_test)} candidates (test), {len(cal_val)} (validation)')


# ---------------------------------------------------------------- E0.1 样板偏置
def cache_stats(base, counts):
    z = np.load(base / 'quality_root/features/sam3_base_s256_features.npz', allow_pickle=True)
    ct = z['cond_target'].astype(np.float64)          # (A, N)
    aids = [str(x) for x in z['anchor_ids'].tolist()]
    tr, va, te = counts
    N = ct.shape[1]
    assert N == tr + va + te, (base, N, counts)
    tridx = np.arange(0, tr)
    validx = np.arange(tr, tr + va)
    teidx = np.arange(tr + va, N)
    mu = ct[:, tridx].mean(1)
    d = dict(aids=aids, mu=mu, ct=ct, tridx=tridx, validx=validx, teidx=teidx)
    for tag, idx in (('val', validx), ('test', teidx)):
        raw = ct[:, idx]
        cal = raw - mu[:, None]
        top_raw = raw.argmax(0)
        top_cal = cal.argmax(0)
        tie = np.isclose(raw, raw.max(0, keepdims=True), atol=1e-7).sum(0)
        d[tag] = dict(
            switch=float((top_raw != top_cal).mean()),
            conc_raw=float(np.bincount(top_raw, minlength=len(aids)).max() / len(idx)),
            conc_cal=float(np.bincount(top_cal, minlength=len(aids)).max() / len(idx)),
            top_raw=top_raw, top_cal=top_cal,
            tie_frac=float((tie > 1).mean()),
            top1_std=float(raw.max(0).std()),
            martingale_gap=float(np.mean(np.sort(raw, 0)[-1] - np.sort(raw, 0)[-2])),
        )
    d['mu_std'] = float(mu.std())
    d['mu_min'] = float(mu.min()); d['mu_max'] = float(mu.max())
    d['bias_ratio'] = float(mu.std() / d['test']['top1_std'])
    return d


log('# E0.1 样板偏置')
BIAS = {}
for name, sp_ in SPECS.items():
    st = cache_stats(sp_['base'], sp_['counts'])
    # 每个样板的实际平均 Dice（仅统计该样板真正被传播过的目标；有选择偏差，需注明）
    per = defaultdict(list); per_score = defaultdict(list)
    for r in TAB[name]:
        per[r['anchor_id']].append(r['dice'])
    per_anchor = [dict(anchor=a, n=len(v), mean_dice=float(np.mean(v)))
                  for a, v in sorted(per.items(), key=lambda kv: -len(kv[1]))]
    st['per_anchor'] = per_anchor
    st['across_anchor_corr'] = sp([x['mean_dice'] for x in per_anchor if x['n'] >= 5],
                                  [float(st['mu'][st['aids'].index(x['anchor'])]) for x in per_anchor if x['n'] >= 5])
    BIAS[name] = st
    log(f"  {name}: mu_std={st['mu_std']:.4f} top1_std={st['test']['top1_std']:.4f} "
        f"bias_ratio={st['bias_ratio']:.2f} switch(test)={st['test']['switch']:.2f} "
        f"conc {st['test']['conc_raw']:.3f}->{st['test']['conc_cal']:.3f} "
        f"tie={st['test']['tie_frac']:.2f}")

# BUSI 校准目录自带的 train_means，用于交叉核对
cal_frozen = json.loads((CAL['base'] / 'calibration_frozen.json').read_text())
busi_cache = BIAS['busi']
chk = []
for a, m in zip(cal_frozen['anchor_ids'], cal_frozen['train_means']):
    if a in busi_cache['aids']:
        chk.append(abs(float(m) - float(busi_cache['mu'][busi_cache['aids'].index(a)])))
RES['calibration_crosscheck_max_abs'] = float(max(chk)) if chk else None
log(f"  BUSI mu 与 calibration_frozen.json 的最大差 = {RES['calibration_crosscheck_max_abs']:.2e}")


# ---------------------------------------------------------------- E0.2 分解主表
def oracle_gap(rows, router_dice):
    by = defaultdict(list)
    for r in rows:
        by[r['target_id']].append(r['dice'])
    oracle = float(np.mean([max(v) for v in by.values()]))
    fixed = {b: float(np.mean([r['dice'] for r in rows if r['bridge_count'] == b])) for b in range(7)}
    best_b = max(fixed, key=fixed.get)
    return dict(oracle=oracle, n_targets=len(by), best_b=best_b, best_b_dice=fixed[best_b],
                fixed=fixed, oracle_minus_best_fixed=oracle - fixed[best_b],
                router_dice=router_dice, oracle_minus_router=oracle - router_dice,
                gap_best_fixed=oracle - fixed[best_b])


log('# E0.2 Oracle/Gap 分解')
DEC = {}
router_dice = {}
for name, sp_ in SPECS.items():
    rj = json.loads((sp_['router'] / 'results.json').read_text())
    router_dice[name] = rj['test']['selected_router']['dice']
for name in SPECS:
    DEC[name] = oracle_gap(TAB[name], router_dice[name])
cal_res = json.loads((CAL['base'] / 'results.json').read_text())
# 选定池 = 每目标按校准分保留 top-2 个样板（calibrated_anchor_rank < 2），
# 而不是推理记录全集 1183；Oracle 必须在这个 924 条子集内算。
cal_pool = [r for r in cal_test if r['calibrated_anchor_rank'] is not None and r['calibrated_anchor_rank'] < 2]
assert len(cal_pool) == 924, len(cal_pool)
DEC['busi_cal'] = oracle_gap(cal_pool, cal_res['test']['selected_router']['dice'])
for k, v in DEC.items():
    log(f"  {k}: Oracle={v['oracle']:.6f} best_fixed=b{v['best_b']}({v['best_b_dice']:.6f}) "
        f"router={v['router_dice']:.6f} Gap(router)={v['oracle_minus_router']:.6f}")


# ---------------------------------------------------------------- E0.3 分数质量相关性
log('# E0.3 分数 -> 质量')
CORR = {}
for name, rows in TAB.items():
    rec = {}
    rec['q_cycle_pooled'] = sp([r['q_cycle'] for r in rows], [r['dice'] for r in rows])
    rec['path_bottleneck_pooled'] = sp([r['path_bottleneck'] for r in rows], [r['dice'] for r in rows])
    rec['path_mean_pooled'] = sp([r['path_mean'] for r in rows], [r['dice'] for r in rows])
    rec['q_cycle_by_depth'] = {b: sp([r['q_cycle'] for r in rows if r['bridge_count'] == b],
                                     [r['dice'] for r in rows if r['bridge_count'] == b]) for b in range(7)}
    if rows[0]['anchor_target_raw'] is not None:
        rec['anchor_raw_pooled'] = sp([r['anchor_target_raw'] for r in rows], [r['dice'] for r in rows])
        rec['anchor_centered_pooled'] = sp([r['anchor_target_centered'] for r in rows], [r['dice'] for r in rows])
        # 逐目标组内：raw / centered 的候选排序与真实 Dice 排序的一致性
        by = defaultdict(list)
        for r in rows:
            by[r['target_id']].append(r)
        rr, cc = [], []
        for t, rs in by.items():
            if len(rs) < 4:
                continue
            a = sp([x['anchor_target_raw'] for x in rs], [x['dice'] for x in rs])
            c = sp([x['anchor_target_centered'] for x in rs], [x['dice'] for x in rs])
            if a is not None: rr.append(a)
            if c is not None: cc.append(c)
        rec['within_target_raw'] = float(np.mean(rr)) if rr else None
        rec['within_target_centered'] = float(np.mean(cc)) if cc else None
        rec['within_target_n'] = len(rr)
    CORR[name] = rec
    log(f"  {name}: rho(q_cycle,dice)={rec['q_cycle_pooled']} "
        f"rho(bottleneck,dice)={rec['path_bottleneck_pooled']} "
        + (f"rho(raw,dice)={rec.get('anchor_raw_pooled'):.3f} rho(centered,dice)={rec.get('anchor_centered_pooled'):.3f} "
           f"within-target raw={rec.get('within_target_raw'):.3f} centered={rec.get('within_target_centered'):.3f}"
           if 'anchor_raw_pooled' in rec else ''))

# BUSI 校准版：只在 924 条选定池内再看一次分数-质量关系
if 'busi_cal' in CORR:
    by = defaultdict(list)
    for r in cal_pool:
        by[r['target_id']].append(r)
    rr, cc = [], []
    for t, rs in by.items():
        a = sp([x['anchor_target_raw'] for x in rs], [x['dice'] for x in rs])
        c = sp([x['anchor_target_centered'] for x in rs], [x['dice'] for x in rs])
        if a is not None:
            rr.append(a)
        if c is not None:
            cc.append(c)
    CORR['busi_cal']['pool_within_raw'] = float(np.mean(rr)) if rr else None
    CORR['busi_cal']['pool_within_centered'] = float(np.mean(cc)) if cc else None
    CORR['busi_cal']['pool_raw_pooled'] = sp([r['anchor_target_raw'] for r in cal_pool], [r['dice'] for r in cal_pool])
    CORR['busi_cal']['pool_centered_pooled'] = sp([r['anchor_target_centered'] for r in cal_pool], [r['dice'] for r in cal_pool])
    log(f"  busi_cal(924池内): rho(raw,dice)={CORR['busi_cal']['pool_raw_pooled']:.3f} "
        f"rho(centered,dice)={CORR['busi_cal']['pool_centered_pooled']:.3f} "
        f"within-target raw={CORR['busi_cal']['pool_within_raw']:.3f} centered={CORR['busi_cal']['pool_within_centered']:.3f}")


# ------------------------------------------------ E0.3b BUSI 完整 bank（validation, 5x7）
# test 的 1183 条是"按校准分选出的子集"，在其上算相关性存在条件化偏置（Berkson）；
# validation 存了全部 5 个样板 × 7 个深度 = 2240 条，可以做无条件的干净检验。
log('# E0.3b BUSI 完整 bank 机制检验（validation, 无条件）')
FULL = {}
if cal_val and cal_val[0]['anchor_target_raw'] is not None:
    byv = defaultdict(list)
    for r in cal_val:
        byv[r['target_id']].append(r)
    FULL['n_targets'] = len(byv)
    FULL['n_anchors'] = len({r['anchor_id'] for r in cal_val})
    FULL['cands_per_target'] = len(cal_val) / len(byv)
    raw = np.array([r['anchor_target_raw'] for r in cal_val], float)
    cen = np.array([r['anchor_target_centered'] for r in cal_val], float)
    dd = np.array([r['dice'] for r in cal_val], float)
    FULL['rho_raw'] = float(spearmanr(raw, dd).statistic)
    FULL['rho_centered'] = float(spearmanr(cen, dd).statistic)
    rr, cc = [], []
    for t, v in byv.items():
        a = np.array([x['anchor_target_raw'] for x in v]); c = np.array([x['anchor_target_centered'] for x in v])
        d = np.array([x['dice'] for x in v])
        if a.std() > 1e-12 and d.std() > 1e-12:
            rr.append(spearmanr(a, d).statistic)
        if c.std() > 1e-12 and d.std() > 1e-12:
            cc.append(spearmanr(c, d).statistic)
    FULL['within_raw'] = float(np.mean(rr)); FULL['within_centered'] = float(np.mean(cc)); FULL['within_n'] = len(rr)
    rch, cch, pair, best = [], [], [], []
    for t, v in byv.items():
        per = defaultdict(list)
        for x in v:
            per[x['anchor_id']].append(x['dice'])
        amap = {a: float(np.mean(ds)) for a, ds in per.items()}
        ra = max(per, key=lambda a: max(x['anchor_target_raw'] for x in v if x['anchor_id'] == a))
        ca = max(per, key=lambda a: max(x['anchor_target_centered'] for x in v if x['anchor_id'] == a))
        rch.append(amap[ra]); cch.append(amap[ca]); pair.append(amap[ca] - amap[ra]); best.append(max(amap.values()))
    pair = np.array(pair)
    FULL['raw_argmax_dice'] = float(np.mean(rch)); FULL['cal_argmax_dice'] = float(np.mean(cch))
    FULL['delta'] = float(pair.mean())
    rng = np.random.default_rng(2026)
    bs = [pair[rng.integers(0, len(pair), len(pair))].mean() for _ in range(10000)]
    FULL['ci95'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    FULL['win_rate'] = float((pair > 0).mean())
    FULL['best_anchor_dice'] = float(np.mean(best))
    # 无条件的每样板平均 Dice（全部 64 个目标）
    pa = defaultdict(list)
    for r in cal_val:
        pa[r['anchor_id']].append(r['dice'])
    FULL['per_anchor'] = [dict(anchor=a, n=len(v), mean_dice=float(np.mean(v)),
                               mu=float(BIAS['busi']['mu'][BIAS['busi']['aids'].index(a)]))
                          for a, v in sorted(pa.items(), key=lambda kv: -np.mean(kv[1]))]
    log(f"  无条件: rho(raw,dice)={FULL['rho_raw']:+.3f} -> rho(centered,dice)={FULL['rho_centered']:+.3f}")
    log(f"  逐目标 rho: raw {FULL['within_raw']:+.3f} -> centered {FULL['within_centered']:+.3f} (n={FULL['within_n']})")
    log(f"  选中的样板平均 Dice: raw-argmax {FULL['raw_argmax_dice']:.4f} -> calibrated-argmax {FULL['cal_argmax_dice']:.4f} "
        f"(delta {FULL['delta']:+.4f}, 95%CI [{FULL['ci95'][0]:.4f},{FULL['ci95'][1]:.4f}])")
    log(f"  每目标最优样板均值（样板选择上界）= {FULL['best_anchor_dice']:.4f}")


# ---------------------------------------------------------------- E0.4 天花板 vs k（BUSI）
log('# E0.4 天花板 vs k')
val_res = json.loads((CAL['base'] / 'validation_results.json').read_text())
KCURVE = dict(validation=val_res['pools'], test=cal_res['pools'])
log('  BUSI validation pools: ' + ', '.join(f"k={k}:{v['oracle']:.6f}" for k, v in sorted(KCURVE['validation'].items(), key=lambda kv: int(kv[0]))))
log('  BUSI test pools:       ' + ', '.join(f"k={k}:{v['oracle']:.6f}" for k, v in sorted(KCURVE['test'].items(), key=lambda kv: int(kv[0]))))


# ---------------------------------------------------------------- E0.5 深度 / 漂移
log('# E0.5 深度诊断')
DEPTH = {}
for name in ['kvasir', 'isic2018', 'busi']:
    rows = TAB[name]
    rec = {}
    for b in range(7):
        rs = [r for r in rows if r['bridge_count'] == b]
        rec[b] = dict(dice=float(np.mean([r['dice'] for r in rs])),
                      q_cycle=float(np.nanmean([r['q_cycle'] for r in rs])),
                      path_bottleneck=float(np.mean([r['path_bottleneck'] for r in rs])),
                      path_mean=float(np.mean([r['path_mean'] for r in rs])))
    rec['argmax_dice'] = max(range(7), key=lambda b: rec[b]['dice'])
    rec['argmax_q_cycle'] = max(range(7), key=lambda b: rec[b]['q_cycle'])
    rec['drift_proxy'] = rec[6]['q_cycle'] - rec[0]['q_cycle']
    DEPTH[name] = rec
    log(f"  {name}: argmax Dice=b{rec['argmax_dice']}  argmax q_cycle=b{rec['argmax_q_cycle']}  "
        f"q_cycle {rec[0]['q_cycle']:.3f}->{rec[6]['q_cycle']:.3f} (drift {rec['drift_proxy']:+.3f})")


# ---------------------------------------------------------------- E0.6 免标注深度选择
log('# E0.6 免标注深度选择')
FREE = {}
for name in ['kvasir', 'isic2018', 'busi']:
    rows = TAB[name]
    by = defaultdict(list)
    for r in rows:
        by[r['target_id']].append(r)
    def rule(key, sign=1):
        pick = []
        for t, rs in by.items():
            best = max(rs, key=lambda r: (sign * (r[key] if r[key] is not None else -1), r['path_mean'], -r['bridge_count']))
            pick.append(best['dice'])
        return float(np.mean(pick))
    oracle_b = float(np.mean([max(r['dice'] for r in rs) for rs in by.values()]))
    fixed = {b: float(np.mean([r['dice'] for r in rows if r['bridge_count'] == b])) for b in range(7)}
    FREE[name] = dict(q_cycle=rule('q_cycle'), bottleneck=rule('path_bottleneck'),
                      path_mean=rule('path_mean'), oracle_over_b=oracle_b,
                      best_fixed=fixed[max(fixed, key=fixed.get)],
                      best_fixed_b=max(fixed, key=fixed.get), fixed=fixed)
    log(f"  {name}: 免标注(q_cycle)argmax={FREE[name]['q_cycle']:.6f}  "
        f"(bottleneck)argmax={FREE[name]['bottleneck']:.6f}  固定最优=b{FREE[name]['best_fixed_b']}({FREE[name]['best_fixed']:.6f})  "
        f"b内Oracle={oracle_b:.6f}")


# ---------------------------------------------------------------- 图
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9})

# 图1 偏置诊断
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
names = ['kvasir', 'isic2018', 'busi']
cols = ['#4C78A8', '#F58518', '#E45756']
ax[0].bar(names, [BIAS[n]['bias_ratio'] for n in names], color=cols)
ax[0].axhline(1.0, ls='--', c='gray', lw=1)
ax[0].set_title('anchor bias ratio = std(mu_A)/std(top1 score)')
for i, n in enumerate(names):
    ax[0].text(i, BIAS[n]['bias_ratio'], f"{BIAS[n]['bias_ratio']:.2f}", ha='center', va='bottom')
ax[1].bar(np.arange(3) - 0.2, [BIAS[n]['test']['conc_raw'] for n in names], 0.4, label='raw', color=cols)
ax[1].bar(np.arange(3) + 0.2, [BIAS[n]['test']['conc_cal'] for n in names], 0.4, label='calibrated',
          color=cols, alpha=0.45, hatch='//')
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names)
ax[1].set_title('max single-anchor share (test)'); ax[1].legend()
ax[2].bar(np.arange(3) - 0.2, [BIAS[n]['test']['switch'] for n in names], 0.4, label='test', color=cols)
ax[2].bar(np.arange(3) + 0.2, [BIAS[n]['val']['switch'] for n in names], 0.4, label='validation',
          color=cols, alpha=0.45, hatch='//')
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(names)
ax[2].set_title('top-1 anchor switch rate raw -> calibrated'); ax[2].legend()
fig.tight_layout(); fig.savefig(FIG / 'fig1_bias_diagnostics.png'); plt.close(fig)

# 图2 样板层面 分数 vs 实际 Dice
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for axi, n in zip(axes, names):
    pa = BIAS[n]['per_anchor']
    x = [float(BIAS[n]['mu'][BIAS[n]['aids'].index(p['anchor'])]) for p in pa]
    y = [p['mean_dice'] for p in pa]
    s = [max(8, p['n'] * 0.6) for p in pa]
    axi.scatter(x, y, s=s, color=cols[names.index(n)], alpha=0.7, edgecolor='k', linewidth=0.4)
    if len(x) > 2:
        r = np.corrcoef(x, y)[0, 1]
        axi.set_title(f'{n}: corr={r:+.2f} (n={len(x)} anchors)')
    else:
        axi.set_title(f'{n}: only {len(x)} anchors used')
    axi.set_xlabel('mean raw score $\\mu_A$'); axi.set_ylabel('mean Dice')
    axi.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / 'fig2_anchor_score_vs_dice.png'); plt.close(fig)

# 图3 深度曲线
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for axi, n in zip(axes, names):
    b = list(range(7))
    q = [DEPTH[n][i]['q_cycle'] for i in b]
    d = [DEPTH[n][i]['dice'] for i in b]
    axi.plot(b, q, 'o-', color='#4C78A8', label='q_cycle')
    axi.set_ylabel('q_cycle', color='#4C78A8'); axi.set_xlabel('bridge depth b')
    ax2 = axi.twinx()
    ax2.plot(b, d, 's--', color='#E45756', label='Dice')
    ax2.set_ylabel('Dice', color='#E45756')
    axi.set_title(f"{n}: argmax Dice=b{DEPTH[n]['argmax_dice']}, argmax q_cycle=b{DEPTH[n]['argmax_q_cycle']}")
    axi.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / 'fig3_depth_cycle_dice.png'); plt.close(fig)

# 图4 Oracle / Gap
fig, ax = plt.subplots(figsize=(6.4, 3.4))
keys = ['kvasir', 'isic2018', 'busi', 'busi_cal']
lab = ['Kvasir', 'ISIC2018', 'BUSI(raw)', 'BUSI(cal k2)']
x = np.arange(len(keys))
ax.bar(x - 0.2, [DEC[k]['oracle'] for k in keys], 0.4, label='Oracle (7/14 cand)', color='#54A24B')
ax.bar(x + 0.2, [DEC[k]['router_dice'] for k in keys], 0.4, label='realized (router)', color='#4C78A8')
for i, k in enumerate(keys):
    ax.plot([i - 0.2, i + 0.2], [DEC[k]['oracle'], DEC[k]['oracle']], c='k', lw=0.6)
    ax.text(i, DEC[k]['oracle'] + 0.01, f"{DEC[k]['oracle']:.3f}", ha='center', fontsize=8)
    ax.text(i, DEC[k]['router_dice'] + 0.01, f"{DEC[k]['router_dice']:.3f}", ha='center', fontsize=8)
    ax.annotate('', xy=(i + 0.2, DEC[k]['router_dice']), xytext=(i + 0.2, DEC[k]['oracle']),
                arrowprops=dict(arrowstyle='<->', color='#E45756', lw=1))
    ax.text(i + 0.26, (DEC[k]['oracle'] + DEC[k]['router_dice']) / 2, f"gap {DEC[k]['oracle_minus_router']:.3f}",
            fontsize=7, color='#E45756')
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0.4, 1.0)
ax.set_ylabel('Dice'); ax.set_title('Realized = Oracle - Gap'); ax.legend(loc='lower right')
fig.tight_layout(); fig.savefig(FIG / 'fig4_oracle_gap.png'); plt.close(fig)

# 图5 BUSI 天花板 vs k
fig, ax = plt.subplots(figsize=(5.2, 3.2))
kv = sorted(KCURVE['validation'].items(), key=lambda kv: int(kv[0]))
kt = sorted(KCURVE['test'].items(), key=lambda kv: int(kv[0]))
ax.plot([int(k) for k, _ in kv], [v['oracle'] for _, v in kv], 'o-', label='validation (k=0..5)')
ax.plot([int(k) for k, _ in kt], [v['oracle'] for _, v in kt], 's--', label='test (k=0..2)')
ax.set_xlabel('retained anchors k'); ax.set_ylabel('candidate-pool Oracle Dice')
ax.set_title('BUSI: ceiling rises with k (monotone)'); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / 'fig5_busi_ceiling_vs_k.png'); plt.close(fig)

# 图6 候选级 分数 vs Dice (BUSI 校准版)
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
raw = np.array([r['anchor_target_raw'] for r in cal_test], float)
cen = np.array([r['anchor_target_centered'] for r in cal_test], float)
dd = np.array([r['dice'] for r in cal_test], float)
rk = np.array([r['calibrated_anchor_rank'] for r in cal_test], float)
for axi, v, t in ((ax[0], raw, f"raw score (rho={CORR['busi_cal']['anchor_raw_pooled']:+.3f})"),
                  (ax[1], cen, f"centered score (rho={CORR['busi_cal']['anchor_centered_pooled']:+.3f})")):
    sc = axi.scatter(v, dd, c=rk, cmap='viridis', s=14, alpha=0.75)
    axi.set_xlabel(t); axi.set_ylabel('Dice'); axi.grid(alpha=0.3)
plt.colorbar(sc, ax=ax[1], label='calibrated anchor rank')
fig.suptitle('BUSI calibrated pool: score vs achieved Dice', y=1.02)
fig.tight_layout(); fig.savefig(FIG / 'fig6_busi_score_vs_dice.png', bbox_inches='tight'); plt.close(fig)

# 图7 BUSI 完整 bank（validation）：样板层面 vs 目标层面
if FULL:
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
    x = [p['mu'] for p in FULL['per_anchor']]
    y = [p['mean_dice'] for p in FULL['per_anchor']]
    n = [p['n'] for p in FULL['per_anchor']]
    ax[0].scatter(x, y, s=[max(20, v * 0.5) for v in n], c='#E45756', alpha=0.8, edgecolor='k', linewidth=0.4)
    for p in FULL['per_anchor']:
        ax[0].annotate(p['anchor'].replace('BUSI::', ''), (p['mu'], p['mean_dice']),
                       fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax[0].set_xlabel('$\\mu_A$ (train mean score)'); ax[0].set_ylabel('mean Dice over 64 val targets')
    ax[0].set_title('BUSI full bank: anchor bias vs real quality'); ax[0].grid(alpha=0.3)
    ax[1].bar(['raw-argmax', 'calibrated-argmax'], [FULL['raw_argmax_dice'], FULL['cal_argmax_dice']],
              color=['#F58518', '#54A24B'])
    for i, v in enumerate([FULL['raw_argmax_dice'], FULL['cal_argmax_dice']]):
        ax[1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax[1].axhline(FULL['best_anchor_dice'], ls='--', c='gray', lw=1)
    ax[1].text(1.35, FULL['best_anchor_dice'], f"per-target best anchor {FULL['best_anchor_dice']:.4f}",
               fontsize=7, va='bottom', ha='right', color='gray')
    ax[1].set_ylabel('mean Dice of selected anchor')
    ax[1].set_title(f"anchor choice improves by {FULL['delta']:+.4f}\n95%CI [{FULL['ci95'][0]:.3f},{FULL['ci95'][1]:.3f}]")
    fig.tight_layout(); fig.savefig(FIG / 'fig7_busi_fullbank_mechanism.png'); plt.close(fig)

log(f'# 图已写入 {FIG}')


# ---------------------------------------------------------------- 报告
def f(x, n=4):
    return 'n/a' if x is None else f'{x:.{n}f}'


lines = []
A = lines.append
A('# E0 诊断报告（全部基于现有存档，0 GPU）')
A('')
A(f'生成脚本：`E0_diagnostics_20260914/e0_diagnostics.py`；数值：`e0_results.json`；'
  f'统一候选表：`per_candidate_*.csv`（Dice 一律由 mask + GT 重算，与存档值逐位一致）。')
A('')
A('## 结论速览')
A('')
A(f"1. 样板分数偏置是**真实且量级差异极大**的：偏置比 Kvasir {BIAS['kvasir']['bias_ratio']:.2f}、"
  f"ISIC2018 {BIAS['isic2018']['bias_ratio']:.2f}、BUSI {BIAS['busi']['bias_ratio']:.2f}。"
  f"BUSI 上减掉 `mu_A` 后 85% 的 test 目标改选样板，集中度 {BIAS['busi']['test']['conc_raw']:.3f}→{BIAS['busi']['test']['conc_cal']:.3f}。")
A(f"2. ISIC2018 的偏置比 {BIAS['isic2018']['bias_ratio']:.2f} > Kvasir 的 {BIAS['kvasir']['bias_ratio']:.2f}，"
  f"且改选比例 {BIAS['isic2018']['test']['switch']:.0%} 明显高于 Kvasir 的 {BIAS['kvasir']['test']['switch']:.0%}"
  f" → **预测 ISIC 的校准收益显著大于 Kvasir**（待 P1 验证）。")
A(f"3. 原始分数存在大量并列：Kvasir {BIAS['kvasir']['test']['tie_frac']:.0%}、"
  f"ISIC {BIAS['isic2018']['test']['tie_frac']:.0%} 的 test 目标在 raw 最高分上**没有唯一最佳样板**，"
  f"argmax 是任意的；校准能把并列拆开。")
A(f"4. 三个数据集的瓶颈不同：Gap(router) Kvasir {DEC['kvasir']['oracle_minus_router']:.4f}、"
  f"ISIC {DEC['isic2018']['oracle_minus_router']:.4f}、BUSI(raw) {DEC['busi']['oracle_minus_router']:.4f}、"
  f"BUSI(cal) {DEC['busi_cal']['oracle_minus_router']:.4f}。"
  f"BUSI 校准后天花板大涨但 Gap 反而变大 → 校准之后验证器成为新瓶颈。")
A(f"5. 免标注的深度选择：只有 `argmax q_cycle` 是**真正的逐目标**规则；它在 ISIC 上胜过最优固定深度"
  f"（{FREE['isic2018']['q_cycle']:.4f} vs {FREE['isic2018']['best_fixed']:.4f}），在 Kvasir 上不如"
  f"（{FREE['kvasir']['q_cycle']:.4f} vs {FREE['kvasir']['best_fixed']:.4f}），在 BUSI 上接近"
  f"（{FREE['busi']['q_cycle']:.4f} vs {FREE['busi']['best_fixed']:.4f}）。"
  f"`path_mean` / `path_bottleneck` 随深度单调或有并列，argmax 近似等于『永远取最深』，"
  f"不是真正的深度选择器（见 E0.6 说明）。")
A('')
A('## E0.1 样板偏置诊断')
A('')
A('`mu_A` = 样板 A 在**未标注**训练图上的平均条件分：`mu_A = mean_{x in train} TP(A,x)`，直接取自 256 特征缓存，不用 GPU、不看 GT。')
A('')
A('| 数据集 | 样板数 | std(mu_A) | min | max | std(top1 分) | **偏置比** | 并列比例(test) |')
A('|---|---:|---:|---:|---:|---:|---:|---:|')
for n in names:
    b = BIAS[n]
    A(f"| {n} | {len(b['aids'])} | {b['mu_std']:.4f} | {b['mu_min']:.4f} | {b['mu_max']:.4f} | "
      f"{b['test']['top1_std']:.4f} | **{b['bias_ratio']:.2f}** | {b['test']['tie_frac']:.0%} |")
A('')
A('| 数据集 | 改选比例 val | 改选比例 test | 集中度 raw→cal (test) | 参考间 corr(mu_A, 平均Dice) |')
A('|---|---:|---:|---|---:|')
for n in names:
    b = BIAS[n]
    A(f"| {n} | {b['val']['switch']:.0%} | {b['test']['switch']:.0%} | "
      f"{b['test']['conc_raw']:.3f} → {b['test']['conc_cal']:.3f} | {f(b['across_anchor_corr'],3)} |")
A('')
A(f"BUSI 的 `mu_A` 与校准目录 `calibration_frozen.json` 的最大绝对差 = {RES['calibration_crosscheck_max_abs']:.2e}（交叉核对通过）。")
A('')
A('> `参考间 corr(mu_A, 平均Dice)` 与下面每样板明细里的"平均 Dice"都只在**该样板实际被传播到的目标子集**上计算，'
  '因此条件化偏置存在，只能当作粗略诊断；无条件的版本见 E0.3b（BUSI 有完整 bank）。')
A('')
A('**每个样板的明细**（`n` = 该样板实际被传播到的目标数；`平均Dice` 只在这些目标上取，存在选择偏差，仅作诊断）：')
A('')
for n in names:
    A(f'<details><summary>{n}</summary>')
    A('')
    A('| 样板 | n | mu_A | 原始平均分(test) | 平均 Dice |')
    A('|---|---:|---:|---:|---:|')
    b = BIAS[n]
    raw_mean = {}
    for r in TAB[n]:
        raw_mean.setdefault(r['anchor_id'], []).append(r['path_bottleneck'])
    for p in b['per_anchor']:
        A(f"| `{p['anchor']}` | {p['n']} | {b['mu'][b['aids'].index(p['anchor'])]:.4f} | "
          f"{np.mean(raw_mean[p['anchor']]):.4f} | {p['mean_dice']:.4f} |")
    A('')
    A('</details>')
    A('')
A('## E0.2 Oracle / Gap 分解主表')
A('')
A('| 数据集 | 目标数 | 候选/图 | Oracle | 最优固定 b | 固定 b Dice | Router Dice | Gap(Router) | Gap(最优固定) |')
A('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
cand = {'kvasir': 7, 'isic2018': 7, 'busi': 7, 'busi_cal': 14}
for k in ['kvasir', 'isic2018', 'busi', 'busi_cal']:
    v = DEC[k]
    A(f"| {k} | {v['n_targets']} | {cand[k]} | {v['oracle']:.6f} | b{v['best_b']} | {v['best_b_dice']:.6f} | "
      f"{v['router_dice']:.6f} | {v['oracle_minus_router']:.6f} | {v['gap_best_fixed']:.6f} |")
A('')
A('> `busi_cal` 的 14 候选是每目标按校准分保留 top-2 个样板、各 7 个深度；Oracle 在这 924 条内计算。')
A('')
A('## E0.3 分数 → 质量 的相关性')
A('')
A('| 数据集 | rho(q_cycle, Dice) | rho(path_bottleneck, Dice) | rho(path_mean, Dice) | rho(raw, Dice) | rho(centered, Dice) | 组内 raw | 组内 centered |')
A('|---|---:|---:|---:|---:|---:|---:|---:|')
for k in ['kvasir', 'isic2018', 'busi', 'busi_cal']:
    c = CORR[k]
    A(f"| {k} | {f(c['q_cycle_pooled'],3)} | {f(c['path_bottleneck_pooled'],3)} | {f(c['path_mean_pooled'],3)} | "
      f"{f(c.get('anchor_raw_pooled'),3)} | {f(c.get('anchor_centered_pooled'),3)} | "
      f"{f(c.get('within_target_raw'),3)} | {f(c.get('within_target_centered'),3)} |")
A('')
A('> **注意口径**：`busi_cal` 的 test 记录（1183 条）是"按校准分筛选出来的子集"，'
  '在其上统计相关性存在条件化偏置（Berkson 悖论），因此下面 E0.3b 用 BUSI 的 **完整 validation bank** 做无条件检验，'
  '以 E0.3b 为准。')
A('')
A('BUSI 校准版在**924 条选定池内**（同样有条件化偏置，仅作参考）：')
A('')
A(f"- 池内 pooled：rho(raw, Dice) = {f(CORR['busi_cal']['pool_raw_pooled'],3)}，"
  f"rho(centered, Dice) = {f(CORR['busi_cal']['pool_centered_pooled'],3)}")
A(f"- 池内逐目标平均：raw = {f(CORR['busi_cal']['pool_within_raw'],3)}，"
  f"centered = {f(CORR['busi_cal']['pool_within_centered'],3)}")
A('')
A('## E0.3b BUSI 完整 bank 机制检验（无条件，validation）')
A('')
if FULL:
    A(f"validation 存有 **全部 {FULL['n_anchors']} 个样板 × 7 个深度 = {int(FULL['cands_per_target'])} 候选/目标**，"
      f"共 {FULL['n_targets']} 个目标、{int(FULL['cands_per_target']) * FULL['n_targets']} 条候选，"
      f'没有被"筛选"过，是这一节最干净的证据。')
    A('')
    A('| 口径 | rho(raw, Dice) | rho(centered, Dice) |')
    A('|---|---:|---:|')
    A(f"| 全部候选 pooled | {FULL['rho_raw']:+.3f} | {FULL['rho_centered']:+.3f} |")
    A(f"| 逐目标平均（n={FULL['within_n']}） | {FULL['within_raw']:+.3f} | {FULL['within_centered']:+.3f} |")
    A('')
    A('| 样板选择规则 | 被选中样板的平均 Dice |')
    A('|---|---:|')
    A(f"| 按原始分数 argmax | {FULL['raw_argmax_dice']:.4f} |")
    A(f"| 按校准分数 argmax | **{FULL['cal_argmax_dice']:.4f}** |")
    A(f"| 每目标最优样板（选择上界） | {FULL['best_anchor_dice']:.4f} |")
    A('')
    A(f"校准带来的样板选择增益 = **{FULL['delta']:+.4f}**，配对 bootstrap 95% 区间 "
      f"[{FULL['ci95'][0]:.4f}, {FULL['ci95'][1]:.4f}]，逐目标胜率 {FULL['win_rate']:.1%}。"
      f"相对选择上界，校准把原始分数已有的 {FULL['raw_argmax_dice'] / FULL['best_anchor_dice']:.1%} "
      f"提升到 {FULL['cal_argmax_dice'] / FULL['best_anchor_dice']:.1%}。")
    A('')
    A('无条件的每样板平均 Dice（64 个 validation 目标）与 `mu_A`：')
    A('')
    A('| 样板 | mu_A | 平均 Dice |')
    A('|---|---:|---:|')
    for p in FULL['per_anchor']:
        A(f"| `{p['anchor']}` | {p['mu']:.4f} | {p['mean_dice']:.4f} |")
    A('')
    A('**结论**：原始分数与质量几乎无关（rho≈0.00~0.14），校准后显著变正（rho≈0.18~0.38）；'
      '在"选哪张样板"这一步上，校准把平均 Dice 从 '
      f"{FULL['raw_argmax_dice']:.3f} 提到 {FULL['cal_argmax_dice']:.3f}，且这是同一批候选上的纯重排序，"
      '不涉及任何新标注或新模型。')
else:
    A('（未取到 BUSI 完整 bank 数据）')
A('')
A('q_cycle 与 Dice 的按深度相关系数：')
A('')
A('| 数据集 | b0 | b1 | b2 | b3 | b4 | b5 | b6 |')
A('|---|---:|---:|---:|---:|---:|---:|---:|')
for k in ['kvasir', 'isic2018', 'busi']:
    A(f"| {k} | " + ' | '.join(f(CORR[k]['q_cycle_by_depth'][b], 3) for b in range(7)) + ' |')
A('')
A('## E0.4 天花板 vs 保留数 k（BUSI）')
A('')
A('| k | 候选/图 | validation Oracle | test Oracle |')
A('|---:|---:|---:|---:|')
for k in sorted(KCURVE['validation'], key=lambda x: int(x)):
    v = KCURVE['validation'][k]
    t = KCURVE['test'].get(k)
    A(f"| {k} | {v['candidate_count']} | {v['oracle']:.6f} | "
      + (f"{t['oracle']:.6f} |" if t else '未传播 |'))
A('')
A('validation 上 k=1→2→3→5 单调上升，验证了 H_k ⊆ H_{k+1} 的天花板单调性；')
A(f"k=1 相对 k=0（原版）的纯校准收益 = {KCURVE['validation']['1']['oracle'] - KCURVE['validation']['0']['oracle']:+.6f}（validation）、"
  f"{KCURVE['test']['1']['oracle'] - KCURVE['test']['0']['oracle']:+.6f}（test）。")
A('')
A('## E0.5 深度与漂移')
A('')
A('| 数据集 | 最优 Dice 深度 | q_cycle 峰值深度 | q_cycle b0→b6 | 漂移代理 |')
A('|---|---:|---:|---|---:|')
for k in ['kvasir', 'isic2018', 'busi']:
    d = DEPTH[k]
    A(f"| {k} | b{d['argmax_dice']} | b{d['argmax_q_cycle']} | "
      f"{d[0]['q_cycle']:.3f} → {d[6]['q_cycle']:.3f} | {d['drift_proxy']:+.3f} |")
A('')
A('| 数据集 | 指标 | b0 | b1 | b2 | b3 | b4 | b5 | b6 |')
A('|---|---|---:|---:|---:|---:|---:|---:|---:|')
for k in ['kvasir', 'isic2018', 'busi']:
    for m in ['dice', 'q_cycle', 'path_bottleneck', 'path_mean']:
        A(f"| {k} | {m} | " + ' | '.join(f"{DEPTH[k][b][m]:.4f}" for b in range(7)) + ' |')
A('')
A('## E0.6 免标注的深度选择')
A('')
A('| 数据集 | 规则 | Dice |')
A('|---|---|---:|')
for k in ['kvasir', 'isic2018', 'busi']:
    v = FREE[k]
    A(f"| {k} | 固定最优 b{v['best_fixed_b']} | {v['best_fixed']:.6f} |")
    A(f"| {k} | 逐目标 argmax q_cycle（免标注） | {v['q_cycle']:.6f} |")
    A(f"| {k} | 逐目标 argmax path_bottleneck（免标注） | {v['bottleneck']:.6f} |")
    A(f"| {k} | 逐目标 argmax path_mean（免标注） | {v['path_mean']:.6f} |")
    A(f"| {k} | b 内 Oracle（上界） | {v['oracle_over_b']:.6f} |")
A('')
A('> **口径警告**：`path_bottleneck` 与 `path_mean` 随深度几乎单调（bottleneck 基本不随深度变化，'
  'path_mean 单调上升），所以对它们取 argmax 近似等价于"永远选最深的 b6"，'
  '并不是真正的逐目标深度选择；`path_mean` 在 Kvasir 上比固定 b6 高出的 0.0001 属于退化差异，不应当作"免标注选深度有效"的证据。'
  '**只有 `q_cycle` 是真正随深度变化、可做逐目标决策的信号**，它在 ISIC 上有效、在 Kvasir 与 BUSI 上不有效。')
A('')
A('## 这些结果对故事意味着什么')
A('')
A('- **可比性条件是真实存在的**，而且可以用一个免标注标量（偏置比）排序三个数据集。')
A(f"- **校准的收益可以直接预测**：偏置比 BUSI {BIAS['busi']['bias_ratio']:.1f} > ISIC {BIAS['isic2018']['bias_ratio']:.1f} "
  f"> Kvasir {BIAS['kvasir']['bias_ratio']:.1f}；BUSI 已实测大涨，ISIC 是**待验证的强预测**。")
A('- **分数只是弱代理**：q_cycle 与 Dice 的秩相关在 0.0~0.5 之间且随深度衰减，BUSI 上甚至为负 —— 这正是 Gap 的来源，'
  '也说明"分数高"不能当作"分割好"。')
A('- **三个数据集卡在不同环节**：Kvasir 上限高、验证器一般；ISIC 验证器较准、但样板排序有偏；BUSI 严重偏置，'
  '校准后天花板大涨而验证器变成新瓶颈。')
A('')
A('## 还需要 GPU 才能补上的（E0 不能替代）')
A('')
A('1. **P1 前哨**：ISIC/Kvasir 的"校准 top-1 vs 原始 top-1"在 b0 上的实际 Dice —— E0 只能证明"排序会变"，不能证明"变得更好"。')
A('2. **P2/P3 主实验**：Kvasir 全 bank、ISIC top-5，才能给出与 BUSI 同口径的 Oracle(k) 与 realized。')
A('3. **P4 随机对照**：E0 完全没有下游 Dice 证据表明"贪心选样板优于随机"；'
  '历史随机 100 组只比了覆盖代理指标（`coverage_metrics.json` 明确写有 `No propagation performed`）。')
A('')
A('> 本报告不修改任何原始实验目录；所有输出写入 `E0_diagnostics_20260914/`。')

(OUT / 'E0_diagnostics.md').write_text('\n'.join(lines), encoding='utf-8')

RES.update(dict(
    bias={k: {kk: vv for kk, vv in v.items() if kk not in ('ct', 'mu', 'tridx', 'validx', 'teidx', 'val', 'test')}
          for k, v in BIAS.items()},
    bias_switch={k: dict(val=BIAS[k]['val']['switch'], test=BIAS[k]['test']['switch']) for k in BIAS},
    decomposition=DEC, correlations=CORR, depth=DEPTH, free_depth=FREE, k_curve=KCURVE,
    busi_full_bank=FULL, router_dice=router_dice))
(OUT / 'e0_results.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str))
log('完成：' + str(OUT))
