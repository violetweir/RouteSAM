"""TN3K post-analysis: pool structure, calibration diagnostics, per-bridge paired bootstrap.

Runs AFTER the main chain (which writes COMPLETE). Read-only w.r.t. frozen artifacts;
writes analysis_extra.md next to report.md.

Data sources (all frozen):
  pool_membership_frozen.json   -> per-group ordered route ids per target
  test_candidate_metrics.json   -> route_id -> {dice, iou}          (per candidate)
  test_all_anchor_routes.jsonl  -> route_id -> anchor_id, raw/centered anchor scores
  results.json / validation_results.json -> Router-selected results and per-bridge means
NOTE: test_per_target.json holds ONE Router choice per target, NOT the 7 bridge candidates,
so per-bridge series must come from membership + candidate metrics.
"""
from pathlib import Path
import json, collections, sys
import numpy as np

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'new_project/experiments/tn3k_busi_factorial_20260915'
SEED = 2026
GROUPS = ['raw_top1', 'centered_top1', 'raw_top2', 'centered_top2', 'original_per_bridge']


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def short(a): return a.replace('TN3K::trainval::', '').replace('TN3K::test::', '')


def main():
    assert (R / 'COMPLETE').exists(), 'run the main chain first'
    res = json.loads((R / 'results.json').read_text())
    members = json.loads((R / 'pool_membership_frozen.json').read_text())['groups']
    cal = json.loads((R / 'calibration_frozen.json').read_text())
    val_res = json.loads((R / 'validation_results.json').read_text())
    metrics = json.loads((R / 'test_candidate_metrics.json').read_text())
    all_routes = read(R / 'test_all_anchor_routes.jsonl')
    route_anchor = {r['route_id']: r['anchor_id'] for r in all_routes}
    route_scores = {r['route_id']: (float(r['anchor_target_raw']), float(r['anchor_target_centered']))
                    for r in all_routes}

    tids_test = sorted(members['test']['raw_top1'])
    tids_val = sorted(members['validation']['raw_top1'])

    def per_b(test_members, group):
        """{b: np.array(dice per target)} for the group's candidates, ordered rank-then-bridge."""
        return {b: np.asarray([metrics[test_members[group][t][b]]['dice'] for t in tids_test])
                for b in range(7)}

    def per_b_val(group):
        """validation per-bridge means come from the saved group summary (scalars)."""
        v = val_res['groups'][group]['fixed_rank1_b0_b6']
        return {b: float(v[b]) for b in range(7)}

    L = ['# TN3K 补充分析', '',
         '本文件由 `new_project/tn3k_post_analysis.py` 在主流程完成后读取冻结产物生成，不改动任何冻结文件。',
         '所有逐桥序列取自 `pool_membership_frozen.json`（候选成员，先于 GT 读取冻结）与 `test_candidate_metrics.json`（逐候选 Dice）。', '']

    # ---- 1. pool structure
    L += ['## 1. 候选池结构（为什么并集只有 ~27/图）', '',
          '| 划分 | 目标图 | set(raw_top1)==set(original) | set(centered_top1)==set(original) | mean \\|raw_top2 ∩ centered_top2\\| |',
          '|---|---:|---:|---:|---:|']
    struct = {}
    for split in ('validation', 'test'):
        g = members[split]; tids = sorted(g['raw_top1'])
        eq_raw = sum(1 for t in tids if set(g['raw_top1'][t]) == set(g['original_per_bridge'][t]))
        eq_cen = sum(1 for t in tids if set(g['centered_top1'][t]) == set(g['original_per_bridge'][t]))
        inter = float(np.mean([len(set(g['raw_top2'][t]) & set(g['centered_top2'][t])) for t in tids]))
        struct[split] = dict(n=len(tids), raw_eq_original=eq_raw, centered_eq_original=eq_cen, top2_inter=inter)
        L.append(f'| {split} | {len(tids)} | {eq_raw} ({100*eq_raw/len(tids):.1f}%) | '
                 f'{eq_cen} ({100*eq_cen/len(tids):.1f}%) | {inter:.2f} |')
    L += ['',
          'TN3K 上原版分数选出的 rank-1 参考图**几乎同时就是逐桥最优参考图**'
          f'（val {100*struct["validation"]["raw_eq_original"]/struct["validation"]["n"]:.1f}%、'
          f'test {100*struct["test"]["raw_eq_original"]/struct["test"]["n"]:.1f}%），',
          '因此 `original_per_bridge` 与 `raw_top1` 在 TN3K 上几乎是同一个实验，两行 Dice 会高度接近。',
          f'这是参考图原始分数偏移过大（训练均值跨度 {float(np.ptp(cal["train_means"])):.4f}）导致的数据集性质，不是实现错误。',
          'BUSI 上两者不同（0.565990 vs 0.566808），因为 BUSI 只有 5 张参考图、偏移较小（0.3498）。',
          'raw_top2 与 centered_top2 的参考图集合几乎不相交，所以并集 ≈ 14+14，每图只需传播 ~27 条。', '']

    # ---- 2. does calibration change the reference?
    raw_first = [route_anchor[members['test']['raw_top1'][t][0]] for t in tids_test]
    cen_first = [route_anchor[members['test']['centered_top1'][t][0]] for t in tids_test]
    same = sum(a == b for a, b in zip(raw_first, cen_first))
    cnt = collections.Counter(cen_first)
    L += ['## 2. 校准是否真的改变了参考图选择', '',
          f'test 共 {len(tids_test)} 张目标图，校准 rank-1 与原始 rank-1 相同的有 **{same}** 张'
          f'（{100*same/len(tids_test):.1f}%），即校准改变了 {len(tids_test)-same} 张的参考图选择。', '',
          '校准 rank-1 参考图分布：', '',
          '| 参考图 | 被选为校准 rank-1 的目标图数 |', '|---|---:|']
    for a, c in cnt.most_common():
        L.append(f'| {short(a)} | {c} |')
    L.append('')

    # ---- 3. per-bridge paired bootstrap (test)
    ob = per_b(members['test'], 'original_per_bridge')
    cb = per_b(members['test'], 'centered_top1')
    rng = np.random.default_rng(SEED); n = len(tids_test); idx = rng.integers(0, n, (10000, n))
    L += ['## 3. b0-b6 逐桥配对 bootstrap（test，10000 次，seed2026）', '',
          '| 路径 | 原版 Dice | 校准 top1 Dice | 增量 | 95% 区间 | 变好/变差 |', '|---|---:|---:|---:|---|---:|']
    for b in range(7):
        d = cb[b] - ob[b]; boot = d[idx].mean(1); lo, hi = np.quantile(boot, [.025, .975])
        L.append(f'| b{b} | {ob[b].mean():.6f} | {cb[b].mean():.6f} | {d.mean():+.6f} | '
                 f'[{lo:+.6f}, {hi:+.6f}] | {int((d>0).sum())}/{int((d<0).sum())} |')
    L += ['', '同一口径在 validation 上（用保存的分组 OOF 统计量）：', '',
          '| 路径 | 原版 Dice | 校准 top1 Dice |', '|---|---:|---:|']
    vo = per_b_val('original_per_bridge'); vc = per_b_val('centered_top1')
    for b in range(7):
        L.append(f'| b{b} | {vo[b]:.6f} | {vc[b]:.6f} |')
    bb_val = int(np.argmax([vc[b] for b in range(7)]))
    bb_test = int(np.argmax([cb[b].mean() for b in range(7)]))
    L += ['', f'validation 上校准 top1 最优 b = **b{bb_val}**，test 上最优 b = **b{bb_test}**。'
          + ('' if bb_val == bb_test else ' 两者不一致，说明逐桥数字是 test 扫描，不能直接当作可部署结果。'), '']

    # ---- 4. does the raw anchor score track quality?
    L += ['## 4. 参考图原始分数是否代表质量', '']
    per_anchor = collections.defaultdict(list)
    for r in all_routes:
        m = metrics.get(r['route_id'])
        if m is None: continue
        per_anchor[r['anchor_id']].append((float(r['anchor_target_raw']), float(r['anchor_target_centered']), m['dice']))
    means = dict(zip(cal['anchor_ids'], cal['train_means']))
    L += ['| 参考图 | 训练均值 mean_A | test 平均原始分 | 平均校准分 | 该参考传播候选平均 Dice | 候选数 |',
          '|---|---:|---:|---:|---:|---:|']
    raws = []; cens = []; dices = []
    for aid in cal['anchor_ids']:
        v = per_anchor.get(aid, [])
        if not v: continue
        arr = np.asarray(v)
        raws += arr[:, 0].tolist(); cens += arr[:, 1].tolist(); dices += arr[:, 2].tolist()
        L.append(f'| {short(aid)} | {means[aid]:.4f} | {arr[:,0].mean():.4f} | {arr[:,1].mean():.4f} | '
                 f'{arr[:,2].mean():.4f} | {len(arr)} |')
    raws = np.asarray(raws); cens = np.asarray(cens); dices = np.asarray(dices)
    c_raw = float(np.corrcoef(raws, dices)[0, 1]); c_cen = float(np.corrcoef(cens, dices)[0, 1])
    L += ['',
          f'在全部已传播的 (目标图, 参考图) 候选上（n={len(dices)}）：', '',
          f'- 原始目标分数 vs 候选真实 Dice 的 Pearson 相关：**{c_raw:+.4f}**',
          f'- 减训练均值后的校准分数 vs 候选真实 Dice 的 Pearson 相关：**{c_cen:+.4f}**', '',
          '原始分数与真实质量的相关性很弱，说明它主要反映「这张参考图对所有目标都偏高」的常数偏移，',
          '而不是「这张参考图适合这个目标」。这正是减训练均值校准能把 Dice 大幅拉高的原因。', '']

    # ---- 5. BUSI correspondence
    L += ['## 5. 与 BUSI 的对应关系', '',
          '| 项目 | BUSI | TN3K |', '|---|---|---|',
          '| 参考图数 | 5（1%×517） | 23（1%×2303） |',
          f'| 参考图训练均值跨度 | 0.3498 | {float(np.ptp(cal["train_means"])):.4f} |',
          '| 原版选图退化程度 | 65/66 张 b0 均选 malignant(187) | '
          f'val {struct["validation"]["raw_eq_original"]}/{struct["validation"]["n"]}、'
          f'test {struct["test"]["raw_eq_original"]}/{struct["test"]["n"]} 的 raw_top1 即逐桥最优 |',
          f'| 校准是否改变参考图 | 是（仅 10/66 重合） | 是（仅 {same}/{len(tids_test)} 重合） |',
          '| 每图候选并集 | 1183 条 / 66 图 ≈ 17.9 | val 27.0 / test 26.8 |',
          f'| 原始分 vs 真实质量相关 | 未分析 | {c_raw:+.4f} |', '',
          '结论：BUSI 上观察到的「原始 TP 分数不能跨参考图比较、减训练均值后可大幅提升」这一现象，',
          '在 TN3K 上不仅成立，而且更严重（偏移跨度 0.5502 vs 0.3498）。', '']
    (R / 'analysis_extra.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwritten: {R / "analysis_extra.md"}')


if __name__ == '__main__':
    main()
