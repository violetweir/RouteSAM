"""TN3K 失败模式诊断（修正版）。

修正上一版两个错误：
  1. 提示框面积原来是归一化比例(0-1)，GT/预测是像素数(0-65536)，相除无意义 -> 统一转成像素。
  2. 只报均值会被极端离群值主导（TN3K 不可达目标 pred/GT 均值 1.775 但中位数只有 0.211），
     所以这里同时报中位数、分位数和超/欠分割比例。

与 BUSI 在同一口径下对比失败模式。
"""
from pathlib import Path
import json, collections
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E = P / 'new_project/experiments'
M = 'sam3enc_anchor_conditioned_target_pooling'
SP = {'BUSI': E / 'busi_calibration_factorial_20260914', 'TN3K': E / 'tn3k_busi_factorial_20260915'}
DATA_TN3K = Path('/Data_8TB/lht/data/tn3k')
DATA_BUSI = Path('/Data_8TB/lht/MK-UNet/BUSI/BUSI_split/test/masks')
PX = 256 * 256


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def bmp(p): return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127


def gt_of(ds, tid):
    a = tid.split('::')
    return DATA_BUSI / f'{a[1]}_mask.png' if ds == 'BUSI' else DATA_TN3K / 'test-mask' / f'{a[2]}.jpg'


def collect(ds):
    root = SP[ds]
    met = json.loads((root / 'test_candidate_metrics.json').read_text())
    mem = json.loads((root / 'pool_membership_frozen.json').read_text())['groups']['test']
    rows = {r['route_id']: r for r in read(root / f'quality_root/{M}/propagation_quality_test/propagation_quality.jsonl')}
    rec = []
    for t in sorted(mem['centered_top2']):
        best = max(mem['centered_top2'][t], key=lambda r: met[r]['dice'])
        row = rows[best]
        gt = bmp(gt_of(ds, t)); pred = bmp(row['forward_mask_path'])
        bx = row['anchor_box_xywh_normalized']
        rec.append(dict(t=t, anchor=row['anchor_id'], dice=met[best]['dice'],
                        gt=int(gt.sum()), pred=int(pred.sum()),
                        box=bx[2] * bx[3] * PX,
                        inter=int((pred & gt).sum()),
                        gtpix=int(gt.sum()), predpix=int(pred.sum())))
    return rec


def stats(name, rec, mask=None):
    r = [x for x in rec if (mask is None or mask(x))]
    if not r: return
    gt = np.array([x['gt'] for x in r], float); pr = np.array([x['pred'] for x in r], float)
    box = np.array([x['box'] for x in r], float); di = np.array([x['dice'] for x in r], float)
    it = np.array([x['inter'] for x in r], float)
    pg = pr / np.maximum(gt, 1)
    print('%-14s n=%4d  Dice中位=%.4f | GT中位=%6.0f 框中位=%6.0f 预测中位=%6.0f | '
          '框/GT中位=%.2f | 预测/GT 中位=%.3f  超分割(>1.5)=%4.1f%%  欠分割(<0.67)=%4.1f%%  交/GT中位=%.3f'
          % (name, len(r), np.median(di), np.median(gt), np.median(box), np.median(pr),
             np.median(box / np.maximum(gt, 1)), np.median(pg),
             100 * (pg > 1.5).mean(), 100 * (pg < 0.67).mean(), np.median(it / np.maximum(gt, 1))))


print('=== A. 失败模式：TN3K vs BUSI（同一口径，中位数为主） ===')
store = {}
for ds in ('BUSI', 'TN3K'):
    rec = collect(ds); store[ds] = rec
    print(f'--- {ds} ---')
    stats('全部', rec)
    stats('可达(>=0.5)', rec, lambda x: x['dice'] >= 0.5)
    stats('不可达(<0.5)', rec, lambda x: x['dice'] < 0.5)
    stats('  <0.2', rec, lambda x: x['dice'] < 0.2)

print('\n=== B. TN3K: 提示框大小 vs 目标 GT 大小（是否提示框过大） ===')
rec = store['TN3K']
gt = np.array([x['gt'] for x in rec], float); box = np.array([x['box'] for x in rec], float)
pr = np.array([x['pred'] for x in rec], float); di = np.array([x['dice'] for x in rec], float)
bg = box / np.maximum(gt, 1); pg = pr / np.maximum(gt, 1)
print(f'  corr(框, 预测)      = {np.corrcoef(box, pr)[0,1]:+.4f}   <- 提示框越大，预测越大')
print(f'  corr(框/GT, 预测/GT)= {np.corrcoef(bg, pg)[0,1]:+.4f}')
print(f'  corr(框/GT, Dice)   = {np.corrcoef(bg, di)[0,1]:+.4f}')
print(f'  corr(预测/GT, Dice) = {np.corrcoef(pg, di)[0,1]:+.4f}')
print(f'  框/GT 中位={np.median(bg):.3f}  均值={bg.mean():.3f}  >2 的比例={100*(bg>2).mean():.1f}%')
print('\n  按 框/GT 四分位分桶（用中位数，避免离群值）:')
qs = np.quantile(bg, [0, .25, .5, .75, 1.0])
print('  %-16s %6s %9s %9s %9s %9s' % ('框/GT 区间', 'n', 'Dice中位', '预测/GT中位', '超分割%', '欠分割%'))
for i in range(4):
    lo, hi = qs[i], qs[i + 1]
    m = (bg >= lo) & (bg <= hi if i == 3 else bg < hi)
    if m.sum() == 0: continue
    print('  %-16s %6d %9.4f %9.3f %8.1f%% %8.1f%%' % (
        f'[{lo:.2f},{hi:.2f}]', m.sum(), np.median(di[m]), np.median(pg[m]),
        100 * (pg[m] > 1.5).mean(), 100 * (pg[m] < 0.67).mean()))

print('\n=== C. TN3K 不可达目标：预测塌缩还是外溢？ ===')
bad = [x for x in rec if x['dice'] < 0.5]
pgb = np.array([x['pred'] / max(x['gt'], 1) for x in bad])
itb = np.array([x['inter'] / max(x['gt'], 1) for x in bad])
print(f'  n={len(bad)}')
for lo, hi, lab in [(0, .1, '几乎全丢(<0.1)'), (.1, .33, '严重欠分割'), (.33, .67, '轻度欠分割'),
                    (.67, 1.5, '面积接近'), (1.5, 1e9, '过分割')]:
    m = (pgb >= lo) & (pgb < hi)
    if m.sum():
        print(f'    {lab:<14} {m.sum():4d} 张 ({100*m.mean():5.1f}%)  交/GT中位={np.median(itb[m]):.3f}')
print('\n  注：交/GT 低 + 面积比低 = 只找到病灶的一小块；交/GT 低 + 面积比接近 = 找错位置。')

print('\n=== D. 预测面积是否与 GT 面积脱钩（塌缩到固定大小？） ===')
for name, m in (('可达', di >= 0.5), ('不可达', di < 0.5)):
    if m.sum() < 3: continue
    print(f'  {name}: corr(GT, 预测)={np.corrcoef(gt[m], pr[m])[0,1]:+.4f}  '
          f'预测中位={np.median(pr[m]):.0f} px  预测变异系数={pr[m].std()/pr[m].mean():.3f}')
