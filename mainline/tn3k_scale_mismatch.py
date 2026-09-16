"""TN3K 根因分析：参考图与目标的结节尺度失配。

结论：corr(|log(参考结节面积/目标结节面积)|, Dice) = -0.4776。
失败组（143/614）参考/目标面积比中位 0.158，可达组 0.641。
参考比目标"大"时 Dice 仍 0.865，只有参考显著"小"于目标才崩 -> 单向失效。
自动选图只按外观做贪心覆盖，未把尺度纳入覆盖目标。
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R = P / 'new_project/experiments/tn3k_busi_factorial_20260915'
M = 'sam3enc_anchor_conditioned_target_pooling'
D = Path('/Data_8TB/lht/data/tn3k')


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def bmp(p): return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127


def main():
    met = json.loads((R / 'test_candidate_metrics.json').read_text())
    mem = json.loads((R / 'pool_membership_frozen.json').read_text())['groups']['test']
    rows = {r['route_id']: r for r in read(R / f'quality_root/{M}/propagation_quality_test/propagation_quality.jsonl')}
    cache = {}
    rec = []
    for t in sorted(mem['centered_top2']):
        gt = bmp(D / 'test-mask' / f"{t.split('::')[2]}.jpg")
        best = max(mem['centered_top2'][t], key=lambda r: met[r]['dice'])
        rr = rows[best]
        aidx = rr['anchor_id'].split('::')[2]
        if aidx not in cache:
            cache[aidx] = int(bmp(D / 'trainval-mask' / f'{aidx}.jpg').sum())
        bx = rr['anchor_box_xywh_normalized']
        rec.append(dict(dice=met[best]['dice'], ag=cache[aidx], tg=int(gt.sum()),
                        box=bx[2] * bx[3] * 65536, pred=int(bmp(rr['forward_mask_path']).sum())))
    di = np.array([x['dice'] for x in rec]); ag = np.array([x['ag'] for x in rec], float)
    tg = np.array([x['tg'] for x in rec], float); pr = np.array([x['pred'] for x in rec], float)
    lr = np.log(ag / np.maximum(tg, 1))
    pg = pr / np.maximum(tg, 1)

    print(f'n={len(rec)}')
    print(f'corr(|log(参考/目标)|, Dice) = {np.corrcoef(np.abs(lr), di)[0,1]:+.4f}')
    print(f'corr(log(参考/目标), Dice)   = {np.corrcoef(lr, di)[0,1]:+.4f}')
    qs = np.quantile(lr, [0, .25, .5, .75, 1.0])
    print('\n%-20s %6s %10s %10s' % ('log(参考/目标) 区间', 'n', 'Dice中位', '欠分割%'))
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        m = (lr >= lo) & (lr <= hi if i == 3 else lr < hi)
        print('%-20s %6d %10.4f %9.1f%%' % (f'[{lo:+.2f},{hi:+.2f}]', m.sum(),
                                            np.median(di[m]), 100 * (pg[m] < 0.67).mean()))
    print('\n%-8s %14s %12s %12s' % ('组', '参考/目标中位', '目标GT中位', '参考GT中位'))
    for name, m in (('可达', di >= 0.5), ('失败', di < 0.5)):
        print('%-8s %14.3f %12.0f %12.0f' % (name, np.median(ag[m] / tg[m]), np.median(tg[m]), np.median(ag[m])))


if __name__ == '__main__':
    main()
