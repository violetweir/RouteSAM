"""TN3K 提示框缩放探针。

诊断：TN3K 的失败是"提示框远小于目标病灶"导致的欠分割（失败目标 框/GT 中位 0.28，
预测/GT 中位 0.211，76% 欠分割）。提示框来自参考图 GT 的 tight box，而自动选出的
参考图结节偏小，所以框系统偏小。

这里在 validation 上把提示框按固定倍率 s 绕中心放大，其余全部沿用主实验的冻结设置
（同一参考图 = 校准 rank-1、同一路径、同一 canvas256、同一评价方式）。
baseline s=1.0 直接取主实验已保存的 validation 数字，不需要重跑。

注意：这是诊断性实验。若 s 在 validation 上选，则 s 是可部署的；本脚本用 validation
选 s，不碰 test GT。
"""
from pathlib import Path
import os
import sys, json, subprocess, collections
import numpy as np

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
MAIN = P / 'new_project/experiments/tn3k_busi_factorial_20260915'
R = P / 'new_project/experiments/tn3k_boxscale_probe_20260916'
M = 'sam3enc_anchor_conditioned_target_pooling'
PY = '/home/violet/anaconda3/envs/sam3/bin/python'
BASE = '/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
SCALES = [1.25, 1.5, 2.0, 3.0]
SHARDS = 2


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def jl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')


def scale_box(box, s):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    nw, nh = min(w * s, 1.0), min(h * s, 1.0)
    nx, ny = cx - nw / 2, cy - nh / 2
    nx = max(0.0, min(nx, 1.0 - nw))
    ny = max(0.0, min(ny, 1.0 - nh))
    return [float(nx), float(ny), float(nw), float(nh)]


def build():
    R.mkdir(parents=True, exist_ok=True)
    (R / 'logs').mkdir(exist_ok=True)
    members = json.loads((MAIN / 'pool_membership_frozen.json').read_text())['groups']['validation']['centered_top1']
    qrows = {r['route_id']: r for r in read(MAIN / f'quality_root/{M}/propagation_quality_validation/propagation_quality.jsonl')}
    tids = sorted(members)
    base_rows = [qrows[members[t][0]] for t in tids]          # b0 of the calibrated rank-1 anchor
    assert all(r['bridge_count'] == 0 for r in base_rows) and len(base_rows) == 576
    counts = {}
    for shard in range(SHARDS):
        out = []
        for i, r in enumerate(base_rows):
            if i % SHARDS != shard: continue
            for s in SCALES:
                nr = dict(r)
                nr['anchor_box_xywh_normalized'] = scale_box(r['anchor_box_xywh_normalized'], s)
                nr['route_id'] = f"{r['route_id']}bs{int(round(s*100))}"
                nr['probe_scale'] = s
                out.append(nr)
        jl(R / f'shard{shard}/quality_root/{M}/validation_pool0_stage1/routes.jsonl', out)
        counts[f'shard{shard}'] = len(out)
    boxes = {s: [] for s in SCALES}
    for r in base_rows:
        for s in SCALES: boxes[s].append(scale_box(r['anchor_box_xywh_normalized'], s)[2] * scale_box(r['anchor_box_xywh_normalized'], s)[3] * 65536)
    save(R / 'probe_policy.json', dict(scales=SCALES, shards=SHARDS, counts=counts, targets=len(tids),
         baseline='s=1.0 取自主实验 validation centered_top1 的 b0', anchor='calibrated rank-1 per target',
         bridge='b0 only', box_area_px_median={str(s): float(np.median(v)) for s, v in boxes.items()}))
    print('built', counts, flush=True)


def run(shard, gpu):
    root = R / f'shard{shard}/quality_root'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH='/Data_8TB/lht/sam3:' + str(P / 'src'),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    cmd = [PY, str(MAIN / 'code/eval_route_propagation_quality.py'), '--checkpoint', BASE, '--mode', M,
           '--root', str(root), '--split', 'validation', '--canvas', '256', '--resume']
    with (R / f'logs/shard{shard}.log').open('a') as f:
        rc = subprocess.Popen(cmd, cwd=P, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT).wait()
    if rc: raise SystemExit(f'shard{shard} failed rc={rc}')
    print(f'shard{shard} done', flush=True)


def report():
    base = json.loads((MAIN / 'validation_results.json').read_text())['groups']['centered_top1']['fixed_rank1_b0_b6'][0]
    rows = []
    for shard in range(SHARDS):
        rows += read(R / f'shard{shard}/quality_root/{M}/propagation_quality_validation/propagation_quality.jsonl')
    assert len(rows) == 576 * len(SCALES), len(rows)
    # GT 来自 eval 记录的 gt_dice_evaluation_only
    by = collections.defaultdict(list)
    for r in rows:
        by[r['probe_scale']].append(float(r['gt_dice_evaluation_only']))
    print(f'\n=== validation b0 Dice vs 提示框倍率 s（n=576，同一参考图/路径） ===')
    print('%-8s %10s %12s %12s' % ('s', 'b0 Dice', '相对 s=1.0', '框面积中位px'))
    areas = json.loads((R / 'probe_policy.json').read_text())['box_area_px_median']
    print('%-8s %10.6f %12s %12s' % ('1.00', base, '—', f"{float(areas['1.25'])/1.25**2:.0f}"))
    best = (1.0, base)
    for s in SCALES:
        v = float(np.mean(by[s]))
        print('%-8s %10.6f %+12.6f %12.0f' % (f'{s:.2f}', v, v - base, float(areas[str(s)])))
        if v > best[1]: best = (s, v)
    print(f'\nvalidation 最优 s = {best[0]}  (b0 Dice {best[1]:.6f}, 相对 baseline {best[1]-base:+.6f})')
    save(R / 'probe_result.json', dict(baseline_s1=base,
         by_scale={str(s): float(np.mean(by[s])) for s in SCALES}, best_s=best[0], best_dice=best[1],
         note='diagnostic on validation only; b0; same anchor/path as main experiment'))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'build': build()
    elif a[0] == 'run': run(int(a[1]), a[2])
    elif a[0] == 'report': report()
