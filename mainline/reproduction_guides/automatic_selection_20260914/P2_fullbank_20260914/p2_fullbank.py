"""P2：Kvasir 全样板 bank（8 样板 × b0–b6）—— 拿到第二个"全深度"数据集。

复用来源（按 route_id 精确匹配 + SHA256 校验）：
  1. 原实验的 700 条/ split（auto8_tp_validation_20260911 + automatic_anchor_tp_test_20260913）
  2. P1 已跑的 b0 路线（cal_top1 / cal_top2 / raw_top2）

用法：
  python p2_fullbank.py prepare
  python p2_fullbank.py routes      # CPU，约 30-60 min
  python p2_fullbank.py seed        # 复用已有 mask
  python p2_fullbank.py propagate   # GPU
  python p2_fullbank.py score
"""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import sys, json, time, hashlib, shutil, subprocess, importlib.util, collections
import numpy as np

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E = P / 'new_project/experiments'
G = P / 'new_project/reproduction_guides/automatic_selection_20260914'
REPRO = G / 'reproduce_automatic_selection.py'
ROOT = G / 'P2_fullbank_20260914'
MODE = 'sam3enc_anchor_conditioned_target_pooling'
PY = '/home/violet/anaconda3/envs/sam3/bin/python'
BASE = '/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
CPU_ENV = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
DEPTHS = 7

SPECS = {
    'kvasir': dict(
        B=E / 'automatic_anchor_tp_test_20260913/kvasir',
        old={'validation': E / 'auto8_tp_validation_20260911',
             'test': E / 'automatic_anchor_tp_test_20260913/kvasir'},
        p1={'validation': G / 'P1_pilot_20260914/kvasir/quality_root', 'test': G / 'P1_pilot_20260914/kvasir/quality_root'},
        counts=(800, 100, 100), no_anchor_limit=None),
    'isic2018': dict(
        B=E / 'automatic_anchor_tp_test_20260913/isic2018',
        old={'validation': E / 'isic2018_auto21_tp_validation_20260911',
             'test': E / 'automatic_anchor_tp_test_20260913/isic2018'},
        p1={'validation': G / 'P1_pilot_20260914/isic2018/quality_root', 'test': G / 'P1_pilot_20260914/isic2018/quality_root'},
        counts=(2075, 259, 260), no_anchor_limit=None),
}
SPLITS = (('validation', 1), ('test', 2))


def R(ds): return ROOT / ds
def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n'); tmp.replace(p)
def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def jl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(ds, stage, **kw):
    save(R(ds) / 'status.json', dict(dataset=ds, stage=stage, time=time.time(), **kw)); print(f'[{ds}] {stage}', kw, flush=True)
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
def qpath(ds, split): return R(ds) / f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl'
def rpath(ds, split): return R(ds) / f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl'


def prepare(ds):
    r = R(ds)
    if r.exists():
        print(f'[{ds}] 已存在，跳过 prepare'); return
    subprocess.check_call([PY, str(REPRO), 'prepare', '--dataset', ds, '--out', str(r)], env=CPU_ENV)
    save(r / 'p2_policy.json', dict(
        dataset=ds, anchors='all human anchors of the automatic budget (Kvasir 8)', depths='b0-b6',
        reuse='original 700/split + P1 b0 routes, matched by route_id with sha256 verification',
        purpose='second full-depth dataset: Oracle(k) curves, calibration gain per k, budget curve',
        no_new_labels=True, model_unchanged=True))
    status(ds, 'prepared')


def _calibration(cond, records, index, aids, train):
    mu = cond[:, train].astype(np.float64).mean(1)
    cal = {}
    for t in records:
        j = index[t['merged_id']]
        centered = cond[:, j].astype(np.float64) - mu
        order = sorted(range(len(aids)), key=lambda a: (-float(centered[a]), aids[a]))
        cal[t['merged_id']] = {aids[a]: dict(anchor_target_raw=float(cond[a, j]),
                                             anchor_target_centered=float(centered[a]),
                                             anchor_train_mean=float(mu[a]),
                                             calibrated_anchor_rank=order.index(a))
                               for a in range(len(aids))}
    return mu, cal


def build_routes(ds):
    r = R(ds); tr, va, te = SPECS[ds]['counts']
    if (r / 'ROUTES_BUILT').exists():
        print(f'[{ds}] routes 已构建，跳过'); return
    g = module(f'p2_src_{ds}', r / 'code/stage1_feature_knn_routes.py')
    records = read(r / 'protocol/merged_manifest.jsonl'); support = read(r / 'protocol/support_manifest.jsonl')
    z = np.load(r / 'quality_root/features/sam3_base_s256_features.npz', allow_pickle=True)
    aids = [str(a) for a in z['anchor_ids']]; cond = z['cond_target']; pm = z['patch_mean']
    index = {x['merged_id']: i for i, x in enumerate(records)}
    train = [i for i, x in enumerate(records) if x['split'] == 'train']
    assert len(train) == tr and len(records) == cond.shape[1]
    sim = pm @ pm.T
    state = dict(mode=MODE, patch_mean=pm, sim=sim, knn_sim=sim, knn_feature='patch_mean',
                 cond_scores=cond, id_to_anchor={a: i for i, a in enumerate(aids)},
                 id_to_index=index, text_blend=0.)
    mu, cal = _calibration(cond, records, index, aids, train)
    save(r / 'calibration_frozen.json', dict(anchor_ids=aids, train_means=mu.tolist(), train_count=tr,
                                             hidden_train_GT_read=False))
    anchors = g.t21.human_pool(support, 512)
    assert {a['anchor_id'] for a in anchors} == set(aids)
    t0 = time.time()
    cache = g.build_rank_cache(state, records, support, train)
    print(f'[{ds}] rank cache 构建完成 {time.time()-t0:.0f}s', flush=True)
    audit = {}
    for split, ci in SPLITS:
        old = read(SPECS[ds]['old'][split] / f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl')
        old_by_tb = collections.defaultdict(list)
        for x in old:
            old_by_tb[(x['target_id'], int(x['bridge_count']))].append(x)
        tgts = sorted((x for x in records if x['split'] == split), key=lambda x: x['merged_id'])
        allroutes = []
        for pos, t in enumerate(tgts, 1):
            j = index[t['merged_id']]
            for a in anchors:
                aid = a['anchor_id']; forbidden = {j, index[aid]}
                beams = [([], g.route_score(state, aid, [], j))]
                for b in range(DEPTHS):
                    if b:
                        expanded = []
                        for path, _ in beams:
                            tail = j if not path else path[0]
                            nodes = g.top_ranked_nodes(cache, state, aid, tail, forbidden | set(path), 32)
                            for node in nodes:
                                new = [node, *path]
                                expanded.append((new, g.route_score(state, aid, new, j)))
                        beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:32]
                    path, sc = max(beams, key=lambda item: item[1])
                    rec = g.make_route(t, b, a, path, sc, records)
                    rec.update(cal[t['merged_id']][aid])
                    allroutes.append(rec)
            if pos % 10 == 0:
                print(f'[{ds}] {split} {pos}/{len(tgts)} targets, {len(allroutes)} routes, '
                      f'{time.time()-t0:.0f}s', flush=True)
        # 原始路线必须复现。注意：Kvasir validation 的原始路线是在 900 行缓存上算的，
        # 这里用 1000 行（含 test），pm@pm.T 的 BLAS 末位会有 ~1e-9 差异；
        # 因此 route_id 与非浮点字段要求完全一致，浮点字段用 1e-5 容差并记录实际最大差。
        byid = {x['route_id']: x for x in allroutes}
        if len(byid) != len(allroutes):
            raise SystemExit(f'[{ds}] {split} route_id 有重复')
        missing = [o['route_id'] for o in old if o['route_id'] not in byid]
        maxdiff = {}
        for oldr in old:
            new = byid[oldr['route_id']]
            for k, v in oldr.items():
                nv = new[k]
                if isinstance(v, float) and isinstance(nv, float):
                    maxdiff[k] = max(maxdiff.get(k, 0.0), abs(v - nv))
                else:
                    assert nv == v, (split, oldr['route_id'], k, v, nv)
        print(f'[{ds}] {split}: 原始 route_id 命中 {len(old)-len(missing)}/{len(old)}；'
              f'浮点字段最大差 ' + ', '.join(f'{k}={v:.2e}' for k, v in sorted(maxdiff.items())), flush=True)
        assert not missing, (split, '有原始路线无法复现', missing[:3])
        assert all(v < 1e-5 for v in maxdiff.values()), (split, maxdiff)
        assert len(allroutes) == len(tgts) * len(anchors) * DEPTHS
        assert len({x['route_id'] for x in allroutes}) == len(allroutes)
        jl(rpath(ds, split), allroutes)
        audit[split] = dict(targets=len(tgts), anchors=len(anchors), depths=DEPTHS, routes=len(allroutes),
                            original_reproduced=len(old) - len(missing),
                            float_field_max_abs_diff=maxdiff)
        print(f'[{ds}] {split}: {len(allroutes)} routes（原始 {len(old)} 条已精确复现）', flush=True)
    save(r / 'route_audit.json', audit)
    (r / 'ROUTES_BUILT').touch(); status(ds, 'routes_built', audit=audit)


def seed(ds):
    r = R(ds)
    if (r / 'SEEDED').exists():
        print(f'[{ds}] 已 seed，跳过'); return
    sources = {}
    for split, _ in SPLITS:
        for src in (SPECS[ds]['old'][split], None):
            if src is None:
                # P1 的 propagation 目录：<p1 root>/<MODE>/propagation_quality_<split>/
                src = SPECS[ds]['p1'][split] / MODE / f'propagation_quality_{split}'
            else:
                src = src / f'quality_root/{MODE}/propagation_quality_{split}'
            f = src / 'propagation_quality.jsonl'
            if not f.exists():
                continue
            for row in read(f):
                if row.get('status') != 'success':
                    continue
                sources.setdefault(row['route_id'], row)
    print(f'[{ds}] 可复用记录 {len(sources)} 条', flush=True)
    audit = {}
    for split, _ in SPLITS:
        routes = read(rpath(ds, split))
        mdir = qpath(ds, split).parent / 'forward_masks'; mdir.mkdir(parents=True, exist_ok=True)
        out = []
        for rt in routes:
            row = sources.get(rt['route_id'])
            if row is None:
                continue
            src = Path(row['forward_mask_path']); src = src if src.is_absolute() else P / src
            assert sha(src) == row['forward_mask_sha256'], rt['route_id']
            dst = mdir / f"{rt['route_id']}.png"
            if not dst.exists():
                shutil.copy2(src, dst)
            row = dict(row); row['forward_mask_path'] = str(dst); out.append(row)
        jl(qpath(ds, split), out)
        audit[split] = dict(routes=len(routes), seeded=len(out), to_propagate=len(routes) - len(out))
        print(f'[{ds}] {split}: 复用 {len(out)}，待传播 {len(routes)-len(out)}', flush=True)
    save(r / 'seed_audit.json', audit)
    (r / 'SEEDED').touch(); status(ds, 'seeded', audit=audit)


def wait_gpu():
    forced = os.environ.get('P2_FORCE_GPU')
    if forced:
        print(f'P2_FORCE_GPU={forced}：跳过显存守卫', flush=True); return int(forced)
    while True:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,uuid',
                                       '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,used_memory',
                                        '--format=csv,noheader,nounits'], text=True)
        busy = set()
        for line in apps.splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) == 2:
                try:
                    if int(parts[1]) > 1024: busy.add(parts[0])
                except ValueError: busy.add(parts[0])
        avail, mem = [], []
        for line in raw.splitlines():
            gpu, free, uid = [x.strip() for x in line.split(',')]
            mem.append(dict(gpu=int(gpu), free=int(free)))
            if int(free) >= 18000 and uid not in busy: avail.append(int(gpu))
        if avail:
            gpu = 1 if 1 in avail else avail[0]; return gpu
        print(f'等待空闲 GPU：{mem}', flush=True); time.sleep(60)


def propagate(ds):
    r = R(ds); gpu = wait_gpu()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH='/Data_8TB/lht/sam3:' + str(P / 'src'), PYTHONUNBUFFERED='1')
    for split, _ in SPLITS:
        log = r / f'logs/{split}_propagation.log'
        cmd = [PY, str(r / 'code/eval_route_propagation_quality.py'), '--checkpoint', BASE,
               '--mode', MODE, '--root', str(r / 'quality_root'), '--split', split,
               '--canvas', '256', '--no-target-gt', '--resume']
        print(f'[{ds}] propagating {split} on GPU{gpu}（日志 {log}）', flush=True)
        with log.open('a') as f:
            rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(f'[{ds}] {split} 传播失败 rc={rc}，见 {log}（可直接重跑同一命令断点续传）')
        status(ds, f'propagated_{split}', gpu=gpu)


def score(ds):
    from PIL import Image
    def mb(p):
        p = Path(p); p = p if p.is_absolute() else P / p
        return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127
    def dice(a, b):
        n = int(a.sum()) + int(b.sum()); return 2 * int((a & b).sum()) / n if n else 1.
    r = R(ds); out = {}
    cal = json.loads((r / 'calibration_frozen.json').read_text())
    aids = [str(a) for a in cal['anchor_ids']]; mu = np.array(cal['train_means'], float)
    order_greedy = [x['merged_id'] for x in read(r / 'protocol/support_manifest.jsonl')]
    assert set(order_greedy) == set(aids)
    for split, _ in SPLITS:
        rows = read(qpath(ds, split)); rts = {x['route_id']: x for x in read(rpath(ds, split))}
        bad = [x['route_id'] for x in rows if x.get('status') != 'success']
        if bad:
            raise SystemExit(f'[{ds}] {split} 有 {len(bad)} 条非 success：{bad[:5]}')
        gt, per, meta = {}, collections.defaultdict(dict), collections.defaultdict(dict)
        for q in rows:
            t = q['target_id']; x = rts[q['route_id']]; a = x['anchor_id']
            if t not in gt:
                gt[t] = mb(q['target_mask_path_evaluation_only'])
            per[t][(a, int(x['bridge_count']))] = dice(mb(q['forward_mask_path']), gt[t])
            meta[t][a] = (x.get('anchor_target_raw'), x.get('anchor_target_centered'))
        rank = {}
        for t in per:
            missing = [a for a in aids if a not in meta[t]]
            assert not missing, (split, t, missing)
            raw_order = sorted(aids, key=lambda a: a, reverse=True)      # 并列 -> anchor_id 大者优先
            raw_order.sort(key=lambda a: -float(meta[t][a][0]))
            cal_order = sorted(aids, key=lambda a: (-float(meta[t][a][1]), a))
            rank[t] = dict(raw=raw_order, cal=cal_order)

        def oracle(mode, k, which):
            vals = []
            for t in per:
                pool = set(rank[t][which][:k])
                vals.append(max(per[t][(a, 0)] for a in pool) if mode == 'b0'
                            else max(v for (a, b), v in per[t].items() if a in pool))
            return float(np.mean(vals))

        ks = [k for k in (1, 2, 3, 5, 8) if k <= len(aids)]
        res = dict(n_targets=len(per), n_anchors=len(aids),
                   candidates_per_target=len(per[next(iter(per))]),
                   bias_ratio=float(np.std(mu) / np.std([max(meta[t][a][0] for a in aids) for t in per])),
                   b0={f'k{k}': dict(raw=oracle('b0', k, 'raw'), cal=oracle('b0', k, 'cal')) for k in ks},
                   full={f'k{k}': dict(raw=oracle('all', k, 'raw'), cal=oracle('all', k, 'cal')) for k in ks},
                   budget={})
        for K in (1, 2, 4, 8):
            if K <= len(aids):
                seg = set(order_greedy[:K])
                res['budget'][f'K{K}'] = float(np.mean(
                    [max(v for (a, b), v in per[t].items() if a in seg) for t in per]))
        out[split] = res
        print(f'==== {ds} / {split}（{len(per)} 目标，{res["candidates_per_target"]} 候选/目标，'
              f'偏置比 {res["bias_ratio"]:.2f}）====')
        for name, tag in (('b0', '仅 b0'), ('full', '全深度 b0-b6')):
            for k, v in res[name].items():
                print(f'  {tag} {k}: raw={v["raw"]:.4f} cal={v["cal"]:.4f} Δ={v["cal"] - v["raw"]:+.4f}')
        print('  预算曲线(贪心前缀, 全深度): ' + ', '.join(f'{k}:{v:.4f}' for k, v in res['budget'].items()))
    save(r / 'p2_results.json', out)
    return out


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    ds_list = sys.argv[2:] if len(sys.argv) > 2 else ['kvasir']
    if stage in ('prepare', 'all'):
        for ds in ds_list: prepare(ds)
    if stage in ('routes', 'all'):
        for ds in ds_list: build_routes(ds)
    if stage in ('seed', 'all'):
        for ds in ds_list: seed(ds)
    if stage in ('propagate', 'all'):
        for ds in ds_list: propagate(ds)
    if stage in ('score', 'all'):
        for ds in ds_list: score(ds)


if __name__ == '__main__':
    main()
