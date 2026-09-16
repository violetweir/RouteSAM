"""P1 前哨实验：只做 b0（无中转站），比较三种样板选择规则的直接分割质量。

  baseline   : 原始实验实际使用的样板（= raw 分数胜出者）
  cal_top1   : 按校准分 (TP - mu_A) 排第一的样板
  cal_top2   : 按校准分排第二的样板

每个 (目标, 样板) 只跑一条 b0 路线；已有的 baseline b0 mask 直接复用并校验哈希，
因此 GPU 只用在"新增的样板-目标对"上。

用法：
  python p1_pilot.py prepare     # 复制冻结代码/协议/特征 + 写 preregistration.json
  python p1_pilot.py routes      # CPU 生成 b0 路线并复用旧 mask
  python p1_pilot.py propagate   # 带显存守卫，逐 split 跑 SAM3
  python p1_pilot.py score       # 统计 + 配对 bootstrap
  python p1_pilot.py all
"""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import sys, json, time, hashlib, shutil, subprocess, importlib.util, collections
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E = P / 'new_project/experiments'
G = P / 'new_project/reproduction_guides/automatic_selection_20260914'
REPRO = G / 'reproduce_automatic_selection.py'
ROOT = G / 'P1_pilot_20260914'
MODE = 'sam3enc_anchor_conditioned_target_pooling'
PY = '/home/violet/anaconda3/envs/sam3/bin/python'
BASE = '/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
CPU_ENV = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')

SPECS = {
    'kvasir':   dict(B=E / 'automatic_anchor_tp_test_20260913/kvasir', counts=(800, 100, 100)),
    'isic2018': dict(B=E / 'automatic_anchor_tp_test_20260913/isic2018', counts=(2075, 259, 260)),
}
# 历史实验里 validation 的传播在单独的目录（test 目录只追加了 test 行）
OLDDIR = {
    'kvasir':   {'validation': E / 'auto8_tp_validation_20260911',
                 'test': E / 'automatic_anchor_tp_test_20260913/kvasir'},
    'isic2018': {'validation': E / 'isic2018_auto21_tp_validation_20260911',
                 'test': E / 'automatic_anchor_tp_test_20260913/isic2018'},
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
def mask(p):
    p = Path(p); p = p if p.is_absolute() else P / p
    return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127
def dice(a, b):
    n = int(a.sum()) + int(b.sum()); return 2 * int((a & b).sum()) / n if n else 1.
def qpath(ds, split): return R(ds) / f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl'
def rpath(ds, split): return R(ds) / f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl'


# ------------------------------------------------------------------ prepare
def prepare(ds):
    r = R(ds)
    if r.exists():
        print(f'[{ds}] 已存在 {r}，跳过 prepare（不覆盖旧输出）'); return
    subprocess.check_call([PY, str(REPRO), 'prepare', '--dataset', ds, '--out', str(r)], env=CPU_ENV)
    # 预测先行登记：跑 GPU 之前就把预测写下来
    z = np.load(r / 'quality_root/features/sam3_base_s256_features.npz', allow_pickle=True)
    cond = z['cond_target'].astype(np.float64); aids = [str(a) for a in z['anchor_ids']]
    man = read(r / 'protocol/merged_manifest.jsonl')
    tr, va, te = SPECS[ds]['counts']
    mu = cond[:, :tr].mean(1)
    teidx = np.arange(tr + va, tr + va + te)
    raw = cond[:, teidx]
    switch = float((raw.argmax(0) != (raw - mu[:, None]).argmax(0)).mean())
    prereg = dict(
        dataset=ds, created=time.time(),
        hypothesis='the raw anchor-conditioned score is not comparable across anchors; '
                   'subtracting the unlabeled-train mean mu_A picks better anchors for each target',
        bias_ratio=float(mu.std() / raw.max(0).std()),
        mu_std=float(mu.std()), top1_score_std=float(raw.max(0).std()),
        top1_switch_rate_test=switch,
        prediction=dict(
            cal_top1_b0_dice='> baseline b0 dice on test',
            cal_top2_b0_oracle='>= cal_top1',
            kvasir_vs_isic='gain on ISIC2018 larger than on Kvasir (bias_ratio 1.10 vs 0.68)'),
        frozen_before_gpu=True, new_labels_added=0, model_changed=False,
        note='only b0 (no intermediate bridges); baseline b0 masks are reused with sha256 verification')
    save(r / 'preregistration.json', prereg)
    status(ds, 'prepared', bias_ratio=prereg['bias_ratio'], switch=switch)


# ------------------------------------------------------------------ routes
def build_routes(ds):
    r = R(ds); B = SPECS[ds]['B']; tr, va, te = SPECS[ds]['counts']
    if (r / 'ROUTES_BUILT').exists():
        print(f'[{ds}] routes 已构建，跳过'); return
    g = module(f'p1_src_{ds}', r / 'code/stage1_feature_knn_routes.py')
    records = read(r / 'protocol/merged_manifest.jsonl'); support = read(r / 'protocol/support_manifest.jsonl')
    z = np.load(r / 'quality_root/features/sam3_base_s256_features.npz', allow_pickle=True)
    aids = [str(a) for a in z['anchor_ids']]; cond = z['cond_target']; pm = z['patch_mean']
    assert cond.shape[0] == len(aids) and len(records) == cond.shape[1]
    sim = pm @ pm.T
    state = dict(mode=MODE, patch_mean=pm, sim=sim, knn_sim=sim, knn_feature='patch_mean',
                 cond_scores=cond, id_to_anchor={a: i for i, a in enumerate(aids)}, text_blend=0.)
    index = {x['merged_id']: i for i, x in enumerate(records)}; state['id_to_index'] = index
    train = [i for i, x in enumerate(records) if x['split'] == 'train']
    assert len(train) == tr, (len(train), tr)
    means = cond[:, train].astype(np.float64).mean(1)          # 只用 train RGB，不看 GT
    anchors = g.t21.human_pool(support, 512)
    by_id = {a['anchor_id']: a for a in anchors}
    assert set(by_id) == set(aids)
    rank = {}
    for x in records:
        j = index[x['merged_id']]
        centered = cond[:, j].astype(np.float64) - means
        rank[x['merged_id']] = sorted(range(len(aids)), key=lambda a: (-float(centered[a]), aids[a]))
    audit = {}
    roles = {}
    for split, ci in SPLITS:
        OD = OLDDIR[ds][split]
        old = read(OD / f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl')
        old_b0 = {x['target_id']: x for x in old if int(x['bridge_count']) == 0}
        assert len(old_b0) == SPECS[ds]['counts'][ci]
        tgts = sorted((x for x in records if x['split'] == split), key=lambda x: x['merged_id'])
        routes = []
        roles[split] = {}
        for t in tgts:
            j = index[t['merged_id']]; order = rank[t['merged_id']]
            needed = {old_b0[t['merged_id']]['anchor_id'], aids[order[0]], aids[order[1]]}
            for aid in sorted(needed):
                ai = aids.index(aid); a = by_id[aid]
                sc = g.route_score(state, aid, [], j)
                rec = g.make_route(t, 0, a, [], sc, records)
                rec.update(anchor_target_raw=float(cond[ai, j]),
                           anchor_target_centered=float(cond[ai, j] - means[ai]),
                           anchor_train_mean=float(means[ai]),
                           calibrated_anchor_rank=order.index(ai))
                routes.append(rec)
            roles[split][t['merged_id']] = dict(baseline=old_b0[t['merged_id']]['anchor_id'],
                                                cal_top1=aids[order[0]], cal_top2=aids[order[1]])
        byid = {x['route_id']: x for x in routes}
        # 原始 b0 路线必须被逐字段精确复现（验证 route_id / box / 分数口径一致）
        for tid, oldrec in old_b0.items():
            new = byid[oldrec['route_id']]
            for k, v in oldrec.items():
                assert new[k] == v, (split, tid, k, v, new[k])
        # 旧的 baseline mask 直接复用（校验哈希），避免重复占 GPU
        oldq = read(OD / f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl')
        mdir = qpath(ds, split).parent / 'forward_masks'; mdir.mkdir(parents=True, exist_ok=True)
        seed = []
        for q in oldq:
            if int(q['bridge_count']) != 0:
                continue
            src = Path(q['forward_mask_path']); src = src if src.is_absolute() else P / src
            assert sha(src) == q['forward_mask_sha256']
            dst = mdir / f"{q['route_id']}.png"; shutil.copy2(src, dst)
            q = dict(q); q['forward_mask_path'] = str(dst); seed.append(q)
        assert len(seed) == len(old_b0)
        jl(rpath(ds, split), routes)
        jl(qpath(ds, split), seed)
        need_new = len(routes) - len(seed)
        audit[split] = dict(targets=len(tgts), routes=len(routes), reused_baseline=len(seed),
                            new_to_propagate=need_new,
                            mean_needed_anchors=len(routes) / len(tgts))
        print(f'[{ds}] {split}: {len(tgts)} targets, {len(routes)} b0 routes, '
              f'{len(seed)} reused, {need_new} new', flush=True)
    save(r / 'b0_roles.json', roles)
    save(r / 'route_audit.json', audit)
    (r / 'ROUTES_BUILT').touch(); status(ds, 'routes_built', audit=audit)


# ------------------------------------------------------------------ propagate
def wait_gpu():
    forced = os.environ.get('P1_FORCE_GPU')
    if forced:
        print(f'P1_FORCE_GPU={forced}：按用户指示跳过显存守卫，直接使用 GPU{forced}', flush=True)
        return int(forced)
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
                    if int(parts[1]) > 1024:
                        busy.add(parts[0])
                except ValueError:
                    busy.add(parts[0])
        avail, mem = [], []
        for line in raw.splitlines():
            gpu, free, uid = [x.strip() for x in line.split(',')]
            mem.append(dict(gpu=int(gpu), free=int(free)))
            if int(free) >= 18000 and uid not in busy:
                avail.append(int(gpu))
        if avail:
            gpu = 1 if 1 in avail else avail[0]
            print(f'GPU {gpu} 可用（>=18000MiB 空闲且无 >1GiB 计算进程）', flush=True)
            return gpu
        print(f'等待空闲 GPU：{mem}（不终止任何他人任务）', flush=True)
        time.sleep(60)


def propagate(ds):
    r = R(ds)
    gpu = wait_gpu()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH='/Data_8TB/lht/sam3:' + str(P / 'src'), PYTHONUNBUFFERED='1')
    for split, _ in SPLITS:
        log = r / f'logs/{split}_propagation.log'
        cmd = [PY, str(r / 'code/eval_route_propagation_quality.py'), '--checkpoint', BASE,
               '--mode', MODE, '--root', str(r / 'quality_root'), '--split', split,
               '--canvas', '256', '--no-target-gt', '--resume']
        print(f'[{ds}] propagating {split} on GPU{gpu}', flush=True)
        with log.open('a') as f:
            rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(f'[{ds}] {split} 传播失败 rc={rc}，见 {log}')
        status(ds, f'propagated_{split}', gpu=gpu)


# ------------------------------------------------------------------ score
def score(ds):
    r = R(ds); out = {}
    roles_all = json.loads((r / 'b0_roles.json').read_text())
    for split, _ in SPLITS:
        rows = read(qpath(ds, split))
        rts = {x['route_id']: x for x in read(rpath(ds, split))}
        bad = [q['route_id'] for q in rows if q.get('status') != 'success']
        if bad:
            raise SystemExit(f'[{ds}] {split} 有 {len(bad)} 条非 success：{bad[:5]}')
        gt, per = {}, collections.defaultdict(dict)
        for q in rows:
            t = q['target_id']
            if t not in gt:
                gt[t] = mask(q['target_mask_path_evaluation_only'])
            per[t][rts[q['route_id']]['anchor_id']] = dice(mask(q['forward_mask_path']), gt[t])
        roles = roles_all[split]
        base, c1, c2 = [], [], []
        for t in sorted(per):
            ro = roles[t]
            for key in ('baseline', 'cal_top1', 'cal_top2'):
                if ro[key] not in per[t]:
                    raise SystemExit(f'[{ds}] {split} {t} 缺少样板 {key}={ro[key]} 的 mask')
            base.append(per[t][ro['baseline']])
            c1.append(per[t][ro['cal_top1']])
            c2.append(per[t][ro['cal_top2']])
        base, c1, c2 = np.array(base), np.array(c1), np.array(c2)
        pair = c1 - base
        rng = np.random.default_rng(2026)
        bs = [pair[rng.integers(0, len(pair), len(pair))].mean() for _ in range(10000)]
        oracle2 = np.maximum(c1, c2)
        out[split] = dict(
            n_targets=len(base),
            baseline_b0=float(base.mean()), cal_top1_b0=float(c1.mean()), cal_top2_b0=float(c2.mean()),
            cal_top2_oracle=float(oracle2.mean()),
            delta_top1_vs_baseline=float(pair.mean()),
            delta_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            win_rate_top1=float((pair > 0).mean()),
            delta_top2oracle_vs_baseline=float((oracle2 - base).mean()),
            per_target_delta=pair.tolist())
    save(r / 'p1_results.json', out)
    return out


# ------------------------------------------------------- P1b：raw 第二名（同预算 k=2）
def _raw_order(cond, aids, j):
    """冻结 pipeline 的原始排序：分数降序，并列时 anchor_id 大者优先。"""
    order = sorted(range(len(aids)), key=lambda a: aids[a], reverse=True)
    order.sort(key=lambda a: -float(cond[a, j]))
    return order


def build_routes_b(ds):
    """补上"原始规则下的第二名样板"的 b0 路线，用于同预算 k=2 公平对照。"""
    r = R(ds); tr, va, te = SPECS[ds]['counts']
    if (r / 'ROUTES_B_BUILT').exists():
        print(f'[{ds}] P1b routes 已构建，跳过'); return
    g = module(f'p1_src_{ds}', r / 'code/stage1_feature_knn_routes.py')
    records = read(r / 'protocol/merged_manifest.jsonl'); support = read(r / 'protocol/support_manifest.jsonl')
    z = np.load(r / 'quality_root/features/sam3_base_s256_features.npz', allow_pickle=True)
    aids = [str(a) for a in z['anchor_ids']]; cond = z['cond_target']; pm = z['patch_mean']
    sim = pm @ pm.T
    state = dict(mode=MODE, patch_mean=pm, sim=sim, knn_sim=sim, knn_feature='patch_mean',
                 cond_scores=cond, id_to_anchor={a: i for i, a in enumerate(aids)}, text_blend=0.)
    index = {x['merged_id']: i for i, x in enumerate(records)}; state['id_to_index'] = index
    means = cond[:, [i for i, x in enumerate(records) if x['split'] == 'train']].astype(np.float64).mean(1)
    by_id = {a['anchor_id']: a for a in g.t21.human_pool(support, 512)}
    roles = json.loads((r / 'b0_roles.json').read_text())
    roles_b = {}
    audit = {}
    for split, ci in SPLITS:
        tgts = sorted((x for x in records if x['split'] == split), key=lambda x: x['merged_id'])
        routes = read(rpath(ds, split)); have = {x['route_id'] for x in routes}
        added = 0
        roles_b[split] = {}
        for t in tgts:
            j = index[t['merged_id']]; ro = roles[split][t['merged_id']]
            order = _raw_order(cond, aids, j)
            raw_rank1 = next(aids[a] for a in order if aids[a] != ro['baseline'])
            roles_b[split][t['merged_id']] = dict(baseline=ro['baseline'], raw_top2=raw_rank1,
                                                  cal_top1=ro['cal_top1'], cal_top2=ro['cal_top2'])
            ai = aids.index(raw_rank1)
            rec = g.make_route(t, 0, by_id[raw_rank1], [], g.route_score(state, raw_rank1, [], j), records)
            if rec['route_id'] in have:
                continue
            rec.update(anchor_target_raw=float(cond[ai, j]),
                       anchor_target_centered=float(cond[ai, j] - means[ai]),
                       anchor_train_mean=float(means[ai]), calibrated_anchor_rank=None)
            routes.append(rec); have.add(rec['route_id']); added += 1
        jl(rpath(ds, split), routes)
        audit[split] = dict(routes=len(routes), newly_added=added)
        print(f'[{ds}] {split}: 新增 raw 第二名路线 {added} 条（现共 {len(routes)}）', flush=True)
    save(r / 'b0_roles_b.json', roles_b)
    save(r / 'route_audit_b.json', audit)
    (r / 'ROUTES_B_BUILT').touch(); status(ds, 'routes_b_built', audit=audit)


def score_b(ds):
    """同预算 k=2：Oracle{raw 前二} vs Oracle{校准前二}。"""
    r = R(ds); roles_b = json.loads((r / 'b0_roles_b.json').read_text()); out = {}
    for split, _ in SPLITS:
        rows = read(qpath(ds, split)); rts = {x['route_id']: x for x in read(rpath(ds, split))}
        bad = [x['route_id'] for x in rows if x.get('status') != 'success']
        if bad:
            raise SystemExit(f'[{ds}] {split} 有 {len(bad)} 条非 success：{bad[:5]}')
        gt, per = {}, collections.defaultdict(dict)
        for q in rows:
            t = q['target_id']
            if t not in gt:
                gt[t] = mask(q['target_mask_path_evaluation_only'])
            per[t][rts[q['route_id']]['anchor_id']] = dice(mask(q['forward_mask_path']), gt[t])
        rb, rc = [], []
        for t in sorted(per):
            ro = roles_b[split][t]
            for k in ('baseline', 'raw_top2', 'cal_top1', 'cal_top2'):
                if ro[k] not in per[t]:
                    raise SystemExit(f'[{ds}] {split} {t} 缺 {k}={ro[k]} 的 mask')
            rb.append(max(per[t][ro['baseline']], per[t][ro['raw_top2']]))
            rc.append(max(per[t][ro['cal_top1']], per[t][ro['cal_top2']]))
        rb, rc = np.array(rb), np.array(rc)
        pair = rc - rb
        rng = np.random.default_rng(2026)
        bs = [pair[rng.integers(0, len(pair), len(pair))].mean() for _ in range(10000)]
        out[split] = dict(n_targets=len(rb), raw_top2_oracle=float(rb.mean()),
                          cal_top2_oracle=float(rc.mean()), delta=float(pair.mean()),
                          delta_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                          win_rate=float((pair > 0).mean()))
    save(r / 'p1b_results.json', out)
    return out


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    ds_list = sys.argv[2:] if len(sys.argv) > 2 else list(SPECS)
    if stage in ('prepare', 'all'):
        for ds in ds_list: prepare(ds)
    if stage in ('routes', 'all'):
        for ds in ds_list: build_routes(ds)
    if stage in ('propagate', 'all'):
        for ds in ds_list: propagate(ds)
    if stage in ('score', 'all'):
        for ds in ds_list:
            res = score(ds)
            print(f'==== {ds} ====')
            for split, v in res.items():
                print(f"  {split}: n={v['n_targets']} baseline={v['baseline_b0']:.4f} "
                      f"cal_top1={v['cal_top1_b0']:.4f} (delta={v['delta_top1_vs_baseline']:+.4f}, "
                      f"CI[{v['delta_ci95'][0]:+.4f},{v['delta_ci95'][1]:+.4f}], win={v['win_rate_top1']:.1%}) "
                      f"cal_top2_oracle={v['cal_top2_oracle']:.4f}")
    if stage in ('routes_b', 'all_b'):
        for ds in ds_list: build_routes_b(ds)
    if stage in ('propagate_b', 'all_b'):
        for ds in ds_list: propagate(ds)
    if stage in ('score_b', 'all_b'):
        for ds in ds_list:
            res = score_b(ds)
            print(f'==== {ds} (P1b 同预算 k=2) ====')
            for split, v in res.items():
                print(f"  {split}: n={v['n_targets']} Oracle(raw前二)={v['raw_top2_oracle']:.4f} "
                      f"Oracle(校准前二)={v['cal_top2_oracle']:.4f} "
                      f"delta={v['delta']:+.4f} CI[{v['delta_ci95'][0]:+.4f},{v['delta_ci95'][1]:+.4f}] "
                      f"win={v['win_rate']:.1%}")


if __name__ == '__main__':
    main()
