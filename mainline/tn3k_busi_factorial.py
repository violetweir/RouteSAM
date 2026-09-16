"""TN3K：把 BUSI 的 PseudoVideo 方法整体复现一遍。

固定条件：TN3K 官方 fold0（train 2303 / validation 576 / test 614），
round(1% x 2303) = 23 张自动参考图，SAM3-base 原始权重，参考图 GT tight box，
无文本，canvas256，patch_mean KNN，TP 路径评分，beam32，b0-b6。

流程：prepare -> select-features -> select -> base-routes -> pool -> smoke -> propagate -> validate -> evaluate

与 BUSI 的对应关系：
  busi_auto_tp.py                        -> prepare / select-features / select / base-routes
  busi_calibrated_multi_anchor.py        -> pool（全参考池 + 减训练均值校准）
  busi_calibration_factorial.py          -> validate / evaluate（raw/centered x top1/top2 + original 对照）
"""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import sys, json, time, hashlib, shutil, subprocess, collections, importlib.util, traceback
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E = P / 'new_project/experiments'
SRC = E / 'busi_calibrated_multi_anchor_20260913'        # 冻结代码来源
R = E / 'tn3k_busi_factorial_20260915'                   # 本次实验目录
D = Path('/Data_8TB/lht/data/tn3k')                      # TN3K 原始数据
MODE = 'sam3enc_anchor_conditioned_target_pooling'
PY = '/home/violet/anaconda3/envs/sam3/bin/python'
CPU_PY = '/home/violet/anaconda3/envs/mkunet_mamba/bin/python'
BASE = '/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
SEED = 2026
COUNTS = {'train': 2303, 'validation': 576, 'test': 614}
BUDGET = 23
GROUPS = ['raw_top1', 'centered_top1', 'raw_top2', 'centered_top2', 'original_per_bridge']
PAIRS = [('centered_top1', 'raw_top1'), ('centered_top2', 'raw_top2'),
         ('raw_top2', 'raw_top1'), ('centered_top2', 'centered_top1'),
         ('centered_top2', 'original_per_bridge')]


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n'); tmp.replace(p)
def jl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
def status(stage, **kw):
    save(R / 'status.json', dict(stage=stage, time=time.time(), **kw)); print(stage, kw, flush=True)
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
def route_mod(): return module('tn3k_route_source', R / 'code/stage1_feature_knn_routes.py')
def prev_mod():
    m = module('tn3k_prev_pipeline', R / 'code/previous_pipeline.py'); m.R = R; return m
def qp(root, split): return root / f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl'
def rp(root, split): return root / f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl'
def bitmap(p): return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127
def metric(a, b):
    i = int((a & b).sum()); s = int(a.sum()) + int(b.sum())
    return dict(dice=2 * i / s if s else 1., iou=i / (s - i) if s - i else 1.)
def run(stage, cmd, env_extra=None, log_name=None):
    env = os.environ.copy()
    env.update(PYTHONPATH='/Data_8TB/lht/sam3:' + str(P / 'src'), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    if env_extra: env.update(env_extra)
    log = R / 'logs' / (log_name or f'{stage}.log')
    with log.open('a') as f:
        proc = subprocess.Popen(cmd, cwd=P, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
        save(R / f'{stage}_process.json', dict(pid=proc.pid, cmd=cmd, gpu=(env_extra or {}).get('CUDA_VISIBLE_DEVICES'), time=time.time(), log=str(log)))
        status('running_' + stage, pid=proc.pid, log=str(log))
        rc = proc.wait()
    if rc: raise RuntimeError(f'{stage} failed rc={rc}; see {log}')
    status('done_' + stage)
    return log


# ---------------------------------------------------------------- prepare
def prepare():
    R.mkdir(exist_ok=False)
    for n in ('code', 'protocol', 'logs', 'selection', 'selected/images', 'selected/masks'):
        (R / n).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, R / 'pipeline.py')
    for n in ('stage1_feature_knn_routes.py', 'eval_route_propagation_quality.py',
              'run_t21_dynamic_pseudovideo.py', 'router.py'):
        shutil.copy2(SRC / 'code' / n, R / 'code' / n)
    shutil.copy2(SRC / 'pipeline.py', R / 'code/previous_pipeline.py')

    fold = json.loads((D / 'tn3k-trainval-fold0.json').read_text())
    records = []
    for split, key in (('train', 'train'), ('validation', 'val')):
        for i in sorted(fold[key]):
            name = f'{i:04d}.jpg'
            ip = D / 'trainval-image' / name; mp = D / 'trainval-mask' / name
            records.append(dict(file_name=str(ip), image_path=str(ip), mask_file_name=str(mp), mask_path=str(mp),
                                merged_dataset='TN3K', merged_id=f'TN3K::trainval::{i:04d}',
                                sample_id=f'trainval::{i:04d}', source_dataset='TN3K', split=split,
                                width=None, height=None))
    for i in range(COUNTS['test']):
        name = f'{i:04d}.jpg'
        ip = D / 'test-image' / name; mp = D / 'test-mask' / name
        records.append(dict(file_name=str(ip), image_path=str(ip), mask_file_name=str(mp), mask_path=str(mp),
                            merged_dataset='TN3K', merged_id=f'TN3K::test::{i:04d}',
                            sample_id=f'test::{i:04d}', source_dataset='TN3K', split='test',
                            width=None, height=None))
    for r in records:
        assert Path(r['image_path']).exists(), r['image_path']
        assert Path(r['mask_path']).exists(), r['mask_path']
    assert len({r['merged_id'] for r in records}) == len(records), 'merged_id collision'
    got = dict(collections.Counter(r['split'] for r in records))
    assert got == COUNTS, got
    # TN3K 的 test/0000-0613 与 trainval/0000-2878 文件名重叠，必须靠 split 前缀区分
    base = collections.Counter(Path(r['image_path']).name for r in records)
    assert max(base.values()) == 2, 'expected exactly the test/trainval name overlap'
    jl(R / 'protocol/merged_manifest.jsonl', records)

    train_rows = [dict(id=r['merged_id'], image_path=r['image_path'], split='train')
                  for r in records if r['split'] == 'train']
    assert len(train_rows) == COUNTS['train']
    save(R / 'train_images_only.json', train_rows)

    hashes = collections.defaultdict(list)
    for r in records: hashes[sha(r['image_path'])].append(dict(id=r['merged_id'], split=r['split']))
    dup_in = [g for g in hashes.values() if len(g) > 1]
    dup_cross = [g for g in dup_in if len({x['split'] for x in g}) > 1]
    save(R / 'dataset_audit.json', dict(split_counts=got, image_mask_pairs_complete=True,
                                        exact_duplicate_groups=dup_in, cross_split_duplicate_groups=dup_cross,
                                        name_collision_pairs=sum(1 for v in base.values() if v == 2),
                                        merged_id_namespaced_by_split=True))

    src = E / 'isic2018_auto21_tp_validation_20260911' / 'code/isic_auto_extract.py'
    text = src.read_text()
    assert text.count("os.environ['CUDA_VISIBLE_DEVICES']='0'") == 1
    text = text.replace("os.environ['CUDA_VISIBLE_DEVICES']='0'",
                        "os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')")
    assert text.count('len(rows)==2075') == 1
    text = text.replace('len(rows)==2075', f"len(rows)=={COUNTS['train']}")
    (R / 'code/selection_extract.py').write_text(text)

    save(R / 'predefined_policy.json', dict(
        dataset='TN3K', counts=COUNTS, anchors=BUDGET, seed=SEED,
        split='official tn3k-trainval-fold0.json train/val + official test 614',
        budget=f'round(1% x {COUNTS["train"]}) = {BUDGET}, same rule as BUSI 5/517, Kvasir 8/800, ISIC2018 21/2075',
        selection='train-only RGB1008 SAM3-base; patch-mean global descriptor + 64 local visual words; '
                  '64-cluster MiniBatchKMeans(random_state=2026,n_init=3,batch_size=2048,max_iter=100); '
                  'IDF on train; sqrt + L2; 0.5*global + 0.5*local cosine; greedy facility, numpy argmax tie-break',
        calibration='cond_target(anchor,target) minus mean over all train RGBs; no hidden train GT read',
        groups=GROUPS,
        ranking='raw TP or centered TP descending; anchor ID ascending breaks ties',
        pool='top1/top2 references fixed per target for all b0-b6; original per-bridge control',
        router='all five pools refit legacy28 Ridge(alpha=1); same image-grouped 5-fold validation; no new hyperparameter search',
        propagation='SAM3-base, anchor GT tight box, no text, canvas256, forward + return, no target GT during inference',
        evaluation='GT nearest resize 256 >127; per-image Dice/IoU macro mean; oracle = mean per-image max candidate Dice',
        test_policy='freeze all five models and test choices before reading test GT; no test-driven winner selection',
        prior_test_seen=False, validation_labels_extra_to_1percent=True,
        gpu='serial per split; validation and test may run on different free GPUs concurrently'))
    save(R / 'code_hashes.json', {str(p): sha(p) for p in (R / 'code').glob('*.py')})
    (R / 'PREPARED').touch()
    status('prepared', counts=got, budget=BUDGET)


# ---------------------------------------------------------------- selection
def select_features(gpu='0'):
    if (R / 'selection_features.npz').exists():
        status('select_features_cached'); return
    run('selection_features', [PY, str(R / 'code/selection_extract.py')],
        env_extra={'CUDA_VISIBLE_DEVICES': str(gpu)})


def select():
    if (R / 'selection/SELECTIONS_FROZEN.json').exists():
        status('select_cached'); return
    from sklearn.cluster import MiniBatchKMeans
    rows = json.loads((R / 'train_images_only.json').read_text())
    z = np.load(R / 'selection_features.npz')
    assert z['ids'].tolist() == [r['id'] for r in rows]
    def norm(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    n = len(rows)
    g = norm(z['global_features'].astype(np.float32))
    sample = norm(z['sampled_patches'].astype(np.float32))
    assert g.shape == (n, 1024) and sample.shape == (n, 64, 1024), (g.shape, sample.shape)
    manual = [sha(r['image_path']) for r in rows]
    seen = set(); eligible = []
    for i, h in enumerate(manual):
        if h not in seen: eligible.append(i); seen.add(h)
    km = MiniBatchKMeans(n_clusters=64, random_state=SEED, n_init=3, batch_size=2048, max_iter=100).fit(sample.reshape(-1, 1024))
    assignment = km.predict(sample.reshape(-1, 1024)).reshape(n, 64)
    h = np.stack([np.bincount(x, minlength=64) for x in assignment]).astype(np.float32) / 64
    idf = np.log((n + 1) / (1 + (h > 0).sum(0))) + 1
    h = norm(np.sqrt(h * idf))
    similarity = (np.clip(g @ g.T, 0, 1) + np.clip(h @ h.T, 0, 1)) * .5
    best = np.zeros(n, dtype=np.float32); chosen = []; trace = []
    for k in range(BUDGET):
        gains = np.maximum(similarity - best[:, None], 0).mean(0)
        allowed = np.zeros(n, dtype=bool); allowed[eligible] = True; allowed[chosen] = False
        gains[~allowed] = -np.inf
        j = int(np.argmax(gains)); chosen.append(j); best = np.maximum(best, similarity[:, j])
        trace.append(dict(k=k + 1, id=rows[j]['id'], gain=float(gains[j]), coverage=float(best.mean())))
    ids = [rows[j]['id'] for j in chosen]
    save(R / 'selection/SELECTIONS_FROZEN.json', dict(time=time.time(), selected_ids=ids,
         method='global_local_facility', seed=SEED, budget=BUDGET, trace=trace,
         mask_pixels_read_before_freeze=0, train_only=True, eligible_count=len(eligible),
         features_sha256=sha(R / 'selection_features.npz')))
    np.savez_compressed(R / 'selection/descriptors.npz', global_features=g, local_histogram=h,
                        local_dictionary=km.cluster_centers_, idf=idf)
    byid = {r['merged_id']: r for r in read(R / 'protocol/merged_manifest.jsonl')}
    support = []; audit = []
    for i, ident in enumerate(ids, 1):
        row = byid[ident].copy()
        row.update(frozen_image_path=row['image_path'], frozen_mask_path=row['mask_path'])
        support.append(row)
        im = Image.open(row['image_path']).convert('RGB')
        m = np.array(Image.open(row['mask_path']).convert('L'))
        assert m.shape == np.asarray(im).shape[:2], ident
        assert m.max() > 127 and (m > 127).any(), ('selected mask invalid', ident)
        imagefile = f'{i:02d}.png'
        im.save(R / 'selected/images' / imagefile); shutil.copy2(row['mask_path'], R / 'selected/masks' / imagefile)
        audit.append(dict(id=ident, foreground_fraction=float((m > 127).mean()),
                          mask_sha256=sha(row['mask_path']), image_sha256=sha(row['image_path'])))
    jl(R / 'protocol/support_manifest.jsonl', support)
    save(R / 'SELECTED_SUPPORT_FROZEN.json', dict(selected_ids=ids,
         selection_sha256=sha(R / 'selection/SELECTIONS_FROZEN.json'), audit=audit))
    cards = ''.join(f'<article><h3>{i:02d} {a["id"]}</h3><p>面积占比 {a["foreground_fraction"]:.2%}</p>'
                    f'<img src="images/{i:02d}.png"><img src="masks/{i:02d}.png"></article>'
                    for i, a in enumerate(audit, 1))
    (R / 'selected/index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>TN3K 自动23张参考图</title>'
        '<style>body{font-family:system-ui;margin:24px;background:#eff3f7}article{background:white;padding:20px;margin:15px}'
        'img{width:44%;vertical-align:top;margin:1%}</style>'
        f'<h1>TN3K 1% 预算：自动选择 {BUDGET} 张参考图</h1>'
        '<p>仅用 train 2303 张 RGB 特征选图，名单冻结后才读取标注。官方 fold0 划分不变。</p>' + cards)
    print('SELECTED', ids, flush=True)
    status('selected', count=len(ids), eligible=len(eligible))


def base_routes():
    for split, n in (('validation', COUNTS['validation']), ('test', COUNTS['test'])):
        frozen = R / f'base_{split}_routes.jsonl'
        if frozen.exists():
            status('base_routes_cached', split=split); continue
        run(f'base_routes_{split}', [PY, str(R / 'code/stage1_feature_knn_routes.py'),
            '--mode', MODE, '--feature-source', 'sam3_base', '--feature-size', '256',
            '--knn-feature', 'patch_mean', '--beam-width', '32', '--min-bridge', '0', '--max-bridge', '6',
            '--split', split, '--protocol-root', str(R / 'protocol'), '--output-root', str(R / 'quality_root')])
        src = rp(R, split); rows = read(src)
        assert len(rows) == n * 7, (split, len(rows))
        counts = collections.Counter(r['target_id'] for r in rows)
        assert set(counts.values()) == {7} and len(counts) == n
        support = {r['merged_id'] for r in read(R / 'protocol/support_manifest.jsonl')}
        train = {r['merged_id'] for r in read(R / 'protocol/merged_manifest.jsonl') if r['split'] == 'train'}
        for row in rows:
            assert row['anchor_id'] in support
            assert set(row['bridge_ids']) <= train and not row['target_gt_used_for_search_or_inference']
        shutil.copy2(src, frozen)
        save(R / f'{split}_ROUTES_FROZEN.json', dict(count=len(rows), sha256=sha(frozen)))
    status('base_routes_done')


# ---------------------------------------------------------------- full pool + calibration
def make_folds():
    """validation 图像级 5 折。每张图自成一组（TN3K 无可靠患者 ID 可做患者级分组）。"""
    if (R / 'folds_frozen.json').exists():
        return json.loads((R / 'folds_frozen.json').read_text())
    val_ids = sorted(r['merged_id'] for r in read(R / 'protocol/merged_manifest.jsonl')
                     if r['split'] == 'validation')
    assert len(val_ids) == COUNTS['validation']
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(val_ids))
    folds = {tid: int(perm[i] % 5) for i, tid in enumerate(val_ids)}
    counts = collections.Counter(folds.values())
    assert max(counts.values()) - min(counts.values()) <= 1, counts
    save(R / 'folds_frozen.json', folds)
    status('folds_frozen', sizes=dict(counts))
    return folds


def pool():
    if (R / 'pool_membership_frozen.json').exists():
        audits = json.loads((R / 'pool_audit.json').read_text())
        status('pool_cached', audit=audits)
        make_folds(); status('pool_done', audit=audits); return
    g = route_mod()
    records = read(R / 'protocol/merged_manifest.jsonl')
    support = read(R / 'protocol/support_manifest.jsonl')
    z = np.load(R / 'quality_root/features/sam3_base_s256_features.npz')
    anchor_ids = z['anchor_ids'].tolist(); cond = z['cond_target']; pm = z['patch_mean']
    assert anchor_ids == [r['merged_id'] for r in support], 'feature anchor order != support order'
    assert cond.shape[1] == len(records) == sum(COUNTS.values())
    sim = pm @ pm.T
    state = dict(mode=MODE, patch_mean=pm, sim=sim, knn_sim=sim, knn_feature='patch_mean',
                 cond_scores=cond, id_to_anchor={a: i for i, a in enumerate(anchor_ids)},
                 text_blend=0.)
    index = {r['merged_id']: i for i, r in enumerate(records)}
    assert len(index) == len(records), 'merged_id collision'
    state['id_to_index'] = index
    train = [i for i, r in enumerate(records) if r['split'] == 'train']
    assert len(train) == COUNTS['train']
    means = cond[:, train].astype(np.float64).mean(1)
    ranking = {}; calibration = {}
    for t in records:
        j = index[t['merged_id']]; centered = cond[:, j].astype(np.float64) - means
        order = sorted(range(len(anchor_ids)), key=lambda a: (-float(centered[a]), anchor_ids[a]))
        ranking[t['merged_id']] = [anchor_ids[a] for a in order]
        calibration[t['merged_id']] = {anchor_ids[a]: dict(
            anchor_target_raw=float(cond[a, j]), anchor_target_centered=float(centered[a]),
            anchor_train_mean=float(means[a]), calibrated_anchor_rank=order.index(a))
            for a in range(len(anchor_ids))}
    save(R / 'calibration_frozen.json', dict(time=time.time(), anchor_ids=anchor_ids,
         train_means=means.tolist(), train_count=len(train), hidden_train_GT_read=False,
         rankings=ranking, scores=calibration,
         feature_sha256=sha(R / 'quality_root/features/sam3_base_s256_features.npz')))
    status('calibration_frozen', anchor_count=len(anchor_ids))

    anchors = g.t21.human_pool(support, 512)
    assert len(anchors) == BUDGET
    cache = g.build_rank_cache(state, records, support, train)
    groups_frozen = {}; audits = {}
    for split, n in (('validation', COUNTS['validation']), ('test', COUNTS['test'])):
        status('building_full_pool', split=split)
        cs = calibration
        allroutes = []
        targets = sorted((r for r in records if r['split'] == split), key=lambda r: r['merged_id'])
        assert len(targets) == n
        for pos, t in enumerate(targets, 1):
            j = index[t['merged_id']]
            for a in anchors:
                aid = a['anchor_id']; forbidden = {j, index[aid]}
                beams = [([], g.route_score(state, aid, [], j))]
                for b in range(7):
                    if b:
                        expanded = []
                        for path, _ in beams:
                            tail = j if not path else path[0]
                            nodes = g.top_ranked_nodes(cache, state, aid, tail, forbidden | set(path), 32)
                            for node in nodes:
                                new = [node, *path]
                                expanded.append((new, g.route_score(state, aid, new, j)))
                        beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:32]
                    path, score = max(beams, key=lambda item: item[1])
                    r = g.make_route(t, b, a, path, score, records)
                    r.update(cs[t['merged_id']][aid]); allroutes.append(r)
            if pos % 25 == 0: status('building_full_pool', split=split, targets=pos, total=n)
        assert len(allroutes) == n * BUDGET * 7
        assert len({r['route_id'] for r in allroutes}) == len(allroutes)
        jl(R / f'{split}_all_anchor_routes.jsonl', allroutes)
        byid_route = {r['route_id']: r for r in allroutes}
        bytarget = collections.defaultdict(list)
        for r in allroutes: bytarget[r['target_id']].append(r)
        old_by = collections.defaultdict(list)
        for r in read(R / f'base_{split}_routes.jsonl'): old_by[r['target_id']].append(r)
        members = {grp: {} for grp in GROUPS}; needed = {}
        reproduced = 0
        for tid, rows in sorted(bytarget.items()):
            anchors_here = {r['anchor_id']: r for r in rows}
            assert len(anchors_here) == BUDGET and len(rows) == BUDGET * 7
            raw = sorted(anchors_here, key=lambda a: (-anchors_here[a]['anchor_target_raw'], a))
            centered = sorted(anchors_here, key=lambda a: (-anchors_here[a]['anchor_target_centered'], a))
            assert all(anchors_here[a]['calibrated_anchor_rank'] == i for i, a in enumerate(centered))
            for kind, order in (('raw', raw), ('centered', centered)):
                for k in (1, 2):
                    rs = sorted([r for r in rows if r['anchor_id'] in order[:k]],
                                key=lambda r: (order.index(r['anchor_id']), r['bridge_count']))
                    assert len(rs) == 7 * k
                    members[f'{kind}_top{k}'][tid] = [r['route_id'] for r in rs]
                    for row in rs: needed[row['route_id']] = row
            rs = sorted(old_by[tid], key=lambda r: r['bridge_count']); assert len(rs) == 7
            members['original_per_bridge'][tid] = [r['route_id'] for r in rs]
            for row in rs:
                assert row['route_id'] in byid_route, ('base route not in pool', row['route_id'])
                new = byid_route[row['route_id']]
                for key in ('anchor_id', 'bridge_ids', 'path_bottleneck_similarity',
                            'path_mean_similarity', 'anchor_box_xywh_normalized'):
                    assert new[key] == row[key], (split, tid, key, row[key], new[key])
                reproduced += 1
                needed[row['route_id']] = new
        jl(rp(R, split), list(needed.values()))
        groups_frozen[split] = members
        audits[split] = dict(targets=n, full_pool=len(allroutes), candidate_union=len(needed),
                             base_routes_reproduced=reproduced,
                             union_per_target=round(len(needed) / n, 3))
        status('pool_split_done', split=split, **audits[split])
    save(R / 'pool_membership_frozen.json', dict(time=time.time(), groups=groups_frozen, test_GT_read=False))
    save(R / 'pool_audit.json', audits)
    for split in ('validation', 'test'):
        rows = read(rp(R, split))
        assert all(r['target_gt_used_for_search_or_inference'] is False for r in rows)
    make_folds()
    status('pool_done', audit=audits)


# ---------------------------------------------------------------- smoke
def smoke(gpu='0', targets=2):
    """2 张 validation 图的前 3 个候选做端到端检查（带 GT，仅诊断）。"""
    if (R / 'smoke_result.json').exists():
        status('smoke_cached', **json.loads((R / 'smoke_result.json').read_text())); return
    base = R / 'smoke'
    root = base / 'quality_root'
    src = read(rp(R, 'validation'))
    picked = []
    for tid in sorted({r['target_id'] for r in src})[:targets]:
        rs = [r for r in src if r['target_id'] == tid]
        picked += sorted(rs, key=lambda r: (r['calibrated_anchor_rank'], r['bridge_count']))[:7]
    jl(rp(base, 'validation'), picked)
    run('smoke_propagation', [PY, str(R / 'code/eval_route_propagation_quality.py'),
        '--checkpoint', BASE, '--mode', MODE, '--root', str(root), '--split', 'validation',
        '--canvas', '256'], env_extra={'CUDA_VISIBLE_DEVICES': str(gpu)})
    rows = read(qp(base, 'validation'))
    assert len(rows) == len(picked) and all(r['status'] == 'success' for r in rows)
    secs = [float(r['seconds']) for r in rows]
    per_b = {}
    for b in range(7):
        v = [r['gt_dice_evaluation_only'] for r in rows if r['bridge_count'] == b]
        if v: per_b[b] = round(float(np.mean(v)), 4)
    out = dict(candidates=len(rows), mean_seconds_per_candidate=round(float(np.mean(secs)), 3),
               estimated_hours_union_1190=round(float(np.mean(secs)) * 35 * 1190 / 3600, 2),
               dice_by_bridge=per_b, note='diagnostic only, includes target GT, not the experiment result')
    save(R / 'smoke_result.json', out)
    status('smoke_done', **out)


# ---------------------------------------------------------------- propagation
def propagate(split, gpu):
    frozen = R / f'{split}_PROPAGATION_COMPLETE'
    if frozen.exists():
        status('propagate_cached', split=split); return
    run(f'propagate_{split}', [PY, str(R / 'code/eval_route_propagation_quality.py'),
        '--checkpoint', BASE, '--mode', MODE, '--root', str(R / 'quality_root'), '--split', split,
        '--canvas', '256', '--no-target-gt', '--resume'],
        env_extra={'CUDA_VISIBLE_DEVICES': str(gpu)})
    rows = read(qp(R, split)); routes = read(rp(R, split))
    assert len(rows) == len(routes) and {r['route_id'] for r in rows} == {r['route_id'] for r in routes}
    assert all(r['status'] == 'success' for r in rows)
    save(R / f'{split}_PREDICTIONS_FROZEN.json', dict(time=time.time(), candidates=len(rows),
         quality_sha256=sha(qp(R, split)), forward_masks_frozen_before_GT_read=True))
    frozen.touch(); status('propagate_done', split=split, candidates=len(rows))


# ---------------------------------------------------------------- router / factorial
def extract(split, with_gt):
    feature = module('tn3k_legacy_feature', R / 'code/router.py')
    rows = {r['route_id']: r for r in read(qp(R, split))}
    members = json.loads((R / 'pool_membership_frozen.json').read_text())['groups'][split]
    allids = sorted(members[GROUPS[0]])
    for r in rows.values():
        assert r['status'] == 'success' and not r['target_gt_used_for_search_or_inference']
        assert sha(r['forward_mask_path']) == r['forward_mask_sha256']
    values = {}; gt = {}
    if with_gt:
        for r in rows.values():
            tid = r['target_id']
            if tid not in gt:
                gt[tid] = bitmap(r['target_mask_path_evaluation_only']); assert gt[tid].any(), tid
            values[r['route_id']] = metric(bitmap(r['forward_mask_path']), gt[tid])
    data = {}
    for group in GROUPS:
        rs = [[rows[rid] for rid in members[group][tid]] for tid in allids]
        x = np.array([[feature.feature_vector(r, False) for r in rr] for rr in rs], dtype=float)
        assert x.shape[2] == 28, x.shape
        data[group] = dict(ids=allids, rows=rs, raw=x, extended=x,
                           y=np.array([[values[r['route_id']]['dice'] for r in rr] for rr in rs]) if with_gt else None)
    return data, values


def paired(vectors):
    rng = np.random.default_rng(SEED); n = len(next(iter(vectors.values())))
    indices = rng.integers(0, n, (10000, n)); out = {}
    for a, b in PAIRS:
        delta = np.asarray(vectors[a]) - np.asarray(vectors[b]); boot = delta[indices].mean(1)
        out[a + '_minus_' + b] = dict(mean=float(delta.mean()),
                                      ci95=np.quantile(boot, [.025, .975]).tolist(),
                                      improved=int((delta > 1e-12).sum()), worsened=int((delta < -1e-12).sum()))
    d = ((np.asarray(vectors['centered_top2']) - np.asarray(vectors['raw_top2']))
         - (np.asarray(vectors['centered_top1']) - np.asarray(vectors['raw_top1'])))
    out['interaction'] = dict(mean=float(d.mean()), ci95=np.quantile(d[indices].mean(1), [.025, .975]).tolist())
    return out


def validate():
    if (R / 'models_frozen.json').exists():
        status('validate_cached'); return
    status('fitting_validation_routers_cpu')
    m = prev_mod()
    data, values = extract('validation', True)
    ids = data[GROUPS[0]]['ids']; n = COUNTS['validation']; assert len(ids) == n
    folds = json.loads((R / 'folds_frozen.json').read_text()); f = np.array([folds[t] for t in ids]); ix = np.arange(n)
    models = {}; summary = {}; vectors = {}; selections = {}
    for group, d in data.items():
        k = d['raw'].shape[1] // 7; config = (k, 'legacy', 1.)
        scores = m.cv(d, ix, f, config); choice = scores.argmax(1); y = d['y'][ix, choice]
        vectors[group] = y.tolist()
        oracle = float(d['y'].max(1).mean()); models[group] = m.fit(d, ix, config)
        selected = [d['rows'][i][j] for i, j in enumerate(choice)]
        selections[group] = [dict(target_id=r['target_id'], route_id=r['route_id'], anchor_id=r['anchor_id'],
                                  bridge_count=r['bridge_count'], dice=values[r['route_id']]['dice']) for r in selected]
        summary[group] = dict(candidates_per_target=d['raw'].shape[1], oof_dice=float(y.mean()), oracle_dice=oracle,
                              oracle_gap=oracle - float(y.mean()),
                              selected_anchors=dict(collections.Counter(r['anchor_id'] for r in selected)),
                              selected_bridges=dict(collections.Counter(r['bridge_count'] for r in selected)),
                              candidate_anchors=dict(collections.Counter(r['anchor_id'] for rr in d['rows'] for r in rr)),
                              fixed_rank1_b0_b6=d['y'][:, :7].mean(0).tolist())
    save(R / 'validation_results.json', dict(n=n, groups=summary, paired_oof=paired(vectors),
         per_target_oof=selections,
         note='all arms prespecified, same alpha1, no hyperparameter search; OOF uses validation GT only'))
    save(R / 'models_frozen.json', dict(time=time.time(), models=models, folds_sha256=sha(R / 'folds_frozen.json'),
         membership_sha256=sha(R / 'pool_membership_frozen.json'),
         validation_quality_sha256=sha(qp(R, 'validation')), test_GT_read=False))
    status('validation_complete', results={k: round(v['oof_dice'], 6) for k, v in summary.items()})


def evaluate():
    status('freezing_test_choices_without_GT')
    m = prev_mod()
    data, _ = extract('test', False); n = COUNTS['test']; ix = np.arange(n)
    frozen = json.loads((R / 'models_frozen.json').read_text()); choices = {}
    for group, d in data.items():
        assert len(d['ids']) == n
        js = m.predict(d, ix, frozen['models'][group]).argmax(1)
        choices[group] = [d['rows'][i][j]['route_id'] for i, j in enumerate(js)]
    save(R / 'test_choices_frozen.json', dict(time=time.time(), models_sha256=sha(R / 'models_frozen.json'),
         choices=choices, test_GT_read=False))
    status('scoring_frozen_test_choices')
    _, values = extract('test', True)
    metrics = {}; vectors = {}; per = {}
    for group, d in data.items():
        byid = {r['route_id']: r for rr in d['rows'] for r in rr}
        rs = [byid[rid] for rid in choices[group]]
        per[group] = [dict(target_id=r['target_id'], route_id=r['route_id'], anchor_id=r['anchor_id'],
                           bridge_count=r['bridge_count'], mask_path=r['forward_mask_path'],
                           **values[r['route_id']]) for r in rs]
        y = [values[r['route_id']]['dice'] for r in rs]; vectors[group] = y
        oracle = float(np.mean([max(values[r['route_id']]['dice'] for r in rr) for rr in d['rows']]))
        metrics[group] = dict(candidates_per_target=d['raw'].shape[1], dice=float(np.mean(y)),
                              iou=float(np.mean([values[r['route_id']]['iou'] for r in rs])),
                              oracle_dice=oracle, oracle_gap=oracle - float(np.mean(y)),
                              selected_anchors=dict(collections.Counter(r['anchor_id'] for r in rs)),
                              selected_bridges=dict(collections.Counter(r['bridge_count'] for r in rs)),
                              candidate_anchors=dict(collections.Counter(r['anchor_id'] for rr in d['rows'] for r in rr)),
                              fixed_rank1_b0_b6=[float(np.mean([values[rr[b]['route_id']]['dice'] for rr in d['rows']]))
                                                 for b in range(7)])
        dest = R / group / 'masks'; dest.mkdir(parents=True, exist_ok=True)
        for r in rs: shutil.copy2(r['forward_mask_path'], dest / (r['target_id'].replace('::', '__') + '.png'))
    validation = json.loads((R / 'validation_results.json').read_text())
    out = dict(validation={k: v for k, v in validation.items() if k != 'per_target_oof'},
               test=metrics, paired_test=paired(vectors), pool=json.loads((R / 'pool_audit.json').read_text()),
               calibration_summary=dict(
                   anchor_train_mean_spread=float(np.ptp(json.loads((R / 'calibration_frozen.json').read_text())['train_means']))))
    save(R / 'results.json', out); save(R / 'test_per_target.json', per); save(R / 'test_candidate_metrics.json', values)
    for f_, h in json.loads((R / 'input_hashes.json').read_text()).items():
        assert sha(f_) == h, (f_, 'changed')
    save(R / 'completion_audit.json', dict(all_test_614=True, prespecified_four_arms_plus_original=True,
         same_alpha_and_folds=True, test_choices_frozen_before_GT_read=True, inputs_unchanged=True,
         base_routes_reproduced_from_pool=True, merged_id_namespaced_by_split=True))

    def row(g): return metrics[g], validation['groups'][g]
    lines = ['# TN3K：复现 BUSI 的自动参考图 + 分数校准 x top1/top2 消融', '',
             f"官方 fold0 划分 train {COUNTS['train']} / validation {COUNTS['validation']} / test {COUNTS['test']}；"
             f"round(1% x {COUNTS['train']}) = {BUDGET} 张自动参考图；SAM3-base，GT box，无文本，canvas256。",
             '每组独立拟合相同 legacy28 Ridge(alpha=1)，沿用同一 validation 图像级 5 折。', '',
             '## 1. 四组 + original 对照', '',
             '| 组别 | 每图候选 | val OOF Dice | test Dice | test IoU | test Oracle | Oracle差距 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        x = metrics[group]; v = validation['groups'][group]
        lines.append(f"| {group} | {x['candidates_per_target']} | {v['oof_dice']:.6f} | {x['dice']:.6f} | "
                     f"{x['iou']:.6f} | {x['oracle_dice']:.6f} | {x['oracle_gap']:.6f} |")
    lines += ['', '## 2. b0-b6 逐桥 Dice（校准 top1 单参考图 vs 原版单参考图）', '',
              '| 路径 | 原版 Dice | 校准 top1 Dice | 校准增量 |', '|---|---:|---:|---:|']
    o = metrics['original_per_bridge']['fixed_rank1_b0_b6']
    c = metrics['centered_top1']['fixed_rank1_b0_b6']
    for b in range(7):
        lines.append(f'| b{b} | {o[b]:.6f} | {c[b]:.6f} | {c[b] - o[b]:+.6f} |')
    ov = validation['groups']['original_per_bridge']['fixed_rank1_b0_b6']
    cv_ = validation['groups']['centered_top1']['fixed_rank1_b0_b6']
    lines += ['', 'validation 上同口径（用于说明 b 不能在 test 上选）：', '',
              '| 路径 | 原版 Dice | 校准 top1 Dice |', '|---|---:|---:|']
    for b in range(7):
        lines.append(f'| b{b} | {ov[b]:.6f} | {cv_[b]:.6f} |')
    lines += ['', f"validation 上校准 top1 最优 b = b{int(np.argmax(cv_))}，test 上最优 b = b{int(np.argmax(c))}。", '',
              '## 3. 候选池 Oracle（GT 上界诊断，不可作为可部署结果）', '',
              '| 候选池 | 每图候选 | val Oracle | test Oracle |', '|---|---:|---:|---:|']
    for group in GROUPS:
        v = validation['groups'][group]; x = metrics[group]
        lines.append(f"| {group} | {x['candidates_per_target']} | {v['oracle_dice']:.6f} | {x['oracle_dice']:.6f} |")
    lines += ['', '## 4. 配对比较（test，10000 次逐图 bootstrap，seed2026）', '',
              '| 对比 | test Dice增量 | 95% 区间 | 变好/变差 |', '|---|---:|---|---:|']
    for name, x in out['paired_test'].items():
        lines.append(f"| {name} | {x['mean']:.6f} | [{x['ci95'][0]:.6f}, {x['ci95'][1]:.6f}] | "
                     f"{x.get('improved', '-')}/{x.get('worsened', '-')} |")
    lines += ['', '## 5. 校准诊断', '',
              f"23 张参考图的训练均值跨度 {out['calibration_summary']['anchor_train_mean_spread']:.4f}。",
              f"原版 b0-b6 逐桥独立选参考图；校准 top1 每张目标图固定同一张 rank-1 参考图，只变桥长，不经过 Router。",
              '', '## 6. 候选池规模', '']
    for split, a in out['pool'].items():
        lines += [f"- {split}: {a['targets']} 张目标图，全池 {a['full_pool']} 条（{BUDGET} 参考 x 7 桥），"
                  f"实际传播并集 {a['candidate_union']} 条（{a['union_per_target']}/图），"
                  f"原版路径从池中精确复现 {a['base_routes_reproduced']} 条。"]
    lines += ['', '四组预先固定，没有根据 test 选择超参数。95% 区间为逐图配对探索性区间，未做多重比较校正。',
              'test 在本轮之前未用于该方法族的任何选参，但 TN3K test 是公开基准，不等同于私有盲测集。',
              '详细每图结果、冻结模型和候选成员见同目录 JSON。']
    (R / 'report.md').write_text('\n'.join(lines) + '\n')
    (R / 'COMPLETE').touch()
    status('complete', test={g: round(metrics[g]['dice'], 6) for g in GROUPS})


# ---------------------------------------------------------------- main
def freeze_inputs():
    files = ([R / 'pipeline.py', R / 'protocol/merged_manifest.jsonl', R / 'protocol/support_manifest.jsonl',
              R / 'selection/SELECTIONS_FROZEN.json', R / 'folds_frozen.json',
              R / 'quality_root/features/sam3_base_s256_features.npz', R / 'pool_membership_frozen.json']
             + list((R / 'code').glob('*.py'))
             + [R / f'{s}_all_anchor_routes.jsonl' for s in ('validation', 'test')])
    save(R / 'input_hashes.json', {str(f): sha(f) for f in files})
    status('inputs_frozen', count=len(files))


if __name__ == '__main__':
    try:
        a = sys.argv[1:]
        if not a: raise ValueError('usage: pipeline.py <action> [args]')
        if a[0] == 'prepare': prepare()
        elif a[0] == 'select-features': select_features(*a[1:])
        elif a[0] == 'select': select()
        elif a[0] == 'base-routes': base_routes()
        elif a[0] == 'pool': pool()
        elif a[0] == 'smoke': smoke(a[1] if len(a) > 1 else '0', int(a[2]) if len(a) > 2 else 2)
        elif a[0] == 'propagate': propagate(a[1], a[2] if len(a) > 2 else '0')
        elif a[0] == 'freeze-inputs': freeze_inputs()
        elif a[0] == 'validate': validate()
        elif a[0] == 'evaluate': evaluate()
        else: raise ValueError(a[0])
    except BaseException:
        if R.exists():
            try: status('failed', error=traceback.format_exc())
            except Exception: pass
        raise
