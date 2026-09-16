"""BUSI: train-only anchor score centering, per-anchor TP b0-b6, grouped-CV Router."""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import sys, json, time, hashlib, shutil, subprocess, traceback, importlib.util, collections
import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E = P / 'new_project/experiments'
B = E / 'busi_auto5_tp_1pct_20260913'
R = E / 'busi_calibrated_multi_anchor_20260913'
MODE = 'sam3enc_anchor_conditioned_target_pooling'
PY = '/home/violet/anaconda3/envs/sam3/bin/python'
BASE = '/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
CONFIGS = [(0, 'legacy', 1.)] + [(k, kind, a) for k in (1, 2, 3, 5) for kind, a in [('legacy', 1.), ('calibrated_peer', 10.), ('calibrated_peer', 100.)]]

def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n'); tmp.replace(p)
def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def jl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(stage, **kw): save(R / 'status.json', dict(stage=stage, time=time.time(), **kw)); print(stage, kw, flush=True)
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
def mask(p): return np.asarray(Image.open(p).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127
def dice(a, b):
    n = int(a.sum()) + int(b.sum()); return 2 * int((a & b).sum()) / n if n else 1.
def quality(split): return R / f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl'
def routes_path(split): return R / f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl'
def cname(c): return f'k{c[0]}_{c[1]}_a{c[2]:g}'

def prepare():
    R.mkdir(exist_ok=False)
    for name in ('code', 'logs', 'protocol'): (R / name).mkdir()
    shutil.copy2(__file__, R / 'pipeline.py')
    for name in ('stage1_feature_knn_routes.py', 'eval_route_propagation_quality.py', 'run_t21_dynamic_pseudovideo.py', 'router.py'):
        shutil.copy2(B / 'code' / name, R / 'code' / name)
    for name in ('merged_manifest.jsonl', 'support_manifest.jsonl'): shutil.copy2(B / 'protocol' / name, R / 'protocol' / name)
    inputs = [B / 'quality_root/features/sam3_base_s256_features.npz', Path(BASE)]
    inputs += list((R / 'code').glob('*.py')) + list((R / 'protocol').glob('*.jsonl'))
    save(R / 'input_hashes.json', {str(p): sha(p) for p in inputs})
    save(R / 'predefined_policy.json', dict(configs=CONFIGS, seed=2026,
         anchors=5, train=517, validation=64, test=66, canvas=256, feature_size=256,
         anchor_selection='unchanged automatic five; same split and labels',
         calibration='cond_target(anchor,target) minus mean cond_target(anchor,517 train RGBs); no hidden train GT',
         retention='top 1/2/3/5 anchors by calibrated target score; all b0-b6 for each retained anchor; original raw 7 as control',
         path_search='original per-anchor TP bottleneck/mean and patch_mean KNN beam32, unchanged',
         prompt='anchor GT tight box, no text; original highest-score-object propagation and predicted-mask return',
         router='original 28 features; calibrated_peer adds 8 peer features and 5 anchor calibration features, centered within target',
         selection='5-fold image-grouped validation CV; nested outer5/inner4 estimates full retention/model choice',
         test_policy='model/retention frozen on validation; freeze every test route choice before reading test GT',
         test_predictions='selected K plus calibrated top1 plus original baseline; report Oracle separately per candidate pool',
         prior_test_seen=True, validation_labels_additional_to_5_train_GT=True,
         gpu='wait for >=18000MiB free and no compute process over 1GiB; no existing process killed'))
    status('prepared')

def make_routes():
    if (R / 'ROUTES_COMPLETE').exists(): return
    status('building_per_anchor_routes_cpu')
    g = module('multi_route_source', R / 'code/stage1_feature_knn_routes.py')
    records = read(R / 'protocol/merged_manifest.jsonl'); support = read(R / 'protocol/support_manifest.jsonl')
    z = np.load(B / 'quality_root/features/sam3_base_s256_features.npz')
    anchor_ids = z['anchor_ids'].tolist(); cond = z['cond_target']; pm = z['patch_mean']
    # Preserve original NumPy dtype/arithmetic and exact manifest ordering.
    sim = pm @ pm.T
    state = dict(mode=MODE, patch_mean=pm, sim=sim, knn_sim=sim, knn_feature='patch_mean', cond_scores=cond,
                 id_to_anchor={a: i for i, a in enumerate(anchor_ids)}, text_blend=0.)
    index = {r['merged_id']: i for i, r in enumerate(records)}; state['id_to_index'] = index
    train = [i for i, r in enumerate(records) if r['split'] == 'train']; assert len(train) == 517
    means = cond[:, train].astype(np.float64).mean(1)
    ranking = {}; calibration = {}
    for t in records:
        j = index[t['merged_id']]; centered = cond[:, j].astype(np.float64) - means
        order = sorted(range(5), key=lambda a: (-float(centered[a]), anchor_ids[a]))
        ranking[t['merged_id']] = [anchor_ids[a] for a in order]
        calibration[t['merged_id']] = {anchor_ids[a]: dict(anchor_target_raw=float(cond[a, j]),
            anchor_target_centered=float(centered[a]), anchor_train_mean=float(means[a]),
            calibrated_anchor_rank=order.index(a)) for a in range(5)}
    save(R / 'calibration_frozen.json', dict(time=time.time(), anchor_ids=anchor_ids, train_means=means.tolist(),
         train_count=517, hidden_train_GT_read=False, rankings=ranking, scores=calibration,
         feature_sha256=sha(B / 'quality_root/features/sam3_base_s256_features.npz')))
    anchors = g.t21.human_pool(support, 512)
    cache = g.build_rank_cache(state, records, support, train)
    audit = {}
    for split, n in [('validation', 64), ('test', 66)]:
        allroutes = []
        for pos, t in enumerate(sorted((r for r in records if r['split'] == split), key=lambda r: r['merged_id']), 1):
            j = index[t['merged_id']]
            for a in anchors:
                aid = a['anchor_id']; forbidden = {j, index[aid]}; beams = [([], g.route_score(state, aid, [], j))]
                for b in range(7):
                    if b:
                        expanded = []
                        for path, _ in beams:
                            tail = j if not path else path[0]
                            nodes = g.top_ranked_nodes(cache, state, aid, tail, forbidden | set(path), 32)
                            for node in nodes:
                                new = [node, *path]; expanded.append((new, g.route_score(state, aid, new, j)))
                        beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:32]
                    path, score = max(beams, key=lambda item: item[1])
                    r = g.make_route(t, b, a, path, score, records); r.update(calibration[t['merged_id']][aid]); allroutes.append(r)
            if pos % 8 == 0: status('building_per_anchor_routes_cpu', split=split, targets=pos, total=n)
        assert len(allroutes) == n * 35 and len({r['route_id'] for r in allroutes}) == n * 35
        byid = {r['route_id']: r for r in allroutes}; groups = collections.defaultdict(list)
        for r in allroutes: groups[(r['target_id'], r['bridge_count'])].append(r)
        oldroutes = read(B / f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl')
        # Original single winner must be exactly reproduced, including tie order and route ID.
        for old in oldroutes:
            new = max(groups[(old['target_id'], old['bridge_count'])], key=lambda r: ((r['path_bottleneck_similarity'], r['path_mean_similarity']), r['anchor_id']))
            assert new['route_id'] == old['route_id'], (split, old['route_id'], new['route_id'])
            for key, value in old.items(): assert new[key] == value, (key, value, new[key])
        jl(R / f'{split}_all_anchor_routes.jsonl', allroutes)
        if split == 'validation': jl(routes_path(split), allroutes)
        oldq = read(B / f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl')
        reused = []
        for q in oldq:
            assert q['route_id'] in byid and sha(q['forward_mask_path']) == q['forward_mask_sha256']
            row = {k: v for k, v in q.items() if not k.startswith('gt_')}; row.update(byid[q['route_id']]); reused.append(row)
        jl(quality(split), reused)
        audit[split] = dict(targets=n, per_target_candidates=35, old_paths_exact_match=len(oldroutes), reused_masks_verified=len(reused), new_candidates=len(allroutes)-len(reused))
    save(R / 'route_reuse_audit.json', audit); (R / 'ROUTES_COMPLETE').touch(); status('routes_complete', audit=audit)

def wait_gpu():
    last = None
    while True:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,uuid', '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,used_memory', '--format=csv,noheader,nounits'], text=True)
        busy = set()
        for line in apps.splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) == 2:
                try:
                    if int(parts[1]) > 1024: busy.add(parts[0])
                except ValueError: busy.add(parts[0])
        available = []; memory = []
        for line in raw.splitlines():
            gpu, free, uid = [x.strip() for x in line.split(',')]; memory.append(dict(gpu=int(gpu), free_mib=int(free)))
            if int(free) >= 18000 and uid not in busy: available.append(int(gpu))
        if available:
            gpu = 1 if 1 in available else available[0]; status('gpu_available', gpu=gpu); return gpu
        if last != memory: status('waiting_for_gpu', memory=memory); last = memory
        time.sleep(30)

def propagate(split):
    if (R / f'{split}_PROPAGATION_COMPLETE').exists(): return
    gpu = wait_gpu()
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH='/Data_8TB/lht/sam3:' + str(P / 'src'), PYTHONUNBUFFERED='1')
    cmd = [PY, str(R / 'code/eval_route_propagation_quality.py'), '--checkpoint', BASE, '--mode', MODE,
           '--root', str(R / 'quality_root'), '--split', split, '--canvas', '256', '--no-target-gt', '--resume']
    status('propagating_' + split, gpu=gpu, command=cmd)
    with (R / f'logs/propagation_{split}.log').open('a') as log:
        proc = subprocess.Popen(cmd, cwd=P, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        save(R / f'{split}_process.json', dict(pid=proc.pid, gpu=gpu, command=cmd, time=time.time()))
        rc = proc.wait()
    if rc: raise RuntimeError(f'{split} propagation failed, return code {rc}; see log')
    rows = read(quality(split)); routes = read(routes_path(split))
    assert len(rows) == len(routes) and {r['route_id'] for r in rows} == {r['route_id'] for r in routes}
    assert all(r['status'] == 'success' for r in rows)
    save(R / f'{split}_PREDICTIONS_FROZEN.json', dict(time=time.time(), candidates=len(rows), quality_sha256=sha(quality(split))))
    (R / f'{split}_PROPAGATION_COMPLETE').touch()

def extract(split, k, with_gt):
    legacy = module('router_feature_source', R / 'code/router.py')
    rows = read(quality(split)); groups = collections.defaultdict(list)
    for r in rows: groups[r['target_id']].append(r)
    old = {r['route_id'] for r in read(B / f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl')}
    ids = sorted(groups); raw = []; extended = []; ys = []; ordered = []; hashes = {}
    for tid in ids:
        rs = [r for r in groups[tid] if (r['route_id'] in old if k == 0 else r['calibrated_anchor_rank'] < k)]
        rs.sort(key=lambda r: (r['calibrated_anchor_rank'], r['bridge_count']) if k else (r['bridge_count'],))
        assert len(rs) == 7 * max(k, 1), (tid, k, len(rs))
        masks = []
        for r in rs:
            assert not r['target_gt_used_for_search_or_inference']
            assert sha(r['forward_mask_path']) == r['forward_mask_sha256']; hashes[r['forward_mask_path']] = r['forward_mask_sha256']; masks.append(mask(r['forward_mask_path']))
        x = np.array([legacy.feature_vector(r, False) for r in rs], dtype=float)
        c = len(rs); peer = np.array([[dice(a, b) for b in masks] for a in masks]); other = peer[~np.eye(c, dtype=bool)].reshape(c, c - 1)
        areas = np.array([m.mean() for m in masks]); med = np.median(areas)
        extra = np.column_stack([other.mean(1), other.min(1), other.max(1), other.std(1), areas, abs(areas-med), np.log((areas+1e-5)/(med+1e-5)), other.mean(1)*x[:,7]])
        cal = np.array([[r['anchor_target_raw'], r['anchor_target_centered'], r['path_bottleneck_similarity']-r['anchor_train_mean'], r['path_mean_similarity']-r['anchor_train_mean'], r['calibrated_anchor_rank']/4] for r in rs])
        raw.append(x); extended.append(np.column_stack([x, extra, cal])); ordered.append(rs)
        if with_gt:
            gt = mask(rs[0]['target_mask_path_evaluation_only']); assert gt.any(); ys.append([dice(m, gt) for m in masks])
    return dict(ids=ids, rows=ordered, raw=np.array(raw), extended=np.array(extended), y=np.array(ys) if with_gt else None, hashes=hashes)

def xarray(d, kind):
    if kind == 'legacy': return d['raw']
    x = d['extended']; return x - x.mean(1, keepdims=True)
def fit(d, ix, config):
    _, kind, alpha = config; x = xarray(d, kind)[ix]; x = x.reshape(-1, x.shape[-1]); y = d['y'][ix].copy()
    if kind != 'legacy': y -= y.mean(1, keepdims=True)
    y = y.ravel(); mean=x.mean(0); std=np.maximum(x.std(0),1e-8); z=(x-mean)/std; ym=y.mean()
    w=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-ym))
    return dict(config=config, means=mean.tolist(), stds=std.tolist(), weights=w.tolist(), intercept=float(ym))
def predict(d, ix, m): return (xarray(d,m['config'][1])[ix]-np.array(m['means']))/np.array(m['stds'])@np.array(m['weights'])+m['intercept']
def selected_y(d, ix, scores): return d['y'][ix, scores.argmax(1)]
def cv(d, ix, folds, config):
    out=np.empty((len(ix), d['raw'].shape[1])); local=folds[ix]
    for f in sorted(set(local)):
        tr=ix[local!=f]; te=ix[local==f]; assert not set(tr)&set(te)
        out[local==f]=predict(d,te,fit(d,tr,config))
    return out

def select_validation():
    if (R / 'models_frozen.json').exists(): return
    status('validation_grouped_cv_cpu')
    data={k:extract('validation',k,True) for k in (0,1,2,3,5)}
    ids=data[0]['ids']; assert len(ids)==64 and all(d['ids']==ids for d in data.values())
    mapping=json.loads((E/'busi_auto5_tp_router_20260913/busi/folds_frozen.json').read_text())
    folds=np.array([mapping[t] for t in ids]); ix=np.arange(64); save(R/'folds_frozen.json',mapping)
    values={}; oof={}
    for c in CONFIGS:
        d=data[c[0]]; picked=selected_y(d,ix,cv(d,ix,folds,c)); values[cname(c)]=float(picked.mean()); oof[cname(c)]=picked.tolist()
    # The control must reproduce the previously completed Router's grouped CV.
    old_result=json.loads((E/'busi_auto5_tp_router_20260913/busi/results.json').read_text())
    expected=old_result['validation']['cv_scores']['legacy_a1']; assert abs(values[cname(CONFIGS[0])]-expected)<1e-8
    chosen=max(CONFIGS,key=lambda c:values[cname(c)]); nested=np.empty(64); outer=[]
    for f in range(5):
        tr=ix[folds!=f]; te=ix[folds==f]; inner={}
        for c in CONFIGS:
            d=data[c[0]]; inner[cname(c)]=float(selected_y(d,tr,cv(d,tr,folds,c)).mean())
        winner=max(CONFIGS,key=lambda c:inner[cname(c)]); d=data[winner[0]]
        nested[te]=selected_y(d,te,predict(d,te,fit(d,tr,winner))); outer.append(dict(fold=f,chosen=cname(winner),inner_scores=inner))
    models=dict(selected=fit(data[chosen[0]],ix,chosen), original=fit(data[0],ix,CONFIGS[0]))
    summaries={str(k):dict(candidate_count=7*max(k,1),oracle=float(d['y'].max(1).mean())) for k,d in data.items()}
    fixed=data[1]['y'].mean(0).tolist()
    result=dict(config_oof_dice=values,selected_config=cname(chosen),selected_config_oof_dice=values[cname(chosen)],
                nested_selection_oof_dice=float(nested.mean()),outer=outer,pools=summaries,calibrated_top1_b0_b6=fixed,
                raw_b0_b6=data[0]['y'].mean(0).tolist(),oof_per_target=oof,target_ids=ids)
    save(R/'validation_results.json',result)
    save(R/'models_frozen.json',dict(time=time.time(),models=models,selected_config=chosen,
         calibrated_top1_fixed_bridge=int(np.argmax(fixed)),validation_quality_sha256=sha(quality('validation')),
         validation_results_sha256=sha(R/'validation_results.json')))
    selected_k=chosen[0]; routes=read(R/'test_all_anchor_routes.jsonl')
    old_ids={r['route_id'] for r in read(B/f'quality_root/{MODE}/test_pool0_stage1/routes.jsonl')}
    # Include original control and calibrated top1 b0-b6 even when winner is original.
    needed=[r for r in routes if r['calibrated_anchor_rank']<max(selected_k,1) or r['route_id'] in old_ids]
    jl(routes_path('test'),needed); save(R/'test_candidate_pool_frozen.json',dict(time=time.time(),selected_k=selected_k,
         route_ids=[r['route_id'] for r in needed],models_sha256=sha(R/'models_frozen.json'),candidate_routes_sha256=sha(routes_path('test'))))
    status('validation_complete',winner=cname(chosen),oof=values[cname(chosen)],nested=float(nested.mean()),test_candidates=len(needed))

def evaluate_test():
    status('freezing_test_choices_without_GT')
    frozen=json.loads((R/'models_frozen.json').read_text()); k=frozen['selected_config'][0]
    data={j:extract('test',j,False) for j in sorted({0,1,k})}; ids=data[0]['ids']; assert len(ids)==66
    assert all(d['ids']==ids for d in data.values()); ix=np.arange(66)
    selected=predict(data[k],ix,frozen['models']['selected']).argmax(1)
    original=predict(data[0],ix,frozen['models']['original']).argmax(1)
    selections={'selected_router':[data[k]['rows'][i][b] for i,b in enumerate(selected)],
                'original_router':[data[0]['rows'][i][b] for i,b in enumerate(original)]}
    for b in range(7):
        selections[f'calibrated_top1_b{b}']=[rs[b] for rs in data[1]['rows']]
        selections[f'original_b{b}']=[rs[b] for rs in data[0]['rows']]
    b=frozen['calibrated_top1_fixed_bridge']; selections['calibrated_top1_val_fixed']=selections[f'calibrated_top1_b{b}']
    save(R/'test_choices_frozen.json',dict(time=time.time(),models_sha256=sha(R/'models_frozen.json'),target_ids=ids,
         routes={name:[r['route_id'] for r in rs] for name,rs in selections.items()},test_GT_read=False))
    status('evaluating_frozen_test_predictions')
    rows=read(quality('test')); gt_by_id={}; metrics={}
    for r in rows:
        if r['target_id'] not in gt_by_id: gt_by_id[r['target_id']]=mask(r['target_mask_path_evaluation_only'])
        gt=gt_by_id[r['target_id']]; assert gt.any(); m=mask(r['forward_mask_path']); union=int((m|gt).sum())
        metrics[r['route_id']]=dict(dice=dice(m,gt),iou=int((m&gt).sum())/union if union else 1.)
    results={}; per={}
    for name,rs in selections.items():
        per[name]=[dict(target_id=r['target_id'],anchor_id=r['anchor_id'],bridge_count=r['bridge_count'],route_id=r['route_id'],mask_path=r['forward_mask_path'],**metrics[r['route_id']]) for r in rs]
        results[name]=dict(dice=float(np.mean([x['dice'] for x in per[name]])),iou=float(np.mean([x['iou'] for x in per[name]])),
             anchors=dict(collections.Counter(r['anchor_id'] for r in rs)),bridges=dict(collections.Counter(r['bridge_count'] for r in rs)))
    pools={str(j):dict(candidates_per_target=7*max(j,1),oracle=float(np.mean([max(metrics[r['route_id']]['dice'] for r in rs) for rs in d['rows']]))) for j,d in data.items()}
    rng=np.random.default_rng(2026); deltas={}
    for name,ref in [('selected_router','original_router'),('calibrated_top1_b0','original_b0'),('calibrated_top1_val_fixed','original_router')]:
        diff=np.array([a['dice']-b['dice'] for a,b in zip(per[name],per[ref])]); boot=diff[rng.integers(0,66,(10000,66))].mean(1)
        deltas[name+'_vs_'+ref]=dict(delta=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    assert abs(results['original_router']['dice']-.5668083894947364)<1e-8
    for name in ('selected_router','original_router','calibrated_top1_val_fixed'):
        dest=R/name/'masks'; dest.mkdir(parents=True,exist_ok=True)
        for r in per[name]: shutil.copy2(r['mask_path'],dest/(r['target_id'].replace('::','__')+'.png'))
    save(R/'test_per_target.json',per); save(R/'test_candidate_metrics.json',metrics)
    out=dict(n_validation=64,n_test=66,selected_config=frozen['selected_config'],
         validation=json.loads((R/'validation_results.json').read_text()),test=results,pools=pools,paired=deltas)
    save(R/'results.json',out)
    for p,h in json.loads((R/'input_hashes.json').read_text()).items(): assert sha(p)==h,(p,'input changed')
    for d in data.values():
        for p,h in d['hashes'].items(): assert sha(p)==h
    save(R/'completion_audit.json',dict(all_66_test_included=True,old_router_reproduced=True,original_inputs_unchanged=True,
         test_choices_frozen_before_GT_read=True,selection_and_nested_CV_grouped_by_image=True))
    lines=['# BUSI 分数校准与多参考候选实验','',
      '固定自动选出的 5 张训练参考图，原 train517 / val64 / test66 划分，SAM3-base，GT box，无文本，256 评价。',
      '校准仅使用 517 张训练 RGB 特征；Router 使用额外的 64 张验证集标注。测试集此前已有诊断分析，本轮参数只在验证集选择。','',
      f"验证集选定：{cname(frozen['selected_config'])}；OOF Dice={out['validation']['selected_config_oof_dice']:.6f}；完整选择流程 nested OOF={out['validation']['nested_selection_oof_dice']:.6f}。",'',
      '| 方法 | test Dice |','|---|---:|']
    for name,res in results.items(): lines.append(f"| {name} | {res['dice']:.6f} |")
    lines+=['','候选池 Oracle 仅作 GT 上界诊断，候选数量不同的 Oracle 分开列出：','']
    for j,pool in pools.items(): lines.append(f"- k={j}（0 为旧路线），{pool['candidates_per_target']} 候选：Oracle Dice {pool['oracle']:.6f}")
    lines+=['','每图结果、验证集全部配置、bootstrap 区间及可复现命令见同目录 JSON 和 README.md。']
    (R/'report.md').write_text('\n'.join(lines)+'\n'); (R/'COMPLETE').touch(); status('complete',test=results['selected_router'])

def run():
    make_routes(); propagate('validation'); select_validation(); propagate('test'); evaluate_test()

if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare': prepare()
        elif sys.argv[1]=='routes': make_routes()
        elif sys.argv[1]=='run': run()
        elif sys.argv[1]=='validate': select_validation()
        elif sys.argv[1]=='evaluate': evaluate_test()
        else: raise ValueError(sys.argv[1])
    except BaseException:
        if R.exists(): status('failed',error=traceback.format_exc())
        raise
