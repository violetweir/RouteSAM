"""Reproduce the historical propagation-quality router, without a student."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

PROJECT = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
RUN = PROJECT / 'work/rerun_kvasir_sam3base_test_20260906'
OLD = PROJECT / 'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6'
OUT = RUN / 'final_masks_no_student'
MODES = ['anchor_conditioned_target_pooling', 'anchor_conditioned_patch_correspondence']

def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def jsonl(path, rr):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rr))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mask(path):
    return np.asarray(Image.open(path).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127

def metric(pred, gt):
    inter = int(np.logical_and(pred, gt).sum())
    a, b = int(pred.sum()), int(gt.sum())
    return {'dice': 2*inter/max(a+b, 1), 'iou': inter/max(a+b-inter, 1)}

def main():
    assert (RUN / 'COMPLETE').exists()
    OUT.mkdir(exist_ok=False)
    source = PROJECT / 'scripts/analyze_propagation_quality_router.py'
    shutil.copy2(source, OUT / source.name)
    shutil.copy2(__file__, OUT / 'evaluate.py')
    spec = importlib.util.spec_from_file_location('historical_router', source)
    router = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router)
    protocol = read(RUN / 'protocol/merged_manifest.jsonl')
    val_ids = {r['merged_id'] for r in protocol if r['split'] == 'validation'}
    test_ids = {r['merged_id'] for r in protocol if r['split'] == 'test'}
    assert len(val_ids) == len(test_ids) == 100 and not (val_ids & test_ids)
    validation, inputs = [], {}
    for mode in MODES:
        p = OLD / ('sam3enc_' + mode) / 'propagation_quality_validation/propagation_quality.jsonl'
        rr = read(p)
        assert len(rr) == 700 and {r['target_id'] for r in rr} == val_ids
        assert len({(r['target_id'], r['bridge_count']) for r in rr}) == 700
        validation.extend(dict(r, feature_mode=mode) for r in rr)
        inputs[str(p)] = sha(p)
    scopes = {'target_pooling': [MODES[0]], 'patch_correspondence': [MODES[1]], 'combined': MODES}
    # Freeze all settings and validation-fitted scorers before opening test results.
    scorers = {}
    for minb in [3, 0]:
        for scope, modes in scopes.items():
            vv = [r for r in validation if r['feature_mode'] in modes and minb <= r['bridge_count'] <= 6]
            scorer = router.Ridge.fit(vv, ridge=1.0, include_mode=(scope == 'combined'))
            scorers[minb, scope] = scorer
            save(OUT / f'router_b{minb}_b6_{scope}.json', vars(scorer))
    save(OUT / 'protocol.json', {'method': 'historical propagation-quality ridge router', 'student_used': False, 'ridge': 1.0, 'fit_split': 'historical validation only', 'test_gt_used_for_selection': False, 'candidate_ranges': [[3,6],[0,6]], 'canvas': 256, 'coverage': 'all 100 test targets, no confidence filtering', 'source_sha256': sha(source), 'input_sha256': inputs, 'mode_mapping': {m: 'sam3enc_' + m for m in MODES}, 'tie_break': 'max(score, -bridge_count, route_id)'})
    test = []
    for mode in MODES:
        p = RUN / 'quality_root' / ('sam3enc_' + mode) / 'propagation_quality_test/propagation_quality.jsonl'
        rr = read(p)
        assert len(rr) == 700 and {r['target_id'] for r in rr} == test_ids
        test.extend(dict(r, feature_mode=mode) for r in rr)
        inputs[str(p)] = sha(p)
    results = []
    for minb in [3, 0]:
        for scope, modes in scopes.items():
            scorer = scorers[minb, scope]
            pool = [r for r in test if r['feature_mode'] in modes and minb <= r['bridge_count'] <= 6]
            groups = router.grouped(pool)
            dest = OUT / f'b{minb}_b6' / scope
            (dest / 'masks').mkdir(parents=True)
            choices = []
            for target, candidates in sorted(groups.items()):
                # The selection function receives no target GT metrics or target GT path.
                clean = [{k:v for k,v in r.items() if not k.startswith('gt_') and 'evaluation_only' not in k} for r in candidates]
                chosen = max(clean, key=lambda r: (scorer.score(r), -int(r['bridge_count']), r['route_id']))
                p = dest / 'masks' / (target.replace('::', '__') + '.png')
                shutil.copy2(chosen['forward_mask_path'], p)
                choices.append({'target_id': target, 'route_id': chosen['route_id'], 'route_mode': 'sam3enc_' + chosen['feature_mode'], 'bridge_count': chosen['bridge_count'], 'router_score': scorer.score(chosen), 'final_mask_path': str(p), 'mask_sha256': sha(p)})
            # Commit final mask choices before GT evaluation.
            jsonl(dest / 'selected_masks.jsonl', choices)
            evaluated = []
            for chosen in choices:
                original = next(r for r in groups[chosen['target_id']] if r['route_id'] == chosen['route_id'] and 'sam3enc_' + r['feature_mode'] == chosen['route_mode'])
                values = metric(mask(chosen['final_mask_path']), mask(original['target_mask_path_evaluation_only']))
                assert abs(values['dice'] - original['gt_dice_evaluation_only']) < 1e-12
                evaluated.append(dict(chosen, **values))
            jsonl(dest / 'per_target_metrics.jsonl', evaluated)
            historical = router.evaluate(pool, scorer)
            dice = float(np.mean([r['dice'] for r in evaluated]))
            assert abs(dice - historical['selected_dice']) < 1e-12
            summary = {'range': f'b{minb}-b6', 'scope': scope, 'targets': len(evaluated), 'dice': dice, 'iou': float(np.mean([r['iou'] for r in evaluated])), 'oracle_dice_evaluation_only': historical['oracle_dice'], 'mode_counts': dict(collections.Counter(r['route_mode'] for r in evaluated)), 'bridge_counts': dict(collections.Counter(r['bridge_count'] for r in evaluated)), 'final_masks': str(dest / 'masks')}
            assert summary['targets'] == 100
            save(dest / 'summary.json', summary)
            results.append(summary)
    save(OUT / 'results.json', results)
    save(OUT / 'input_sha256.json', inputs)
    lines = ['# Kvasir SAM3-base：无学生网络的最终 mask 评测', '', '使用本次重新传播的 test 候选，复现旧报告的传播质量 Router。未加载学生 checkpoint 或学生预测，未使用 B7。', '', 'Router 为 ridge=1.0 的岭回归评分器，只在原历史 validation 传播记录上拟合；使用回环、路径、mask 轨迹、SAM 分数等特征。测试选路函数不接收 target GT 指标或 GT 路径。先保存最终选路，再逐像素重新计算 Dice/IoU。', '', '每个设置覆盖全部 100 张原 test 图。b3-b6 为旧报告无学生阶段的设置，b0-b6 为完整候选范围；二者均预先固定，没有根据 test 选择范围。单模式各自拟合 Router，联合模式额外使用模式特征。', '', '| 候选范围 | 候选路线 | Test Dice | Test IoU | Oracle（仅分析） |', '|---|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['range']} | {r['scope']} | {r['dice']:.6f} | {r['iou']:.6f} | {r['oracle_dice_evaluation_only']:.6f} |")
    lines += ['', '最终 mask 是 Router 分数最高的一个 SAM3 候选 mask，直接保存为 256×256 二值 PNG，没有像素融合。每组 mask、逐图选路与逐图指标保存在对应 b*_b6/<scope>/ 子目录。', '', '来源：README.md 的 Candidate-Invariant Propagation-Quality Router，以及 reproduction_reports/C0_256_base_reproduction.md 的 Step 2；代码为 scripts/eval_ft1pct_pq_router.py 和 scripts/analyze_propagation_quality_router.py。历史 DINO 与本次 SAM3-base@256 选路特征不同，不能将不同版本的报告数值视为同一配置。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'COMPLETE').touch()
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
