import collections
import datetime
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R = P / 'new_project/experiments/tp_student_rescreen_20260910'
TP = P / 'work/kvasir_tp_filterfirst_students_20260909'
OLD = P / 'work/kvasir_tp_student_mainline_20260907'
OUT = R / 'b7_new_student'

def read(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def jsonl(p, rows):
    Path(p).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

def mask(p):
    m = np.asarray(Image.open(p).convert('L')) > 127
    assert m.shape == (256, 256), (p, m.shape)
    return m

def dice(a, b):
    n = int(a.sum()) + int(b.sum())
    return 2 * int(np.logical_and(a, b).sum()) / n if n else 1.

def iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    return inter / max(int(a.sum()) + int(b.sum()) - inter, 1)

def paired(new, old):
    a = {r['target_id']: r for r in new}
    b = {r['target_id']: r for r in old}
    assert set(a) == set(b)
    delta = np.array([a[k]['dice'] - b[k]['dice'] for k in sorted(a)])
    rng = np.random.default_rng(2026)
    means = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return dict(mean_delta=float(delta.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                wins=int((delta > 1e-12).sum()), losses=int((delta < -1e-12).sum()),
                ties=int((abs(delta) <= 1e-12).sum()),
                changed_routes=sum(a[k]['route_id'] != b[k]['route_id'] for k in a))

def main():
    OUT.mkdir(exist_ok=False)
    source_results = json.loads((R / 'results.json').read_text())
    files = [OLD / 'code/mainline.py', R / 'results.json']
    for split in ['validation', 'test']:
        files.append(TP / f'quality/anchor_conditioned_target_pooling/propagation_quality_{split}/propagation_quality.jsonl')
        for variant in ['best', 'final']:
            files.append(R / f'runs/soft/{split}_{variant}/per_target_metrics.jsonl')
    for variant in ['best', 'final']:
        cp = Path(source_results[variant]['checkpoint'])
        assert sha(cp) == source_results[variant]['sha256']
        files.append(cp)
    config = dict(created=datetime.datetime.now().astimezone().isoformat(),
                  formula='(max(q_return,1e-6)*max(q_multi,1e-6)^2*max(q_model,1e-6)^2)^0.2',
                  q_multi='Mean pairwise Dice to other six TP candidates',
                  q_model='Dice against the frozen new student binary prediction',
                  tie_break=['b7_score', 'q_multi', 'q_return', 'route_id'],
                  hard_filter=False, output='Copy selected TP mask; no pixel fusion',
                  primary_checkpoint='best (selected by validation)',
                  final_checkpoint='Predeclared end-of-training comparison, not test selection',
                  input_hashes={str(p): sha(p) for p in files})
    save(OUT / 'frozen_config.json', config)
    all_results = {}
    # Freeze all four selections before loading any GT mask for evaluation.
    for split in ['validation', 'test']:
        qpath = TP / f'quality/anchor_conditioned_target_pooling/propagation_quality_{split}/propagation_quality.jsonl'
        clean = [{k: row[k] for k in ['target_id', 'route_id', 'bridge_count', 'q_cycle',
                                      'forward_mask_path', 'forward_mask_sha256']} for row in read(qpath)]
        groups = collections.defaultdict(list)
        for row in clean:
            groups[row['target_id']].append(row)
        assert len(clean) == 700 and len(groups) == 100
        for variant in ['best', 'final']:
            dest = OUT / f'{split}_{variant}'
            (dest / 'masks').mkdir(parents=True)
            students = {r['target_id']: {k: r[k] for k in ['mask_path', 'mask_sha256']}
                        for r in read(R / f'runs/soft/{split}_{variant}/per_target_metrics.jsonl')}
            assert set(students) == set(groups)
            scores, chosen = [], []
            for target, group in sorted(groups.items()):
                assert len(group) == 7 and {r['bridge_count'] for r in group} == set(range(7))
                assert all(sha(r['forward_mask_path']) == r['forward_mask_sha256'] for r in group)
                assert sha(students[target]['mask_path']) == students[target]['mask_sha256']
                mm = [mask(r['forward_mask_path']) for r in group]
                student = mask(students[target]['mask_path'])
                candidates = []
                for i, row in enumerate(group):
                    a = float(row['q_cycle'])
                    b = float(np.mean([dice(mm[i], m) for j, m in enumerate(mm) if i != j]))
                    c = dice(mm[i], student)
                    score = (max(a, 1e-6) * max(b, 1e-6)**2 * max(c, 1e-6)**2)**.2
                    candidates.append(dict(target_id=target, route_id=row['route_id'],
                                           bridge_count=row['bridge_count'], q_return=a, q_multi=b,
                                           q_model=c, b7_score=score, source_mask_path=row['forward_mask_path']))
                scores.extend(candidates)
                pick = dict(max(candidates, key=lambda r: (r['b7_score'], r['q_multi'], r['q_return'], r['route_id'])))
                out = dest / 'masks' / (target.replace('::', '__') + '.png')
                shutil.copy2(pick['source_mask_path'], out)
                assert sha(out) == sha(pick['source_mask_path'])
                pick.update(final_mask_path=str(out), mask_sha256=sha(out))
                chosen.append(pick)
            jsonl(dest / 'candidate_scores.jsonl', scores)
            jsonl(dest / 'selected_masks.jsonl', chosen)
            print('Frozen', split, variant, len(chosen), flush=True)
    frozen = {str(p): sha(p) for p in OUT.glob('*/selected_masks.jsonl')}
    save(OUT / 'SELECTIONS_FROZEN.json', frozen)
    for variant in ['best', 'final']:
        all_results[variant] = {}
        for split in ['validation', 'test']:
            dest = OUT / f'{split}_{variant}'
            rows = read(dest / 'selected_masks.jsonl')
            quality = read(TP / f'quality/anchor_conditioned_target_pooling/propagation_quality_{split}/propagation_quality.jsonl')
            index = {r['route_id']: r for r in quality}
            groups = collections.defaultdict(list)
            for r in quality:
                groups[r['target_id']].append(r)
            students = {r['target_id']: r for r in read(R / f'runs/soft/{split}_{variant}/per_target_metrics.jsonl')}
            student_metrics = []
            for row in rows:
                original = index[row['route_id']]
                gt = np.asarray(Image.open(original['target_mask_path_evaluation_only']).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 127
                pred = mask(row['final_mask_path'])
                d, j = dice(pred, gt), iou(pred, gt)
                assert abs(d - original['gt_dice_evaluation_only']) < 1e-12
                assert abs(j - original['gt_iou_evaluation_only']) < 1e-12
                row.update(dice=d, iou=j, oracle=max(x['gt_dice_evaluation_only'] for x in groups[row['target_id']]))
                sm = mask(students[row['target_id']]['mask_path'])
                student_metrics.append(dict(target_id=row['target_id'], dice=dice(sm, gt), iou=iou(sm, gt)))
            result = dict(count=len(rows), selected_dice=float(np.mean([r['dice'] for r in rows])),
                          selected_iou=float(np.mean([r['iou'] for r in rows])),
                          oracle=float(np.mean([r['oracle'] for r in rows])),
                          student_direct_dice=float(np.mean([r['dice'] for r in student_metrics])),
                          student_direct_iou=float(np.mean([r['iou'] for r in student_metrics])),
                          bridge_counts=dict(collections.Counter(r['bridge_count'] for r in rows)))
            assert abs(result['student_direct_dice'] - source_results[variant][split]['dice']) < 1e-12
            assert abs(result['student_direct_iou'] - source_results[variant][split]['iou']) < 1e-12
            if split == 'test':
                assert abs(result['oracle'] - .9074262547898724) < 1e-12
                result['versus_old448_b7'] = paired(rows, read(OLD / 'b7_test/per_target_metrics.jsonl'))
                result['versus_old580_b7'] = paired(rows, read(TP / 'b7_test/per_target_metrics.jsonl'))
                result['versus_tp_router_mean_delta'] = result['selected_dice'] - .8854326463411542
            jsonl(dest / 'per_target_metrics.jsonl', rows)
            jsonl(dest / 'student_direct_metrics.jsonl', student_metrics)
            save(dest / 'summary.json', result)
            all_results[variant][split] = result
            print(variant, split, json.dumps(result), flush=True)
    assert all(sha(p) == h for p, h in frozen.items())
    assert all(sha(p) == h for p, h in config['input_hashes'].items())
    save(OUT / 'results.json', all_results)
    save(OUT / 'completion_audit.json', dict(selections=400, candidates_scored=2800,
         all_selections_frozen_before_gt_evaluation=True, masks_copied_unchanged=True,
         source_metrics_reproduced=True, input_hashes_unchanged=True, oracle_unchanged=True))
    lines = ['# 新学生 B7 评估结果', '',
             '日期：2026-09-10。训练池为 8 张 GT + 620 张重新筛选的伪标签。评估完整 validation 100 张和 test 100 张。', '',
             '## 方法', '',
             '沿用原 B7 公式：`score=(max(R,1e-6) × max(C,1e-6)^2 × max(S,1e-6)^2)^(1/5)`。',
             'R 为返回一致性；C 为当前候选与其余 6 张 TP 候选的平均 Dice；S 为候选与冻结学生二值预测的 Dice。',
             '每图使用原 SAM3-base TP b0–b6 全部 7 张候选，不做训练池的 A/B 门槛筛选；取最高分并复制该 SAM3 mask。无像素融合。',
             '两个 checkpoint 的四组选择均在读取 GT mask 前冻结，GT 仅用于之后的指标计算。没有重新训练、改变权重或根据 test 调参。', '',
             '## 指标', '',
             '| 学生 checkpoint | Epoch | 学生直接 test Dice | B7 val Dice | B7 test Dice | B7 test IoU |',
             '|---|---:|---:|---:|---:|---:|']
    for v in ['best', 'final']:
        t, val = all_results[v]['test'], all_results[v]['validation']
        lines.append(f"| {v} | {source_results[v]['epoch']} | {t['student_direct_dice']:.6f} | {val['selected_dice']:.6f} | {t['selected_dice']:.6f} | {t['selected_iou']:.6f} |")
    lines += ['', 'best 按学生 validation 指标选出，是预先确定的主结果；final 是固定训练终点的补充评估，不按 test 反选 checkpoint。', '',
              '## 历史比较', '',
              '| 方法 | Test Dice | Test IoU |', '|---|---:|---:|',
              '| 原 448 张伪标签流程 B7 | 0.892639 | 0.834468 |',
              '| 原 580 张伪标签流程 B7 | 0.890846 | 0.831564 |',
              '| SAM3-base TP b0–b6 + 独立 Router | 0.885433 | 0.828274 |', '',
              '历史学生流程和本轮训练策略存在多项差异，因此只能比较结果，不能把差异全部归因于训练池大小。', '']
    for v in ['best', 'final']:
        t = all_results[v]['test']
        lines.append(f"### {v}")
        for key, label in [('versus_old448_b7', '原 448 B7'), ('versus_old580_b7', '原 580 B7')]:
            d = t[key]
            lines.append(f"- 相对{label}：Dice 差 {d['mean_delta']:+.6f}；逐图胜/负/平 {d['wins']}/{d['losses']}/{d['ties']}；更换候选 {d['changed_routes']}/100；配对 bootstrap 95% CI [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]（10000 次，seed=2026）。")
        lines.append(f"- 相对 TP 独立 Router：Dice 差 {t['versus_tp_router_mean_delta']:+.6f}。")
        lines.append(f"- 同一候选池 Oracle Dice {t['oracle']:.6f}，与实际 B7 相差 {t['oracle'] - t['selected_dice']:.6f}。Oracle 使用 GT，只用于分析上限。")
        lines.append('')
    lines += ['## 归档', '', f'实验目录：`{OUT}`。',
              '含输入哈希、固定公式、逐候选评分、冻结选择清单、400 张最终 mask、逐图 Dice/IoU、汇总和完整性核查。', '']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    shutil.copy2(OUT / 'report.md', P / 'new_project/new_b7_results.md')
    (OUT / 'COMPLETE').write_text(datetime.datetime.now().astimezone().isoformat())

if __name__ == '__main__':
    main()
