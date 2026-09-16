"""Paired descriptive uncertainty after the fixed experiment has completed."""
import json
from pathlib import Path
import numpy as np

r=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_tp_guided_pc_joint_20260907')
assert (r/'COMPLETE').exists()
variants=['tp_baseline','joint_v1','joint_v2']
rows={v:{x['target_id']:x for x in [json.loads(l) for l in (r/v/'test_per_target_metrics.jsonl').read_text().splitlines()]} for v in variants}
ids=sorted(rows['tp_baseline']);assert len(ids)==100
assert all(set(rows[v])==set(ids) for v in variants)
rng=np.random.default_rng(2026)
bootstrap_ids=rng.integers(0,len(ids),size=(10000,len(ids)))
result={}
for v in variants[1:]:
    delta=np.array([rows[v][i]['dice']-rows['tp_baseline'][i]['dice'] for i in ids])
    result[v]={'paired_dice_delta':float(delta.mean()),'bootstrap_95_percentile_ci':np.quantile(delta[bootstrap_ids].mean(1),[.025,.975]).tolist(),'improved':int((delta>1e-12).sum()),'tied':int((abs(delta)<=1e-12).sum()),'worse':int((delta< -1e-12).sum())}
(r/'paired_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['','## 配对 Test 差异分析','', '以同一张 test 图为配对单位，与 TP baseline 比较；固定 seed=2026、10000 次图像 bootstrap。区间仅描述当前 100 张 test 的抽样不确定性，不用于调参。','', '| 方法 | Dice 差值 | 95% bootstrap 区间 | 改善/持平/下降图数 |','|---|---:|---:|---:|']
for v,d in result.items():
    lo,hi=d['bootstrap_95_percentile_ci']
    lines.append(f"| {v} | {d['paired_dice_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {d['improved']}/{d['tied']}/{d['worse']} |")
lines+=['','## Anchor 分布诊断','', '| 方法 | Test 700 候选中最多的 anchor 占比 | 最终 100 masks 中最多的 anchor 占比 |','|---|---:|---:|']
for v in variants:
    audit=json.loads((r/v/'route_audit_test.json').read_text())
    s=json.loads((r/v/'test_summary.json').read_text())
    lines.append(f"| {v} | {max(audit['anchor_counts'].values())/700:.2%} | {max(s['anchor_counts'].values())/100:.2%} |")
append='\n'.join(lines)+'\n'
for path in [r/'report.md',r.parents[1]/'reproduction_reports/Kvasir_TP_guided_PC_joint_20260907.md']:
    text=path.read_text()
    if '## 配对 Test 差异分析' not in text:path.write_text(text+append)
print(json.dumps(result,indent=2))
