from pathlib import Path
import json,time,numpy as np
R=Path(__file__).resolve().parent
B=R.parent/'tn3k_busi_factorial_20260915'
while not (R/'anchors_COMPLETE').exists():
 time.sleep(15)
rows=json.loads((R/'anchors_results.json').read_text());cal=json.loads((B/'calibration_frozen.json').read_text());groups=json.loads((B/'pool_membership_frozen.json').read_text())['groups']['validation'];metrics=json.loads((R/'validation_candidate_metrics.json').read_text())
per=[]
for r in rows:
 tid=r['id'];m=r['results'];aids=sorted(m)
 raw=sorted(aids,key=lambda a:(-cal['scores'][tid][a]['anchor_target_raw'],a))
 centered=sorted(aids,key=lambda a:(-cal['scores'][tid][a]['anchor_target_centered'],a))
 x={'id':tid,'all23_b0_oracle':max(v['dice'] for v in m.values())}
 for name,order in [('raw',raw),('centered',centered)]:
  x[name+'_top1_b0']=m[order[0]]['dice'];x[name+'_top2_b0_oracle']=max(m[a]['dice'] for a in order[:2])
 x['old_centered_top2_b0_b6_oracle']=max(metrics[a]['dice'] for a in groups['centered_top2'][tid]);per.append(x)
means={k:float(np.mean([r[k] for r in per])) for k in per[0] if k!='id'}
result={'n':len(per),'means':means,'all23_b0_low_count':sum(r['all23_b0_oracle']<.5 for r in per),'per_target':per,'warning':'Validation random64; oracle uses GT for diagnosis only. All23 b0 does not equal all23 b0-b6.'}
(R/'all_anchor_summary.json').write_text(json.dumps(result,indent=2))
with (R/'诊断报告.md').open('a') as f:
 f.write('\n## 全23参考b0诊断完成\n\n以下结果为同一64张验证图，全部23参考仅运行b0，没有新增标注。Oracle只能用于诊断。\n\n| 指标 | Dice |\n|---|---:|\n')
 for k,v in means.items():f.write(f'| {k} | {v:.6f} |\n')
 f.write(f"\n全部23张参考的b0候选仍全部低于0.5：{result['all23_b0_low_count']}/64。详细结果见all_anchor_summary.json。\n")
(R/'COMPLETE').touch()
