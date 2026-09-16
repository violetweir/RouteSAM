"""User-requested frozen S2/S3 best/final test audit; no training or selection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'work/kvasir_tp_filterfirst_students_20260909'
OUT=R/'s2_s3_test_audit'
PY='/home/violet/anaconda3/envs/mkunet_mamba/bin/python'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
OUT.mkdir(exist_ok=True)
gt={x['target_id']:x['target_mask_path_evaluation_only'] for x in read(R/'quality/anchor_conditioned_target_pooling/propagation_quality_test/propagation_quality.jsonl')}
assert len(gt)==100
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',SC_SAM_ROOT=str(P/'third_party/SC-SAM'))
results=[]
for student in ['S2','S3']:
 for variant,filename in [('valbest','student_best.pth'),('final','student_final.pth')]:
  name=student+'_'+variant;ckpt=R/'students'/student/filename;digest=sha(ckpt);dest=OUT/name
  vals=read(R/'students'/student/'validation.jsonl')
  selected=max(vals,key=lambda x:x['dice']) if variant=='valbest' else vals[-1]
  print('EXPORT',name,flush=True)
  cmd=[PY,'-u',str(R/'code/cuda_runner.py'),str(R/'code/export_t25_student_predictions.py'),'--run-dir',str(R/'students'/student),'--checkpoint',str(ckpt),'--output-root',str(dest),'--splits','test']
  with (OUT/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  pred=read(dest/'student_predictions_test.jsonl');assert len(pred)==100 and {x['merged_id'] for x in pred}==set(gt)
  metrics=[]
  for row in pred:
   a=np.asarray(Image.open(row['student_binary_mask']).convert('L'))>127
   b=np.asarray(Image.open(gt[row['merged_id']]).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
   assert a.shape==b.shape==(256,256)
   inter=int((a&b).sum());total=int(a.sum())+int(b.sum());union=total-inter
   metrics.append(dict(target_id=row['merged_id'],dice=2*inter/total if total else 1.,iou=inter/union if union else 1.,mask_sha256=sha(Path(row['student_binary_mask']))))
  assert sha(ckpt)==digest
  (dest/'per_target_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in metrics))
  result=dict(student=student,checkpoint_type=variant,iteration=selected['iteration'],validation_dice_training_entry=selected['dice'],test_dice=float(np.mean([x['dice'] for x in metrics])),test_iou=float(np.mean([x['iou'] for x in metrics])),count=100,checkpoint=str(ckpt),checkpoint_sha256=digest)
  results.append(result);save(OUT/'results.json',results);print(json.dumps(result),flush=True)
lines=['# 当前 580 张版本：S2/S3 冻结 checkpoint 的 test 评估','','按用户要求评估既有 validation-best 与 final-40000，不训练、不按 test 选择模型。使用与 X3 单图结果相同的导出和 Dice 口径：256×256、前景概率阈值 0.5、逐图 Dice 后对完整 100 张求平均。','','| 模型 | checkpoint | 步数 | validation Dice（训练入口） | test Dice | test IoU |','|---|---|---:|---:|---:|---:|']
for x in results:lines.append(f"| {x['student']} | {x['checkpoint_type']} | {x['iteration']} | {x['validation_dice_training_entry']:.6f} | {x['test_dice']:.6f} | {x['test_iou']:.6f} |")
lines+=['','validation 列来自训练日志；test 列统一采用已有导出入口。四个 checkpoint 与全部 400 张输出 mask 均记录 SHA256。既有 X3-best test Dice 为 0.853667。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
(P/'reproduction_reports/Kvasir_TP580_S2_S3_test_checkpoints_20260909.md').write_text('\n'.join(lines)+'\n')
(OUT/'COMPLETE').write_text('complete\n')
