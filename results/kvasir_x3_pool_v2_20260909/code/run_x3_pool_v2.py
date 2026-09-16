"""Isolated corrected teachers -> unified pool -> X3 validation; no test runs."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
SRC=P/'work/kvasir_tp_filterfirst_students_20260909'
R=P/'work/kvasir_x3_pool_v2_20260909'
PY='/home/violet/anaconda3/envs/mkunet_mamba/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def event(stage,status,**extra):
 row=dict(time=datetime.datetime.now().astimezone().isoformat(),stage=stage,status=status,**extra);save(R/'status.json',row)
 with (R/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def command(stage,script,args):
 event(stage,'running');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',SC_SAM_ROOT=str(P/'third_party/SC-SAM'))
 for k in ['ISIC_LIGHT_STRONG_AUG','ISIC_RESIZED_CACHE_ROOT']:env.pop(k,None)
 gpu=script in ['run_t24_student.py','export_t25_student_predictions.py']
 cmd=[PY,'-u']+([str(R/'code/cuda_runner.py')] if gpu else [])+[str(R/'code'/script),*map(str,args)]
 save(R/'stages'/f'{stage}.command.json',dict(argv=cmd,gpu=1 if gpu else None))
 with (R/'logs'/f'{stage}.log').open('w') as log:
  proc=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);save(R/'active_process.json',dict(stage=stage,pid=proc.pid));rc=proc.wait()
 if rc:raise RuntimeError(f'{stage} failed ({rc}), see logs')
 (R/'stages'/f'{stage}.complete').write_text('complete\n');event(stage,'complete')
def setup(bundle):
 assert not R.exists();R.mkdir();(R/'logs').mkdir();(R/'stages').mkdir()
 shutil.copytree(SRC/'code',R/'code',ignore=shutil.ignore_patterns('__pycache__'))
 for name in ['run_x3_pool_v2.py','supervision_batch.py','test_x3_pool_v2.py','build_x3_pool_v2.py']:shutil.copy2(bundle/name,R/'code'/name)
 original=(SRC/'code/run_t24_student.py').read_text();patched=original
 patched=patched.replace('from probability_io import load_probability','from probability_io import load_probability\nfrom supervision_batch import align_supervision_batch')
 patched=patched.replace('choices=("S0", "S1", "S2", "S3")','choices=("S0", "S1", "S2", "S3", "X3")')
 patched=patched.replace('    return parser.parse_args()','    parser.add_argument("--quality-weight-mode", choices=("normalized_q_multi", "manifest"), default="normalized_q_multi")\n    return parser.parse_args()',1)
 key='            normalized = (q - q_min) / max(q_max - q_min, 1e-12)'
 patched=patched.replace(key,key+'\n            weight = max(0.2, min(1.0, normalized))\n            if getattr(args, "quality_weight_mode", "normalized_q_multi") == "manifest":\n                if row.get("preview_only"):\n                    raise ValueError("Preview pools cannot train official X3")\n                weight = float(row["explicit_quality_weight"])\n                if not math.isfinite(weight) or not 0 < weight <= 1:\n                    raise ValueError("Invalid absolute manifest weight")')
 patched=patched.replace('"quality_weight": max(0.2, min(1.0, normalized)),','"quality_weight": weight,')
 key='            image = batch["image"].cuda(non_blocking=True)'
 assert patched.count(key)==1
 patched=patched.replace(key,'            expected_labeled = len(batch["is_labeled"]) if args.experiment == "S0" else args.labeled_bs\n            batch = align_supervision_batch(batch, expected_labeled)\n'+key)
 patched=patched.replace('        entropy_weight=0.0,','        entropy_weight=0.0,\n        supervision_partition="explicit is_labeled membership, all fields aligned before loss",')
 patched=patched.replace('else "normalized_q_multi"','else args.quality_weight_mode')
 patched=patched.replace('quality_normalization="fixed over accepted pseudo labels; clip to [0.2,1.0]",','quality_normalization=("absolute manifest weights; no minmax rescaling" if args.quality_weight_mode == "manifest" else "fixed over accepted pseudo labels; clip to [0.2,1.0]"),')
 patched=patched.replace('                    "lr": lr,','                    "lr": lr,\n                    "supervised_labeled_count": int(batch["is_labeled"][:labeled_bs].sum()),\n                    "pseudo_labeled_count": int(batch["is_labeled"][labeled_bs:].sum()),')
 assert patched!=original and 'align_supervision_batch(batch' in patched
 compile(patched,'run_t24_student.py','exec');(R/'code/run_t24_student.py').write_text(patched)
 save(R/'config.json',dict(created_at=datetime.datetime.now().astimezone().isoformat(),source_run=str(SRC),teacher_pool=580,teacher_fix='align all batch fields by is_labeled before computing sliced supervised and pseudo losses',teacher_training='same frozen 580 hard/soft pools, same seed and 40k schedule; only supervision partition fix',x3='new uniform quality-driven pool; 6 GT + 6 pseudo; manifest absolute weights; unified soft loss; fresh initialization',scope='fixed S2/S3 -> all-train exports -> unified 792-target pool -> X3 40k -> validation only',test_evaluated=False,source_code_sha256=sha(SRC/'code/run_t24_student.py'),fixed_code_sha256=sha(R/'code/run_t24_student.py')))
 command('regression_tests','test_x3_pool_v2.py',[])
 (R/'PREPARED').write_text('prepared\n')
def export(name,stage,ckpt,splits):command('export_'+name+'_'+'_'.join(splits),'export_t25_student_predictions.py',['--run-dir',R/'students'/stage,'--checkpoint',R/'students'/stage/ckpt,'--output-root',R/'predictions'/name,'--splits',*splits])
def run():
 assert (R/'PREPARED').exists()
 with (R/'STARTED').open('x') as f:f.write('started\n')
 try:
  common=['--data-path',P/'work/kvasir_1pct_anchors/baseline_data','--labeled-list',SRC/'protocol/frozen_labeled_images.txt','--seed',2026,'--max-iterations',40000,'--val-interval',200,'--num-workers',4,'--defer-test']
  for stage,manifest in [('S2',SRC/'pseudo_manifest_original.jsonl'),('S3',SRC/'S3_consensus/pseudo_consensus.jsonl')]:
   command('train_'+stage,'run_t24_student.py',[*common,'--pseudo-manifest',manifest,'--output-dir',R/'students'/stage,'--experiment',stage])
   for suffix,ckpt in [('valbest','student_best.pth'),('final','student_final.pth')]:export(stage+'_'+suffix,stage,ckpt,['train'])
  command('build_unified_pool','build_x3_pool_v2.py',['--committee-root',R,'--output-root',R/'pool'])
  summary=json.loads((R/'pool/summary.json').read_text());assert summary['accepted']>=6
  command('train_X3','run_t24_student.py',[*common,'--pseudo-manifest',R/'pool/pseudo_manifest_x3.jsonl','--output-dir',R/'students/X3','--experiment','X3','--quality-weight-mode','manifest'])
  export('X3_best','X3','student_best.pth',['validation'])
  vals={name:max(read(R/f'students/{name}/validation.jsonl'),key=lambda x:x['dice']) for name in ['S2','S3','X3']}
  result=dict(pool=summary,best_validation=vals,test_evaluated=False,checkpoint_sha256=sha(R/'students/X3/student_best.pth'))
  save(R/'results.json',result)
  report='\n'.join(['# X3 统一质量训练池 v2：validation 阶段完成','','已先修复 S2/S3 batch 监督错位并重新训练，再基于修正后的四个快照审核全部 792 张训练图。新 X3 使用统一软标签、绝对图像权重及 6 GT + 6 伪标签采样。','',f'训练池：{summary["tiers"]}，接纳 {summary["accepted"]} 张。','','| 模型 | 最佳 validation Dice（训练入口） | 步数 |','|---|---:|---:|',*[f'| {name} | {row["dice"]:.6f} | {row["iteration"]} |' for name,row in vals.items()],'','本轮没有评估 test。旧结果存在监督错位，且本轮同时改变 X3 池与训练配方，因此新旧差异不能归因于单一因素。后续消融应基于修正后的教师固定比较。'])+'\n'
  (R/'report.md').write_text(report);(P/'reproduction_reports/Kvasir_X3_pool_v2_validation_20260909.md').write_text(report)
  event('pipeline','complete',validation_dice=vals['X3']['dice']);(R/'COMPLETE').write_text('complete\n')
 except BaseException:
  err=traceback.format_exc();(R/'FAILED.txt').write_text(err);event('pipeline','failed',error=err);raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['setup','run']);parser.add_argument('--bundle',type=Path);args=parser.parse_args()
 if args.action=='setup':setup(args.bundle)
 else:run()
