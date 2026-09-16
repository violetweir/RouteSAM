"""Train the rebuilt pool and evaluate frozen best/final checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

PY='/home/violet/anaconda3/envs/mkunet_mamba/bin/python'
def save(p,x):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();r=args.root
 assert (r/'PREFLIGHT_OK').exists()
 with (r/'STARTED').open('x') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S %z')+'\n')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4');os.environ.update({k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
 try:
  with (r/'logs/train_student.log').open('w') as log:
   proc=subprocess.Popen([PY,'-u',str(r/'code/train_student.py'),'--root',str(r),'--variant','soft'],env=env,cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   save(r/'status.json',dict(stage='training',pid=proc.pid,started_at=time.strftime('%Y-%m-%d %H:%M:%S %z')));rc=proc.wait()
  if rc:raise RuntimeError(f'Training exited {rc}')
  import numpy as np
  import torch
  from train_student import read,sha,StudentDataset,loader,SamUnet,evaluate
  cfg=json.loads((r/'config.json').read_text());base=Path(cfg['reference_run']);d=r/'runs/soft'
  hist=read(d/'train_epochs.jsonl');assert len(hist)==816 and hist[-1]['iteration']==cfg['total_iterations']
  assert all(x['unique_images']==cfg['train_count'] and x['gt_count']==8 and x['pseudo_count']==cfg['pseudo_count'] and x['steps_in_epoch']==cfg['steps_per_epoch'] for x in hist)
  best=json.loads((d/'student_best.json').read_text());summary=json.loads((d/'summary.json').read_text())
  frozen={name:dict(checkpoint=str(d/filename),sha256=sha(d/filename),epoch=best['epoch'] if name=='best' else 816) for name,filename in [('best','student_best.pth'),('final','student_final.pth')]};save(r/'FROZEN_BEFORE_TEST.json',frozen)
  torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
  results={};save(r/'status.json',dict(stage='frozen_checkpoint_evaluation'))
  for name in ['best','final']:
   model=SamUnet(argparse.Namespace(in_channels=3,num_classes=2));model.load_state_dict(torch.load(frozen[name]['checkpoint'],map_location='cpu'));model=model.cuda().eval()
   val=evaluate(model,loader(StudentDataset(read(r/'data/validation_manifest.jsonl'),'hard',False),1,workers=1),d/f'validation_{name}')
   assert abs(val['dice']-(best['dice'] if name=='best' else summary['final_validation']['dice']))<1e-10
   test=evaluate(model,loader(StudentDataset(read(r/'data/test_manifest.jsonl'),'hard',False),1,workers=1),d/f'test_{name}')
   metrics=read(d/f'test_{name}/per_target_metrics.jsonl');reference={x['target_id']:x for x in read(base/f'runs/soft/test_{name}/per_target_metrics.jsonl')}
   delta=np.array([x['dice']-reference[x['target_id']]['dice'] for x in metrics]);rng=np.random.default_rng(2026);ci=np.quantile(delta[rng.integers(0,100,(10000,100))].mean(1),[.025,.975])
   for x in metrics:assert sha(x['mask_path'])==x['mask_sha256']
   assert sha(frozen[name]['checkpoint'])==frozen[name]['sha256']
   results[name]=dict(**frozen[name],validation=val,test=test,versus_original_soft_pool=dict(mean_delta=float(delta.mean()),ci95=ci.tolist(),wins=int((delta>1e-12).sum()),losses=int((delta< -1e-12).sum()),ties=int((abs(delta)<=1e-12).sum())))
   del model;torch.cuda.empty_cache()
  results['pool']=json.loads((r/'data/pool_summary.json').read_text());save(r/'results.json',results)
  save(r/'completion_audit.json',dict(epochs=816,iterations=cfg['total_iterations'],images_per_epoch=cfg['train_count'],all_epochs_no_replacement=True,GT_per_epoch=8,frozen_checkpoints_unchanged=True,validation_reproduced=True,test_masks_verified=200))
  report='\n'.join(['# TP + 单学生统一重筛：实验完成','','按冻结 A/B 条件审核全部 792 张训练图，不保留旧池特权；冻结软标签 validation-best 教师只参与准入。新标签仍为 .75M+.25合格TP共识。','',f'池变化：{results["pool"]}。',f'统一等权训练816 epoch，batch12，每轮{cfg["steps_per_epoch"]}步，共{cfg["total_iterations"]}步；复用上一阶段初始权重。','','| checkpoint | epoch | validation Dice | test Dice | 相比旧软标签池 test 差异 |','|---|---:|---:|---:|---:|',*[f'| {name} | {results[name]["epoch"]} | {results[name]["validation"]["dice"]:.6f} | {results[name]["test"]["dice"]:.6f} | {results[name]["versus_original_soft_pool"]["mean_delta"]:+.6f} |' for name in ['best','final']],'','validation-best 为主要模型选择规则；final 为预先固定的816epoch终点补充。二者均先冻结再评估test，不按test改筛选门槛。池大小变化导致总iter变化，因此本轮是固定epoch的重筛方案比较。'])+'\n'
  (r/'report.md').write_text(report);(r.parents[1]/'tp_student_rescreen_results.md').write_text(report)
  save(r/'status.json',dict(stage='complete',results=results));(r/'COMPLETE').write_text('complete\n')
 except BaseException:
  err=traceback.format_exc();(r/'FAILED.txt').write_text(err);save(r/'status.json',dict(stage='failed',error=err));raise
if __name__=='__main__':main()
