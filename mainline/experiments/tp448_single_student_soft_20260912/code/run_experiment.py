"""Run paired uniform-epoch students, then freeze and test both once."""
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
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 os.environ.update({k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
 children={};logs=[]
 try:
  for variant in ['hard','soft']:
   log=(r/'logs'/f'train_{variant}.log').open('w');logs.append(log)
   proc=subprocess.Popen([PY,'-u',str(r/'code/train_student.py'),'--root',str(r),'--variant',variant],env=env,cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   children[variant]=proc
  save(r/'status.json',dict(stage='training',pids={k:p.pid for k,p in children.items()},started_at=time.strftime('%Y-%m-%d %H:%M:%S %z')))
  while True:
   states={k:p.poll() for k,p in children.items()}
   if any(code is not None and code!=0 for code in states.values()):raise RuntimeError(f'Training failed: {states}; see per-run logs')
   if all(code==0 for code in states.values()):break
   time.sleep(10)
  for f in logs:f.close()
  import numpy as np
  import torch
  from train_student import read,sha,StudentDataset,loader,SamUnet,evaluate
  torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
  histories={k:read(r/f'runs/{k}/train_epochs.jsonl') for k in children}
  assert all(len(x)==816 for x in histories.values())
  for a,b in zip(histories['hard'],histories['soft']):
   assert a['order_sha256']==b['order_sha256'] and a['gt_count']==b['gt_count']==8 and a['unique_images']==b['unique_images']==588 and a['steps_in_epoch']==b['steps_in_epoch']==49
  frozen={k:dict(checkpoint=str(r/f'runs/{k}/student_best.pth'),sha256=sha(r/f'runs/{k}/student_best.pth'),selection=json.loads((r/f'runs/{k}/student_best.json').read_text())) for k in children}
  save(r/'FROZEN_BEFORE_TEST.json',frozen);save(r/'status.json',dict(stage='frozen_checkpoint_evaluation'))
  results={}
  for variant in ['hard','soft']:
   model=SamUnet(argparse.Namespace(in_channels=3,num_classes=2));model.load_state_dict(torch.load(frozen[variant]['checkpoint'],map_location='cpu'));model=model.cuda().eval()
   val=evaluate(model,loader(StudentDataset(read(r/'data/validation_manifest.jsonl'),'hard',False),1,workers=1),r/f'runs/{variant}/validation_best')
   assert abs(val['dice']-frozen[variant]['selection']['dice'])<1e-10
   test=evaluate(model,loader(StudentDataset(read(r/'data/test_manifest.jsonl'),'hard',False),1,workers=1),r/f'runs/{variant}/test_best')
   assert sha(frozen[variant]['checkpoint'])==frozen[variant]['sha256']
   results[variant]=dict(best_epoch=frozen[variant]['selection']['epoch'],validation=val,test=test,checkpoint_sha256=frozen[variant]['sha256']);del model;torch.cuda.empty_cache()
  hard={x['target_id']:x for x in read(r/'runs/hard/test_best/per_target_metrics.jsonl')};soft=read(r/'runs/soft/test_best/per_target_metrics.jsonl')
  delta=np.array([x['dice']-hard[x['target_id']]['dice'] for x in soft]);rng=np.random.default_rng(2026);ci=np.quantile(delta[rng.integers(0,100,(10000,100))].mean(1),[.025,.975])
  results['paired_test_soft_minus_hard']=dict(mean=float(delta.mean()),ci95=ci.tolist(),wins=int((delta>1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()),losses=int((delta< -1e-12).sum()))
  save(r/'results.json',results);save(r/'completion_audit.json',dict(epochs_each=816,iterations_each=39984,all_epoch_orders_match=True,GT_per_epoch=8,pseudo_per_epoch=580,validation_checkpoint_reproduced=True,frozen_checkpoints_unchanged=True,test_count_each=100))
  report='\n'.join(['# 单学生等权训练：硬标签与合格候选软标签对照','','588 张图统一随机采样，每轮不放回；batch=12，816 epoch=39,984 iter；所有样本及像素等权，统一逐图 BCE + Soft Dice。两组共用初始权重、每轮顺序和逐图增强随机种子。软标签 Y=.75M+.25P，P 只平均返回合格的 TP 候选。','','| 版本 | 最佳 epoch | validation Dice | test Dice | test IoU |','|---|---:|---:|---:|---:|',*[f'| {k} | {results[k]["best_epoch"]} | {results[k]["validation"]["dice"]:.6f} | {results[k]["test"]["dice"]:.6f} | {results[k]["test"]["iou"]:.6f} |' for k in ['hard','soft']],'',f'配对 test 差异（soft-hard）：{results["paired_test_soft_minus_hard"]}。','','两个模型均完成全部训练，固定各自 validation-best 后才进行一次 test 评估。本轮为单 seed 对照；两组统一采样及损失设置均不同于历史 S2/S3，不应将与旧实验的差异只归因于标签软化。'])+'\n'
  (r/'report.md').write_text(report);(r.parents[1]/'single_student_experiment_results.md').write_text(report)
  save(r/'status.json',dict(stage='complete',results=results));(r/'COMPLETE').write_text('complete\n')
 except BaseException:
  for p in children.values():
   if p.poll() is None:p.terminate()
  error=traceback.format_exc();(r/'FAILED.txt').write_text(error);save(r/'status.json',dict(stage='failed',error=error));raise
if __name__=='__main__':main()
