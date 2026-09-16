"""Train one TP448 soft student on GPU0 and automatically test best/final."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import traceback

R=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/tp448_single_student_soft_20260912')
PY='/home/violet/anaconda3/envs/mkunet_mamba/bin/python'
def save(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def main():
 assert (R/'PREFLIGHT_OK').exists()
 (R/'STARTED').open('x').close()
 os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 cmd=[PY,'-u',str(R/'code/train_student.py'),'--root',str(R),'--variant','soft']
 save(R/'train_command.json',dict(argv=cmd,gpu=0))
 with (R/'logs/train_soft.log').open('w') as log:
  child=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  save(R/'status.json',dict(stage='training',gpu=0,pid=child.pid,epochs=816,iterations=31008))
  rc=child.wait()
  if rc:raise RuntimeError(f'training exited {rc}')
 sys.path.insert(0,str(R/'code'))
 import torch
 import train_student as t
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 run=R/'runs/soft';history=t.read(run/'train_epochs.jsonl');assert len(history)==816
 assert all(r['unique_images']==456 and r['gt_count']==8 and r['pseudo_count']==448 and r['steps_in_epoch']==38 for r in history)
 assert history[-1]['iteration']==31008
 summary=json.loads((run/'summary.json').read_text())
 frozen={v:dict(path=str(run/f'student_{v}.pth'),sha256=t.sha(run/f'student_{v}.pth')) for v in ['best','final']}
 save(R/'FROZEN_BEFORE_TEST.json',frozen);save(R/'status.json',dict(stage='evaluating_frozen_best_final'))
 results={}
 for v in ['best','final']:
  model=t.SamUnet(argparse.Namespace(in_channels=3,num_classes=2))
  model.load_state_dict(torch.load(frozen[v]['path'],map_location='cpu'));model=model.cuda().eval()
  val=t.evaluate(model,t.loader(t.StudentDataset(t.read(R/'data/validation_manifest.jsonl'),'hard',False),1,workers=1),run/f'validation_{v}')
  expected=summary['best_validation']['dice'] if v=='best' else summary['final_validation']['dice']
  assert abs(val['dice']-expected)<1e-10
  test=t.evaluate(model,t.loader(t.StudentDataset(t.read(R/'data/test_manifest.jsonl'),'hard',False),1,workers=1),run/f'test_{v}')
  assert t.sha(frozen[v]['path'])==frozen[v]['sha256']
  results[v]=dict(epoch=summary['best_validation']['epoch'] if v=='best' else 816,validation=val,test=test,checkpoint_sha256=frozen[v]['sha256'])
  del model;torch.cuda.empty_cache()
 for path,h in json.loads((R/'data/input_sha256.json').read_text()).items():assert t.sha(path)==h
 save(R/'results.json',results)
 lines=['# 原448张池＋新版单学生软标签训练结果','','GPU0，seed2026；448伪标签+8GT，逐图等权，816epoch=31008步。软目标=.75主mask+.25返回合格TP共识；与580张版本复用相同初始权重和共有图像软标签。','','|权重|Epoch|Val Dice|Test Dice|Test IoU|','|---|---:|---:|---:|---:|']
 for v,r in results.items():lines.append(f"|{v}|{r['epoch']}|{r['validation']['dice']:.6f}|{r['test']['dice']:.6f}|{r['test']['iou']:.6f}|")
 lines+=['','参考：580张新soft best Test=0.851243，final=0.859530；448张旧S3 best=0.860283，final=0.853202。固定epoch下池缩小，总步数由39984降为31008，因此不是固定优化步数的单因素比较。','',f'路径：`{R}`']
 (R/'report.md').write_text('\n'.join(lines)+'\n')
 save(R/'completion_audit.json',dict(epochs=816,total_steps=31008,all_epochs_456_unique_images=True,validation_reproduced=True,test_count_per_checkpoint=100,source_data_hashes_unchanged=True))
 save(R/'status.json',dict(stage='complete',results=results));(R/'COMPLETE').touch()

if __name__=='__main__':
 try:main()
 except BaseException:
  error=traceback.format_exc();(R/'FAILED.txt').write_text(error);save(R/'status.json',dict(stage='failed',error=error));raise
