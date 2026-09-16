"""Evaluate legacy S2/S3 or uniform-epoch students, preserving old outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--family',choices=['legacy','single'],required=True)
 ap.add_argument('--run-root',type=Path,required=True)
 ap.add_argument('--students',nargs='+',required=True)
 ap.add_argument('--output-dir',type=Path,required=True)
 ap.add_argument('--gpu',default='0')
 args=ap.parse_args();r=args.run_root;o=args.output_dir
 assert set(args.students)<=({'S2','S3'} if args.family=='legacy' else {'hard','soft'})
 os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',SC_SAM_ROOT=str(P/'third_party/SC-SAM'))
 o.mkdir(parents=True,exist_ok=False)
 frozen={}
 for student in args.students:
  for kind in ['best','final']:
   ck=r/('students' if args.family=='legacy' else 'runs')/student/f'student_{kind}.pth'
   frozen[f'{student}_{kind}']=dict(path=str(ck),sha256=sha(ck))
 save(o/'FROZEN_BEFORE_TEST.json',frozen)
 results=[]
 if args.family=='legacy':
  wrapper=o/'cuda_runner.py'
  wrapper.write_text("import sys,runpy,torch\ntorch.set_num_threads(4)\ntorch.cuda.set_per_process_memory_fraction(.25)\nsys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n")
  for student in args.students:
   run=r/'students'/student;cfg=json.loads((run/'protocol.json').read_text());data=Path(cfg['data_path'])/'test'
   meta={x['merged_id']:x for x in read(data/'metadata.jsonl')};assert len(meta)==100
   vals=read(run/'validation.jsonl')
   for kind in ['best','final']:
    dest=o/f'{student}_{kind}';dest.mkdir();entry=frozen[f'{student}_{kind}']
    cmd=[sys.executable,'-u',str(wrapper),str(r/'code/export_t25_student_predictions.py'),'--run-dir',str(run),'--checkpoint',entry['path'],'--output-root',str(dest),'--splits','test']
    save(dest/'command.json',cmd)
    with (dest/'export.log').open('w') as log:subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=read(dest/'student_predictions_test.jsonl');assert len(rows)==100 and {x['merged_id'] for x in rows}==set(meta)
    values=[]
    for row in rows:
     g=Path(meta[row['merged_id']]['mask_file_name']);g=g if g.is_absolute() else data/g
     a=np.asarray(Image.open(row['student_binary_mask']).convert('L'))>127
     b=np.asarray(Image.open(g).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
     assert a.shape==b.shape==(256,256) and b.any()
     i=int((a&b).sum());n=int(a.sum())+int(b.sum())
     values.append(dict(target_id=row['merged_id'],dice=2*i/n,iou=i/(n-i),mask_sha256=sha(row['student_binary_mask'])))
    save(dest/'per_target_metrics.json',values)
    val=max(vals,key=lambda x:x['dice']) if kind=='best' else vals[-1]
    results.append(dict(student=student,kind=kind,iteration=val['iteration'],validation_dice_training_entry=val['dice'],test_dice=float(np.mean([x['dice'] for x in values])),test_iou=float(np.mean([x['iou'] for x in values])),checkpoint_sha256=entry['sha256']))
 else:
  sys.path.insert(0,str(r/'code'))
  import torch
  import train_student as t
  torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
  for student in args.students:
   summary=json.loads((r/'runs'/student/'summary.json').read_text())
   for kind in ['best','final']:
    dest=o/f'{student}_{kind}';entry=frozen[f'{student}_{kind}']
    model=t.SamUnet(argparse.Namespace(in_channels=3,num_classes=2))
    model.load_state_dict(torch.load(entry['path'],map_location='cpu'));model=model.cuda().eval()
    val=t.evaluate(model,t.loader(t.StudentDataset(t.read(r/'data/validation_manifest.jsonl'),'hard',False),1,workers=1),dest/'validation')
    expected=summary['best_validation']['dice'] if kind=='best' else summary['final_validation']['dice']
    assert abs(val['dice']-expected)<1e-10
    test=t.evaluate(model,t.loader(t.StudentDataset(t.read(r/'data/test_manifest.jsonl'),'hard',False),1,workers=1),dest/'test')
    results.append(dict(student=student,kind=kind,epoch=summary['best_validation']['epoch'] if kind=='best' else summary['epochs'],validation_dice=val['dice'],test_dice=test['dice'],test_iou=test['iou'],checkpoint_sha256=entry['sha256']))
    del model;torch.cuda.empty_cache()
 for entry in frozen.values():assert sha(entry['path'])==entry['sha256']
 save(o/'results.json',results);(o/'COMPLETE').touch();print(json.dumps(results,indent=2))

if __name__=='__main__':main()
