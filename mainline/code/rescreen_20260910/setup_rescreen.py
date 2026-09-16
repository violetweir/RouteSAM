"""Prepare an isolated replayable all-train re-screen experiment."""
import argparse
import json
from pathlib import Path
import shutil
import hashlib

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project')
BASE=P/'experiments/single_student_hard_soft_20260909'
R=P/'experiments/tp_student_rescreen_20260910'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main(bundle):
 assert not R.exists();R.mkdir();(R/'logs').mkdir();(R/'runs').mkdir();(R/'data').mkdir()
 shutil.copytree(BASE/'code',R/'code',ignore=shutil.ignore_patterns('__pycache__'))
 for name in ['setup_rescreen.py','rescreen_pool.py','test_rescreen.py','run_rescreen.py']:shutil.copy2(bundle/name,R/'code'/name)
 source=(BASE/'code/train_student.py').read_text();patched=source
 substitutions={
  "rows=read(root/'data/train_manifest.jsonl');assert len(rows)==588 and sum(x['is_gt'] for x in rows)==8":"rows=read(root/'data/train_manifest.jsonl');n=len(rows);assert n==cfg['train_count'] and sum(x['is_gt'] for x in rows)==8",
  "len(set(seen))==588 and gt_count==8":"len(set(seen))==n and gt_count==8",
  'pseudo_count=588-gt_count':'pseudo_count=n-gt_count',
  'assert total_steps==39984':"assert total_steps==cfg['total_iterations']",
 }
 for a,b in substitutions.items():assert patched.count(a)==1;patched=patched.replace(a,b)
 compile(patched,'train_student.py','exec');(R/'code/train_student.py').write_text(patched)
 shutil.copy2(BASE/'initial_model.pth',R/'initial_model.pth')
 shutil.copy2(BASE/'runs/soft/student_best.pth',R/'teacher.pth')
 best=json.loads((BASE/'runs/soft/student_best.json').read_text());base_results=json.loads((BASE/'results.json').read_text())
 assert sha(R/'teacher.pth')==base_results['soft']['checkpoint_sha256']
 save(R/'teacher.json',dict(source=str(BASE/'runs/soft/student_best.pth'),sha256=sha(R/'teacher.pth'),epoch=best['epoch'],validation_dice=best['dice'],selection='existing validation-best, not final checkpoint',frozen=True))
 cfg=json.loads((BASE/'config.json').read_text());cfg.update(reference_run=str(BASE),experiment='all792 TP plus frozen soft student re-screen',code_sha256={str(f.relative_to(R)):sha(f) for f in (R/'code').rglob('*.py')},test_policy='after training, freeze validation-best and predetermined epoch816 final; evaluate both once; best remains primary selection',dataset_size_policy='816 epochs maintained; iterations=816*ceil((N+8)/12); no padding, no replacement, no dropped final batch')
 assert sha(R/'initial_model.pth')==cfg['initial_checkpoint_sha256'];save(R/'config.json',cfg)
 save(R/'setup_audit.json',dict(training_changes=list(substitutions),source_trainer_sha256=sha(BASE/'code/train_student.py'),new_trainer_sha256=sha(R/'code/train_student.py'),same_initial_weights=True,teacher_frozen_by_validation=True))
 (R/'SETUP_COMPLETE').write_text('complete\n');print(str(R))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);args=parser.parse_args();main(args.bundle)
