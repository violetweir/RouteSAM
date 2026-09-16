"""Two concurrent GPU0 workers evaluate frozen TP448 S2/S3 best/final."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'work/kvasir_tp_student_mainline_20260907'
OUT=P/'new_project/experiments/tp448_s2_s3_test_20260912'
PY='/home/violet/anaconda3/envs/mkunet_mamba/bin/python'

def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def worker(student):
    frozen=json.loads((OUT/'FROZEN_BEFORE_TEST.json').read_text())
    gt={r['target_id']:r['target_mask_path_evaluation_only'] for r in read(R/'quality/anchor_conditioned_target_pooling/propagation_quality_test/propagation_quality.jsonl')}
    assert len(gt)==100
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',SC_SAM_ROOT=str(P/'third_party/SC-SAM'),PYTHONUNBUFFERED='1')
    results=[]
    for variant,filename in [('best','student_best.pth'),('final','student_final.pth')]:
        ckpt=R/'students'/student/filename
        assert sha(ckpt)==frozen['checkpoint_sha256'][str(ckpt)]
        dest=OUT/student/variant;dest.mkdir(parents=True,exist_ok=False)
        cmd=[PY,'-u',str(OUT/'cuda_runner.py'),str(R/'code/export_t25_student_predictions.py'),'--run-dir',str(R/'students'/student),'--checkpoint',str(ckpt),'--output-root',str(dest),'--splits','test']
        save(dest/'command.json',dict(argv=cmd,gpu=0))
        with (dest/'export.log').open('w') as log:
            subprocess.run(cmd,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        pred=read(dest/'student_predictions_test.jsonl')
        assert len(pred)==100 and {r['merged_id'] for r in pred}==set(gt)
        metrics=[]
        for row in pred:
            a=np.asarray(Image.open(row['student_binary_mask']).convert('L'))>127
            b=np.asarray(Image.open(gt[row['merged_id']]).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
            assert a.shape==b.shape==(256,256) and b.any()
            inter=int((a&b).sum());total=int(a.sum())+int(b.sum())
            metrics.append(dict(target_id=row['merged_id'],dice=2*inter/total,iou=inter/(total-inter),mask_sha256=sha(Path(row['student_binary_mask']))))
        assert sha(ckpt)==frozen['checkpoint_sha256'][str(ckpt)]
        vals=read(R/'students'/student/'validation.jsonl')
        val=max(vals,key=lambda r:r['dice']) if variant=='best' else vals[-1]
        result=dict(student=student,variant=variant,iteration=val['iteration'],validation_dice=val['dice'],test_dice=float(np.mean([r['dice'] for r in metrics])),test_iou=float(np.mean([r['iou'] for r in metrics])),count=100,checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt))
        save(dest/'per_target_metrics.json',metrics);save(dest/'result.json',result)
        results.append(result);print(json.dumps(result),flush=True)
    save(OUT/student/'results.json',results)
    (OUT/student/'COMPLETE').touch()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--student',choices=['S2','S3']);args=ap.parse_args()
    if args.student:
        worker(args.student);return
    OUT.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copy2(__file__,OUT/'evaluate_tp448_students.py')
    (OUT/'cuda_runner.py').write_text("import sys,runpy,torch\ntorch.set_num_threads(4)\ntorch.cuda.set_per_process_memory_fraction(.25)\nsys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n")
    checkpoints={}
    for student in ['S2','S3']:
        run=R/'students'/student
        config=json.loads((run/'protocol.json').read_text())
        summary=json.loads((run/'summary.json').read_text())
        assert config['pseudo_count']==448 and config['labeled_count']==8
        assert summary['final_iteration']==40000 and (run/'TRAINING_COMPLETE').exists()
        assert len(read(Path(config['pseudo_manifest'])))==448
        for name in ['student_best.pth','student_final.pth']:
            checkpoints[str(run/name)]=sha(run/name)
    save(OUT/'FROZEN_BEFORE_TEST.json',dict(training=False,gpu=0,parallel_students=['S2','S3'],checkpoint_sha256=checkpoints,source_root=str(R),metric_resolution=256,pseudo_count=448,gt_count=8))
    processes=[]
    for student in ['S2','S3']:
        cmd=[PY,'-u',str(OUT/'evaluate_tp448_students.py'),'--student',student]
        with (OUT/(student+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        processes.append((student,child));save(OUT/(student+'.process.json'),dict(pid=child.pid,argv=cmd,gpu=0))
    failures=[]
    for student,child in processes:
        rc=child.wait()
        if rc:failures.append((student,rc))
    if failures:raise RuntimeError(failures)
    results=[x for student in ['S2','S3'] for x in json.loads((OUT/student/'results.json').read_text())]
    for ckpt,digest in checkpoints.items():assert sha(Path(ckpt))==digest
    save(OUT/'results.json',results)
    lines=['# 448张 TP-only：S2/S3 独立 test 补测','','2026-09-12。找到原完整40000步权重，未重新训练。S2/S3两个任务在GPU0并行，每个任务依次测试best/final。原数据、权重只读，输出独立归档。','','完整test100，RGB/GT尺寸256，原导出入口、前景概率>=0.5、逐图Dice/IoU宏平均。validation列来自训练记录；不按test反选checkpoint。','','|学生|权重|迭代|训练Val Dice|Test Dice|Test IoU|','|---|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['student']}|{r['variant']}|{r['iteration']}|{r['validation_dice']:.6f}|{r['test_dice']:.6f}|{r['test_iou']:.6f}|")
    lines+=['','四个checkpoint前后SHA256相同，400张test预测及对应GT身份、尺寸和mask哈希已检查。','',f'服务器目录：`{OUT}`。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'COMPLETE').touch();print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():(OUT/f'FAILED_{os.getpid()}.txt').write_text(traceback.format_exc())
        raise
