"""Controlled TP filter-before-router student rerun, isolated from 448 baseline."""
import argparse
import collections
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

import numpy as np

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
OLD=P/'work/kvasir_tp_student_mainline_20260907'
DIAG=P/'work/kvasir_tp_pseudo_filter_20260909'
R=P/'work/kvasir_tp_filterfirst_students_20260909'
PY='/home/violet/anaconda3/envs/mkunet_mamba/bin/python'

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def jsonl(p,rows):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def stamp():return datetime.datetime.now().astimezone().isoformat()
def event(stage,state,**kw):
    row=dict(time=stamp(),stage=stage,status=state,**kw);save(R/'status.json',row)
    with (R/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def command(stage,script,args):
    marker=R/'stages'/f'{stage}.complete'
    if marker.exists():return
    event(stage,'running')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',SC_SAM_ROOT=str(P/'third_party/SC-SAM'))
    for key in ['ISIC_LIGHT_STRONG_AUG','ISIC_RESIZED_CACHE_ROOT']:env.pop(key,None)
    is_cuda=script in ['run_t24_student.py','run_s27_student.py','export_t25_student_predictions.py']
    cmd=[PY,'-u']+([str(R/'code/cuda_runner.py')] if is_cuda else [])+[str(R/'code'/script),*map(str,args)]
    save(R/'stages'/f'{stage}.command.json',dict(argv=cmd,gpu=0,allocator_fraction=.25 if is_cuda else None))
    with (R/'logs'/f'{stage}.log').open('a') as log:
        proc=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        save(R/'active_process.json',dict(stage=stage,pid=proc.pid,gpu=0));rc=proc.wait()
    if rc:raise RuntimeError(f'{stage} exited {rc}; see logs/{stage}.log')
    marker.write_text(stamp());event(stage,'complete')

def setup():
    assert not R.exists(),R
    R.mkdir();(R/'logs').mkdir();(R/'stages').mkdir()
    shutil.copytree(OLD/'code',R/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__,R/'code/run_students.py');shutil.copytree(OLD/'protocol',R/'protocol');shutil.copy2(OLD/'router.json',R/'router.json')
    (R/'code/cuda_runner.py').write_text("import sys,runpy,torch\ntorch.set_num_threads(4)\ntorch.cuda.set_per_process_memory_fraction(.25)\nsys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n")
    for split in ['train','validation']:
        src=OLD/f'quality/anchor_conditioned_target_pooling/propagation_quality_{split}/propagation_quality.jsonl'
        dst=R/f'quality/anchor_conditioned_target_pooling/propagation_quality_{split}/propagation_quality.jsonl';dst.parent.mkdir(parents=True);shutil.copy2(src,dst)
    preview=read(DIAG/'train_preview_manifest.jsonl');original=read(OLD/'pseudo_manifest_original.jsonl')
    assert len(preview)==580 and len(original)==448
    by={r['target_id']:r for r in preview};assert len(by)==580
    for r in original:
        assert r['route_id']==by[r['target_id']]['route_id']
        assert r['pseudo_mask_path']==by[r['target_id']]['pseudo_mask_path']
        assert r['explicit_quality_weight']==by[r['target_id']]['explicit_quality_weight']
    router=module('filter_student_router',R/'code/router.py');model=router.Ridge(**json.loads((R/'router.json').read_text()))
    clean=lambda r:{k:v for k,v in r.items() if not k.startswith('gt_') and 'evaluation_only' not in k}
    rows=read(R/'quality/anchor_conditioned_target_pooling/propagation_quality_train/propagation_quality.jsonl')
    groups=router.grouped(rows);global_quality={r['target_id']:r['q_multi'] for r in read(OLD/'train_top1_and_quality.jsonl')}
    support={r['merged_id'] for r in read(R/'protocol/support_manifest.jsonl')}
    records=read(R/'protocol/merged_manifest.jsonl');unlabeled={r['merged_id'] for r in records if r['split']=='train'}-support
    assert len(support)==8 and len(unlabeled)==792 and set(groups)==unlabeled
    assert not set(by)&support
    expected=set()
    for target,rr in groups.items():
        eligible=[r for r in rr if r['q_cycle']>=.95]
        if global_quality[target]>=.90 and eligible:
            selected=max(eligible,key=lambda r:(model.score(clean(r)),-r['bridge_count'],r['route_id']))
            expected.add(target);assert selected['route_id']==by[target]['route_id']
            assert sha(selected['forward_mask_path'])==selected['forward_mask_sha256']
    assert expected==set(by)
    for r in preview:r.pop('preview_only',None)
    jsonl(R/'pseudo_manifest_original.jsonl',preview)
    cfg=json.loads((OLD/'config.json').read_text());cfg.update(created_at=stamp(),experiment='TP candidate return filtering before independent Router',
        reference_run=str(OLD),initial_pseudo_count=580,remaining_for_committee=212,
        changed_rule='Keep image-level q_multi>=.90; rank only candidates with q_return>=.95 using the unchanged frozen TP Router',
        unchanged='8/792 identities; SAM3 candidates; Router; thresholds; student initialization/seed/loss/batch/steps; uint16 fix; committee rules; X3 selection and B7 formula',
        test_policy='S2/S3 test deferred. Freeze X3 ordinary validation-best and B7 rule; then evaluate all 100 test targets once, irrespective of whether validation improved. No test tuning.',
        scope='New 580 pool -> regenerate S3 -> retrain S2/S3 -> rebuild committee on remaining 212 -> retrain X3 -> validation -> frozen test',
        original_448_masks_and_weights_unchanged=True,cuda_allocator_fraction=.25)
    cfg['code_sha256']={f.name:sha(f) for f in (R/'code').glob('*.py')}
    save(R/'config.json',cfg);save(R/'pool_audit.json',dict(original_count=448,new_count=580,added_count=132,remaining_count=212,all_original_routes_masks_weights_unchanged=True,
        original_protocol_sha256=sha(OLD/'protocol/merged_manifest.jsonl'),new_protocol_sha256=sha(R/'protocol/merged_manifest.jsonl'),all_selected_mask_hashes_verified=580,
        candidate_selection_recomputed=True,training_gt_not_used_for_selection=True))
    # Reuse the original preflight with only the isolated directory/count updated.
    pre=(OLD/'code/preflight.py').read_text().replace("work/kvasir_tp_student_mainline_20260907","work/kvasir_tp_filterfirst_students_20260909").replace('len(ds.rows)==456','len(ds.rows)==588')
    (R/'code/preflight.py').write_text(pre)
    command('s3_consensus','prepare_phase1_s3_consensus.py',['--original-manifest',R/'pseudo_manifest_original.jsonl','--quality-root',R/'quality','--output-root',R/'S3_consensus','--min-bridge',0,'--max-bridge',6])
    assert len(read(R/'S3_consensus/pseudo_consensus.jsonl'))==580
    command('preflight','preflight.py',[])
    (R/'PREPARED').write_text(stamp());event('prepare','complete',initial_pool=580,remaining=212)

def paired(rows,old,key='dice'):
    ref={r['target_id']:r[key] for r in old};d=np.array([r[key]-ref[r['target_id']] for r in rows]);rng=np.random.default_rng(2026)
    ci=np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975])
    return dict(mean_delta=float(d.mean()),ci95=ci.tolist(),wins=int((d>1e-12).sum()),losses=int((d< -1e-12).sum()),ties=int((abs(d)<=1e-12).sum()))

def run():
    assert (R/'PREPARED').exists()
    with (R/'STARTED').open('x') as f:f.write(stamp())
    m=module('filter_student_mainline',R/'code/mainline.py');m.R=R;m.command=command
    try:
        common=['--data-path',m.DATA,'--labeled-list',R/'protocol/frozen_labeled_images.txt','--seed',2026,'--max-iterations',40000,'--val-interval',200,'--num-workers',4]
        for name,manifest in [('S2',R/'pseudo_manifest_original.jsonl'),('S3',R/'S3_consensus/pseudo_consensus.jsonl')]:
            command('train_'+name,'run_t24_student.py',[*common,'--pseudo-manifest',manifest,'--output-dir',R/'students'/name,'--experiment',name,'--defer-test'])
            for suffix,ckpt in [('valbest','student_best.pth'),('final','student_final.pth')]:m.export(name+'_'+suffix,name,ckpt,['train'])
        args=['--train-metadata',m.DATA/'train/metadata.jsonl','--labeled-list',R/'protocol/frozen_labeled_images.txt','--quality-root',R/'quality','--original-manifest',R/'pseudo_manifest_original.jsonl','--output-dir',R/'audit','--min-bridge',0,'--max-bridge',6]
        for name in ['S2_valbest','S2_final','S3_valbest','S3_final']:args+=['--predictions',R/f'predictions/{name}/student_predictions_train.jsonl','--predictions-name',name]
        command('committee_audit','phase1_audit_tiers.py',args)
        command('build_x3_manifest','phase1_build_x3_manifest.py',['--original',R/'pseudo_manifest_original.jsonl','--tier-a',R/'audit/tier_A.jsonl','--tier-b',R/'audit/tier_B.jsonl','--output',R/'pseudo_manifest_x3.jsonl'])
        pp=read(R/'pseudo_manifest_x3.jsonl');counts=collections.Counter(r['sample_type'] for r in pp)
        assert counts['original']==580 and counts['tier_a']+counts['tier_b']>=6
        assert len(pp)==len({r['target_id'] for r in pp})
        assert json.loads((R/'audit/summary.json').read_text())['remaining_count']==212
        command('train_X3','run_s27_student.py',[*common,'--pseudo-manifest',R/'pseudo_manifest_x3.jsonl','--output-dir',R/'students/X3','--experiment','X3','--batch-size',12,'--gt-bs',3,'--original-bs',3,'--new-bs',6])
        ckpt=R/'students/X3/student_best.pth'
        save(R/'frozen_official_checkpoint.json',dict(checkpoint=str(ckpt),sha256=sha(ckpt),selection='ordinary validation Dice best',b7_rule='Original formula, original TP b0-b6 candidates',no_test_used_to_select=True))
        m.export('X3_best','X3','student_best.pth',['train','validation']);val=m.b7('validation')
        valcomp=paired(read(R/'b7_validation/per_target_metrics.jsonl'),read(OLD/'b7_validation/per_target_metrics.jsonl'))
        save(R/'validation_comparison.json',dict(new=val,vs_original_448_B7=valcomp))
        (R/'FROZEN_BEFORE_TEST').write_text(stamp())
        src=OLD/'quality/anchor_conditioned_target_pooling/propagation_quality_test/propagation_quality.jsonl'
        dst=R/'quality/anchor_conditioned_target_pooling/propagation_quality_test/propagation_quality.jsonl';dst.parent.mkdir(parents=True);shutil.copy2(src,dst)
        m.export('X3_best','X3','student_best.pth',['test']);test=m.b7('test')
        testcomp=paired(read(R/'b7_test/per_target_metrics.jsonl'),read(OLD/'b7_test/per_target_metrics.jsonl'))
        directcomp=paired(read(R/'b7_test/student_direct_metrics.jsonl'),read(OLD/'b7_test/student_direct_metrics.jsonl'))
        result=dict(experiment='filter candidates before Router',initial_pool=580,committee=json.loads((R/'audit/summary.json').read_text()),x3_pool=dict(counts),validation=val,test=test,
            validation_vs_448_B7=valcomp,test_vs_448_B7=testcomp,test_X3_vs_448_X3=directcomp,
            old_448_results=json.loads((OLD/'results.json').read_text())['test'],single_TP_router_test=.8854326463411542)
        save(R/'results.json',result)
        verified=0
        for r in read(R/'b7_test/per_target_metrics.jsonl'):
            assert sha(r['final_mask_path'])==r['mask_sha256'];verified+=1
        assert verified==100 and sha(ckpt)==json.loads((R/'frozen_official_checkpoint.json').read_text())['sha256']
        save(R/'completion_audit.json',dict(final_masks_verified=100,mask_dice_recomputed_by_b7=True,frozen_checkpoint_hash_matches=True,only_initial_filter_rule_changed=True))
        old=json.loads((OLD/'results.json').read_text())
        lines=['# Kvasir：先筛返回候选再 Router 的完整学生对照','','只改变首批伪标签筛选顺序；原 448 张的 mask 与图像权重逐项保持，新增 132 张形成 580 张种子池。原数据划分、单 TP 候选、冻结 Router、学生结构与训练参数、委员会规则及 B7 公式保持。S3 共识、S2/S3、委员会、X3 全部重新生成或训练。', '',f'新委员会分级：{result["committee"]["tier_counts"]}；X3 伪标签来源：{dict(counts)}。', '', '| 方法 | 原 448 池 | 新 580 池 |','|---|---:|---:|',f'| X3 validation 单图（导出入口） | {old["validation"]["student_direct_dice"]:.6f} | {val["student_direct_dice"]:.6f} |',f'| B7 validation | {old["validation"]["selected_dice"]:.6f} | {val["selected_dice"]:.6f} |',f'| X3 test 单图 | {old["test"]["student_direct_dice"]:.6f} | {test["student_direct_dice"]:.6f} |',f'| B7 test | {old["test"]["selected_dice"]:.6f} | {test["selected_dice"]:.6f} |', '',f'B7 test 配对差异：{testcomp}。X3 test 配对差异：{directcomp}。', '', 'X3 checkpoint 均按普通 validation Dice 选择，再冻结进行一次完整 test 评估。新增 132 张不等于最终 X3 额外增加 132 张：委员会对新的剩余 212 张重新审核，并对训练清单去重。', '', '单次固定 seed 对照用于观察该筛选改动的效果，不能将更高覆盖率直接当作性能提升；本实验不自动覆盖原单 TP 独立 Router 基线。']
        (R/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(R/'report.md',P/'reproduction_reports/Kvasir_TP_filterfirst_students_20260909.md')
        event('pipeline','complete',test_dice=test['selected_dice']);(R/'COMPLETE').write_text(stamp())
    except BaseException:
        err=traceback.format_exc();(R/'FAILED.txt').write_text(err);event('pipeline','failed',error=err);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['setup','run']);args=parser.parse_args()
    if args.action=='setup':setup()
    else:run()
