"""Isolated TP-only continuation: pseudo pool -> S2/S3 -> committee -> X3 -> B7."""
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
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'work/kvasir_tp_student_mainline_20260907'
B=P/'work/kvasir_tp_b0_b6_router_baseline_20260907'
OLD=P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6/sam3enc_anchor_conditioned_target_pooling'
TEST=P/'work/rerun_kvasir_sam3base_test_20260906/quality_root/sam3enc_anchor_conditioned_target_pooling'
DATA=P/'work/kvasir_1pct_anchors/baseline_data'
MODE='anchor_conditioned_target_pooling'
PY='/home/violet/anaconda3/envs/mkunet_mamba/bin/python'
FILES=['mainline.py','prepare_phase1_s3_consensus.py','phase1_audit_tiers.py','phase1_build_x3_manifest.py','run_t24_student.py','run_s27_student.py','export_t25_student_predictions.py']

def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def jsonl(p,rr):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rr))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
    return h.hexdigest()
def mask(p):return np.asarray(Image.open(p).convert('L'))>127
def dice(a,b):
    n=int(a.sum())+int(b.sum());return 2*int(np.logical_and(a,b).sum())/n if n else 1.
def clean(row):return {k:v for k,v in row.items() if not k.startswith('gt_') and 'evaluation_only' not in k}
def quality(split):return read(R/'quality'/MODE/f'propagation_quality_{split}/propagation_quality.jsonl')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def event(stage,status,extra=None):
    row={'time':datetime.datetime.now().astimezone().isoformat(),'stage':stage,'status':status,**(extra or {})}
    save(R/'status.json',row)
    with (R/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def setup():
    assert (B/'FROZEN').exists() and not R.exists()
    R.mkdir();(R/'code').mkdir();(R/'logs').mkdir();(R/'stages').mkdir()
    for name in FILES:shutil.copy2(Path(__file__).parent/name,R/'code'/name)
    shutil.copytree(B/'protocol',R/'protocol');shutil.copy2(B/'router.json',R/'router.json');shutil.copy2(B/'router_implementation.py',R/'code/router.py')
    cfg={'created_at':datetime.datetime.now().astimezone().isoformat(),'baseline':str(B),'baseline_test_dice':.8854326463411542,'user_authorized_students':True,'sam3_frozen':True,'student_model':'historical SC-SAM SamUnet','route_mode':MODE,'min_bridge':0,'max_bridge':6,'candidate_count':7,'canvas':256,'split':{'train':800,'validation':100,'test':100,'labeled_train':8,'unlabeled_train':792},'pseudo_thresholds':{'q_multi':.90,'q_return':.95},'threshold_policy':'retain documented thresholds as fixed controls; report TP-only coverage and validation diagnostics; no test tuning','s3_consensus':{'beta':4.,'pixel_min_weight':.1},'training':{'seed':2026,'iterations_each':40000,'validation_interval':200,'s2_s3_batch':12,'s2_s3_labeled_batch':6,'x3_gt_original_new_batch':[3,3,6],'learning_rate':.01,'gpu':0,'schedule':'sequential; GPU1 already occupied'},'committee':'S2 val-best + S2 final + S3 val-best + S3 final, original C0 document thresholds; TP-only candidates','x3_checkpoint':'ordinary validation Dice best, final retained as validation diagnostic only','b7':'(max(q_return,1e-6)*max(q_multi,1e-6)^2*max(q_model,1e-6)^2)^.2; full 100-target coverage, no confidence filtering','test_policy':'S2/S3 tests deferred; X3 val-best frozen before official X3 direct and TP-only B7 test; final checkpoint not selected using test','scope':'C0 Step3 through Step10; later SAM3 LoRA is a separate continuation','code_sha256':{p.name:sha(p) for p in (R/'code').glob('*.py')}}
    save(R/'config.json',cfg);inputs={}
    for split in ['train','validation','test']:
        source=(TEST if split=='test' else OLD)/f'propagation_quality_{split}/propagation_quality.jsonl'
        rr=read(source);inputs[str(source)]=sha(source)
        assert len(rr)==(5544 if split=='train' else 700) and all(x['status']=='success' for x in rr)
        for row in rr:
            path=Path(row['forward_mask_path']);row['forward_mask_path']=str(path if path.is_absolute() else P/path)
            row['feature_mode']=MODE
        if split=='train':rr=[clean(row) for row in rr]
        jsonl(R/'quality'/MODE/f'propagation_quality_{split}/propagation_quality.jsonl',rr)
    save(R/'input_sha256.json',inputs)
    save(B/'student_continuation.json',{'authorized':True,'new_run':str(R),'baseline_metrics_and_masks_unchanged':True,'time':cfg['created_at']})
    event('setup','complete')

def pseudo_prepare():
    router=module('router',R/'code/router.py');scorer=router.Ridge(**json.loads((R/'router.json').read_text()))
    records=read(R/'protocol/merged_manifest.jsonl');labels={r['merged_id'] for r in read(R/'protocol/support_manifest.jsonl')};unlabeled={r['merged_id'] for r in records if r['split']=='train'}-labels
    summaries={}
    for split in ['validation','train']:
        groups=router.grouped(quality(split));selected=[]
        if split=='train':assert set(groups)==unlabeled and len(groups)==792
        for target,rr in sorted(groups.items()):
            assert len(rr)==7 and sorted(r['bridge_count'] for r in rr)==list(range(7))
            chosen=max([clean(r) for r in rr],key=lambda r:(scorer.score(r),-r['bridge_count'],r['route_id']))
            masks=[mask(r['forward_mask_path']) for r in rr];assert all(m.shape==(256,256) for m in masks)
            qmulti=float(np.mean([dice(masks[i],masks[j]) for i in range(7) for j in range(i+1,7)]))
            row={'target_id':target,'pseudo_mask_path':chosen['forward_mask_path'],'q_multi':qmulti,'q_return':float(chosen['q_cycle']),'route_id':chosen['route_id'],'feature_mode':MODE,'bridge_count':chosen['bridge_count'],'anchor_id':chosen['anchor_id'],'routeco_selected_score':scorer.score(chosen),'n_candidates':7}
            selected.append(row)
        jsonl(R/f'{split}_top1_and_quality.jsonl',selected)
        accepted=[r for r in selected if r['q_multi']>=.90 and r['q_return']>=.95]
        assert accepted,'Empty fixed-threshold pseudo pool'
        info={'total':len(selected),'accepted':len(accepted),'coverage':len(accepted)/len(selected),'q_multi_mean':float(np.mean([r['q_multi'] for r in accepted])),'q_return_mean':float(np.mean([r['q_return'] for r in accepted]))}
        if split=='validation':
            index={r['route_id']:r for r in quality(split)}
            info['selected_dice_descriptive_in_sample']=float(np.mean([index[r['route_id']]['gt_dice_evaluation_only'] for r in accepted]))
            info['note']='Frozen router fitted on full validation; coverage/retained quality is descriptive, not OOF evidence.'
        else:
            q=[r['q_multi'] for r in accepted];lo,hi=min(q),max(q)
            for r in accepted:r.update(sample_type='original',explicit_quality_weight=float(np.clip((r['q_multi']-lo)/max(hi-lo,1e-12),.2,1.)))
            assert len(accepted)>=6 and {r['target_id'] for r in accepted}<=unlabeled
            assert all(not any(k.startswith('gt_') or 'evaluation_only' in k for k in r) for r in accepted)
            jsonl(R/'pseudo_manifest_original.jsonl',accepted)
        summaries[split]=info
    save(R/'pseudo_pool_summary.json',summaries);event('pseudo_pool','complete',summaries)

def command(stage,script,args):
    marker=R/'stages'/f'{stage}.complete'
    if marker.exists():return
    event(stage,'running')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',SC_SAM_ROOT=str(P/'third_party/SC-SAM'))
    for key in ['ISIC_LIGHT_STRONG_AUG','ISIC_RESIZED_CACHE_ROOT']:env.pop(key,None)
    cmd=[PY,'-u',str(R/'code'/script),*map(str,args)]
    save(R/'stages'/f'{stage}.command.json',{'argv':cmd,'gpu':0})
    with (R/'logs'/f'{stage}.log').open('a') as log:
        proc=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        save(R/'active_process.json',{'stage':stage,'pid':proc.pid,'gpu':0})
        rc=proc.wait()
    if rc:raise RuntimeError(f'{stage} exited {rc}; see logs/{stage}.log')
    marker.write_text(datetime.datetime.now().astimezone().isoformat());event(stage,'complete')

def prepare():
    if not (R/'pseudo_manifest_original.jsonl').exists():pseudo_prepare()
    command('s3_consensus','prepare_phase1_s3_consensus.py',['--original-manifest',R/'pseudo_manifest_original.jsonl','--quality-root',R/'quality','--output-root',R/'S3_consensus','--min-bridge',0,'--max-bridge',6])
    original=read(R/'pseudo_manifest_original.jsonl');consensus=read(R/'S3_consensus/pseudo_consensus.jsonl')
    assert {r['target_id'] for r in original}=={r['target_id'] for r in consensus}
    assert all(r['n_candidates']==7 and Path(r['pseudo_consensus_path']).exists() and Path(r['pixel_weight_path']).exists() for r in consensus)
    (R/'PREPARED').write_text(datetime.datetime.now().astimezone().isoformat())

def export(name,run,checkpoint,splits):
    command('export_'+name+'_'+'_'.join(splits),'export_t25_student_predictions.py',['--run-dir',R/'students'/run,'--checkpoint',R/'students'/run/checkpoint,'--output-root',R/'predictions'/name,'--splits',*splits])

def b7(split):
    rr=quality(split);groups=collections.defaultdict(list)
    for row in rr:groups[row['target_id']].append(row)
    students={r['merged_id']:r for r in read(R/f'predictions/X3_best/student_predictions_{split}.jsonl')}
    chosen_rows=[];dest=R/f'b7_{split}'/'masks';dest.mkdir(parents=True,exist_ok=True)
    # Select using only masks, q_cycle and the frozen student; attach GT later.
    for target,group in sorted(groups.items()):
        assert len(group)==7
        mm=[mask(r['forward_mask_path']) for r in group];student=mask(students[target]['student_binary_mask'])
        candidates=[]
        for i,row in enumerate(group):
            a=float(row['q_cycle']);b=float(np.mean([dice(mm[i],m) for j,m in enumerate(mm) if i!=j]));c=dice(mm[i],student)
            score=(max(a,1e-6)*max(b,1e-6)**2*max(c,1e-6)**2)**.2
            candidates.append({'target_id':target,'route_id':row['route_id'],'bridge_count':row['bridge_count'],'q_return':a,'q_multi':b,'q_model':c,'b7_score':score,'source_mask_path':row['forward_mask_path']})
        pick=max(candidates,key=lambda r:(r['b7_score'],r['q_multi'],r['q_return'],r['route_id']))
        out=dest/(target.replace('::','__')+'.png');shutil.copy2(pick['source_mask_path'],out);pick.update(final_mask_path=str(out),mask_sha256=sha(out));chosen_rows.append(pick)
    assert len(chosen_rows)==100
    jsonl(R/f'b7_{split}/selected_masks.jsonl',chosen_rows)
    index={r['route_id']:r for r in rr};student_metrics=[]
    for row in chosen_rows:
        original=index[row['route_id']];gt=np.asarray(Image.open(original['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
        pred=mask(row['final_mask_path']);d=dice(pred,gt);assert abs(d-original['gt_dice_evaluation_only'])<1e-12
        inter=int(np.logical_and(pred,gt).sum());row.update(dice=d,iou=inter/max(int(pred.sum())+int(gt.sum())-inter,1),oracle=max(x['gt_dice_evaluation_only'] for x in groups[row['target_id']]))
        sm=mask(students[row['target_id']]['student_binary_mask']);si=int(np.logical_and(sm,gt).sum());student_metrics.append({'target_id':row['target_id'],'dice':dice(sm,gt),'iou':si/max(int(sm.sum())+int(gt.sum())-si,1)})
    result={'count':100,'selected_dice':float(np.mean([r['dice'] for r in chosen_rows])),'selected_iou':float(np.mean([r['iou'] for r in chosen_rows])),'oracle':float(np.mean([r['oracle'] for r in chosen_rows])),'student_direct_dice':float(np.mean([r['dice'] for r in student_metrics])),'student_direct_iou':float(np.mean([r['iou'] for r in student_metrics])),'bridge_counts':dict(collections.Counter(r['bridge_count'] for r in chosen_rows))}
    jsonl(R/f'b7_{split}/per_target_metrics.jsonl',chosen_rows);jsonl(R/f'b7_{split}/student_direct_metrics.jsonl',student_metrics);save(R/f'b7_{split}/summary.json',result)
    return result

def run():
    assert (R/'PREPARED').exists()
    with (R/'STARTED').open('x') as f:f.write(datetime.datetime.now().astimezone().isoformat())
    try:
        common=['--data-path',DATA,'--labeled-list',R/'protocol/frozen_labeled_images.txt','--seed',2026,'--max-iterations',40000,'--val-interval',200,'--num-workers',4]
        for name,pseudo in [('S2',R/'pseudo_manifest_original.jsonl'),('S3',R/'S3_consensus/pseudo_consensus.jsonl')]:
            command('train_'+name,'run_t24_student.py',[*common,'--pseudo-manifest',pseudo,'--output-dir',R/'students'/name,'--experiment',name,'--defer-test'])
            assert (R/'students'/name/'TRAINING_COMPLETE').exists()
            for suffix,checkpoint in [('valbest','student_best.pth'),('final','student_final.pth')]:export(name+'_'+suffix,name,checkpoint,['train'])
        args=['--train-metadata',DATA/'train/metadata.jsonl','--labeled-list',R/'protocol/frozen_labeled_images.txt','--quality-root',R/'quality','--original-manifest',R/'pseudo_manifest_original.jsonl','--output-dir',R/'audit','--min-bridge',0,'--max-bridge',6]
        for name in ['S2_valbest','S2_final','S3_valbest','S3_final']:args+=['--predictions',R/f'predictions/{name}/student_predictions_train.jsonl','--predictions-name',name]
        command('committee_audit','phase1_audit_tiers.py',args)
        command('build_x3_manifest','phase1_build_x3_manifest.py',['--original',R/'pseudo_manifest_original.jsonl','--tier-a',R/'audit/tier_A.jsonl','--tier-b',R/'audit/tier_B.jsonl','--output',R/'pseudo_manifest_x3.jsonl'])
        pseudo=read(R/'pseudo_manifest_x3.jsonl');counts=collections.Counter(r['sample_type'] for r in pseudo)
        assert counts['tier_a']+counts['tier_b']>=6,'Insufficient accepted expansion targets for documented 3/3/6 X3 batch'
        allowed={r['target_id'] for r in read(R/'train_top1_and_quality.jsonl')};assert {r['target_id'] for r in pseudo}<=allowed and len({r['target_id'] for r in pseudo})==len(pseudo)
        command('train_X3','run_s27_student.py',[*common,'--pseudo-manifest',R/'pseudo_manifest_x3.jsonl','--output-dir',R/'students/X3','--experiment','X3','--batch-size',12,'--gt-bs',3,'--original-bs',3,'--new-bs',6])
        assert (R/'students/X3/TRAINING_COMPLETE').exists()
        ckpt=R/'students/X3/student_best.pth';save(R/'frozen_official_checkpoint.json',{'checkpoint':str(ckpt),'sha256':sha(ckpt),'selection':'ordinary validation Dice best','b7_rule':'fixed original geometric formula, TP-only b0-b6','no_test_used_to_select':True})
        export('X3_best','X3','student_best.pth',['train','validation']);val=b7('validation')
        (R/'FROZEN_BEFORE_TEST').write_text(datetime.datetime.now().astimezone().isoformat())
        export('X3_best','X3','student_best.pth',['test']);test=b7('test')
        result={'baseline':json.loads((B/'baseline_card.json').read_text()),'pseudo_pool':json.loads((R/'pseudo_pool_summary.json').read_text()),'committee':json.loads((R/'audit/summary.json').read_text()),'x3_manifest':dict(counts),'validation':val,'test':test,'test_delta_vs_router':test['selected_dice']-.8854326463411542}
        save(R/'results.json',result)
        report='\n'.join(['# Kvasir TP-only 学生主线完成报告','','新基线：SAM3-base TP b0-b6独立Router，test Dice0.885433。原划分与8标注/792无标注身份保持；候选为同一TP-only七路线。','','按C0文档完成伪标签、S2/S3、四成员委员会、X3和B7。X3只按普通validation Dice选择best checkpoint；B7公式固定，test全100张，不作置信度过滤。','','| 方法 | Test Dice | Test IoU |','|---|---:|---:|','| TP Router基线 | 0.885433 | 0.828274 |',f"| X3学生单图 | {test['student_direct_dice']:.6f} | {test['student_direct_iou']:.6f} |",f"| X3-best + TP-only B7 | {test['selected_dice']:.6f} | {test['selected_iou']:.6f} |",'',f"B7候选Oracle（仅分析）：{test['oracle']:.6f}。",'',f'完整产物：`{R}`。训练、验证、模型、逐图预测、最终mask、脚本哈希和事件日志均保留。'])+'\n'
        (R/'report.md').write_text(report)
        with (P/'reproduction_reports/Kvasir_TP_student_mainline_20260907.md').open('x') as f:f.write(report)
        event('pipeline','complete');(R/'COMPLETE').write_text(datetime.datetime.now().astimezone().isoformat())
    except BaseException:
        (R/'FAILED').write_text(traceback.format_exc());event('pipeline','failed',{'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    if sys.argv[1]=='setup':setup()
    elif sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    else:raise ValueError(sys.argv[1])
