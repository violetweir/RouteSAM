"""Round2 pool A: frozen Round1 B7, validation calibration, post-freeze GT audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,collections,hashlib,json,random,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
S=P/'new_project/experiments/tp_student_rescreen_20260910'
TP=P/'work/kvasir_tp_filterfirst_students_20260909'
DATA=P/'work/kvasir_1pct_anchors/baseline_data'
R=P/'new_project/experiments/round2_pool_a_20260910'
BASE=Path('/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt')
sys.path.insert(0,str(S/'code'))
from train_student import SamUnet

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def jsonl(p,rows):Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def binary(p):
 m=np.asarray(Image.open(p).convert('L'))>127
 assert m.shape==(256,256)
 return m
def dice(a,b):
 n=int(a.sum())+int(b.sum());return 2*int((a&b).sum())/n if n else 1.
def metrics(a,b):
 i=int((a&b).sum());u=int((a|b).sum())
 return dict(dice=dice(a,b),iou=i/u if u else 1.)
def resolve_image(row,split):
 p=Path(row['file_name']);return p if p.is_absolute() else DATA/split/p

@torch.inference_mode()
def predict(model,metadata,split):
 dest=R/'student_predictions'/split;(dest/'masks').mkdir(parents=True)
 results=[]
 for row in sorted(metadata,key=lambda x:x['merged_id']):
  image=resolve_image(row,split)
  a=np.asarray(Image.open(image).convert('RGB').resize((256,256),Image.Resampling.NEAREST),dtype=np.float32)/255
  a=(a-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
  x=torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).unsqueeze(0).cuda()
  _,prob=model(x);mask=(prob[0,1]>=.5).cpu().numpy()
  f=dest/'masks'/(row['merged_id'].replace('::','__')+'.png');Image.fromarray((mask*255).astype(np.uint8)).save(f)
  results.append(dict(target_id=row['merged_id'],image_path=str(image),mask_path=str(f),mask_sha256=sha(f)))
 jsonl(dest/'manifest.jsonl',results)
 print('Predicted',split,len(results),flush=True)
 return {r['target_id']:r for r in results}

def select_b7(split,students):
 q=TP/f'quality/anchor_conditioned_target_pooling/propagation_quality_{split}/propagation_quality.jsonl'
 keys=['target_id','route_id','bridge_count','forward_mask_path','forward_mask_sha256','q_cycle','target_image_path']
 groups=collections.defaultdict(list)
 for original in read(q):
  row={k:original[k] for k in keys};groups[row['target_id']].append(row)
 assert set(groups)==set(students)
 dest=R/'b7'/split;dest.mkdir(parents=True)
 selected=[];scores=[]
 for target,group in sorted(groups.items()):
  assert len(group)==7 and {r['bridge_count'] for r in group}==set(range(7))
  mm=[]
  for r in group:
   assert sha(r['forward_mask_path'])==r['forward_mask_sha256'];mm.append(binary(r['forward_mask_path']))
  student=binary(students[target]['mask_path']);local=[]
  for i,r in enumerate(group):
   a=float(r['q_cycle']);b=float(np.mean([dice(mm[i],m) for j,m in enumerate(mm) if i!=j]));c=dice(mm[i],student)
   score=(max(a,1e-6)*max(b,1e-6)**2*max(c,1e-6)**2)**.2
   local.append(dict(target_id=target,image_path=r['target_image_path'],route_id=r['route_id'],bridge_count=r['bridge_count'],
                     q_return=a,q_multi=b,q_model=c,b7_score=score,nonempty=bool(mm[i].any()),
                     source_mask_path=r['forward_mask_path'],source_mask_sha256=r['forward_mask_sha256']))
  scores.extend(local);selected.append(dict(max(local,key=lambda r:(r['b7_score'],r['q_multi'],r['q_return'],r['route_id']))))
 jsonl(dest/'candidate_scores.jsonl',scores);jsonl(dest/'selected_all.jsonl',selected)
 return selected

def main():
 assert R.exists() and not (R/'policy.json').exists()
 source=json.loads((S/'results.json').read_text())['best'];checkpoint=Path(source['checkpoint'])
 assert sha(checkpoint)==source['sha256']
 thresholds=[0.0,.5,.6,.7,.75]+[round(i/100,2) for i in range(80,100)]
 policy=dict(stage='Round2 pool A; Round1 ends at B7',student_checkpoint=str(checkpoint),student_sha256=source['sha256'],student_epoch=source['epoch'],
    sam3_checkpoint=str(BASE),sam3_sha256=sha(BASE),candidate_source='Existing SAM3-base TP b0-b6, seven per target; masks and return scores frozen',
    formula='(max(R,1e-6)*max(C,1e-6)^2*max(S,1e-6)^2)^0.2',tie_break=['b7_score','q_multi','q_return','route_id'],
    candidate_selection='Highest B7 among all seven; no historical membership, A/B tiers, Router re-ranking or GT in selection',
    pool_acceptance='Selected mask nonempty AND B7>=validation-frozen threshold',threshold_grid=thresholds,
    calibration='Maximum coverage with validation retained mean Dice>=0.95 and retained count>=20; ties choose lower threshold',
    fallback='If none meets 0.95, among thresholds keeping>=20 select highest validation retained mean Dice, then higher coverage, then lower threshold; explicitly report unmet quality target',
    validation_quality_target=.95,validation_min_count=20,
    audit='User explicitly requested pool A mean Dice; unlabeled train GT allowed only after threshold and membership frozen; audit never changes pool',
    test_used=False,training_started=False,seed=2026)
 save(R/'policy.json',policy)
 inputs=[TP/'protocol/support_manifest.jsonl',TP/'protocol/merged_manifest.jsonl',DATA/'train/metadata.jsonl',DATA/'validation/metadata.jsonl',
    S/'b7_new_student/validation_best/per_target_metrics.jsonl',S/'runs/soft/validation_best/per_target_metrics.jsonl']
 inputs += [TP/f'quality/anchor_conditioned_target_pooling/propagation_quality_{split}/propagation_quality.jsonl' for split in ['train','validation']]
 save(R/'input_hashes.json',{str(p):sha(p) for p in inputs})
 train_meta=read(DATA/'train/metadata.jsonl');val_meta=read(DATA/'validation/metadata.jsonl')
 support={r['merged_id'] for r in read(TP/'protocol/support_manifest.jsonl')}
 train_meta=[r for r in train_meta if r['merged_id'] not in support]
 assert len(train_meta)==792 and len(val_meta)==100 and len(support)==8
 forbidden={str(Path(r['mask_file_name']).resolve()) for r in train_meta}
 original_open=Image.open;audit_enabled=False;train_gt_reads=[]
 def guarded_open(fp,*a,**kw):
  if isinstance(fp,(str,Path)) and str(Path(fp).resolve()) in forbidden:
   assert audit_enabled,'Unlabeled train GT opened before pool freeze'
   train_gt_reads.append(str(Path(fp).resolve()))
  return original_open(fp,*a,**kw)
 Image.open=guarded_open
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.2)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
 model=SamUnet(argparse.Namespace(in_channels=3,num_classes=2));model.load_state_dict(torch.load(checkpoint,map_location='cpu'));model=model.cuda().eval()
 train_predictions=predict(model,train_meta,'train');val_predictions=predict(model,val_meta,'validation')
 old_predictions={r['target_id']:r for r in read(S/'runs/soft/validation_best/per_target_metrics.jsonl')}
 assert all(np.array_equal(binary(p['mask_path']),binary(old_predictions[t]['mask_path'])) for t,p in val_predictions.items())
 del model;torch.cuda.empty_cache()
 train_selected=select_b7('train',train_predictions);val_selected=select_b7('validation',val_predictions)
 old_selected={r['target_id']:r for r in read(S/'b7_new_student/validation_best/per_target_metrics.jsonl')}
 assert all(r['route_id']==old_selected[r['target_id']]['route_id'] and abs(r['b7_score']-old_selected[r['target_id']]['b7_score'])<1e-12 for r in val_selected)
 save(R/'ALL_SELECTIONS_FROZEN.json',dict(train_sha256=sha(R/'b7/train/selected_all.jsonl'),validation_sha256=sha(R/'b7/validation/selected_all.jsonl'),unlabeled_gt_reads=0))
 val_by={r['merged_id']:r for r in val_meta};val_metrics=[]
 for r in val_selected:
  gt=np.asarray(Image.open(val_by[r['target_id']]['mask_file_name']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
  m=metrics(binary(r['source_mask_path']),gt)
  assert abs(m['dice']-old_selected[r['target_id']]['dice'])<1e-12
  val_metrics.append({**r,**m})
 jsonl(R/'validation_per_target_metrics.jsonl',val_metrics)
 table=[]
 for threshold in thresholds:
  kept=[r for r in val_metrics if r['nonempty'] and r['b7_score']>=threshold]
  row=dict(threshold=threshold,count=len(kept),coverage=len(kept)/100,mean_dice=float(np.mean([r['dice'] for r in kept])) if kept else None,
           mean_iou=float(np.mean([r['iou'] for r in kept])) if kept else None)
  table.append(row)
 save(R/'validation_threshold_table.json',table)
 eligible=[r for r in table if r['count']>=20 and r['mean_dice']>=.95]
 target_met=bool(eligible)
 if eligible:choice=max(eligible,key=lambda r:(r['count'],-r['threshold']))
 else:choice=max([r for r in table if r['count']>=20],key=lambda r:(r['mean_dice'],r['count'],-r['threshold']))
 frozen_rule=dict(**choice,quality_target_met=target_met,policy_sha256=sha(R/'policy.json'),table_sha256=sha(R/'validation_threshold_table.json'))
 save(R/'THRESHOLD_FROZEN.json',frozen_rule)
 kept=[r for r in train_selected if r['nonempty'] and r['b7_score']>=choice['threshold']]
 pool=R/'pool_a';(pool/'masks').mkdir(parents=True)
 manifest=[]
 for r in kept:
  dest=pool/'masks'/(r['target_id'].replace('::','__')+'.png');shutil.copy2(r['source_mask_path'],dest)
  assert sha(dest)==r['source_mask_sha256']
  manifest.append({**r,'pseudo_mask_path':str(dest),'pseudo_mask_sha256':sha(dest),'sample_weight':1.,'sample_type':'round2_pool_a_b7'})
 jsonl(pool/'manifest.jsonl',manifest)
 save(R/'POOL_A_FROZEN.json',dict(count=len(manifest),manifest_sha256=sha(pool/'manifest.jsonl'),threshold_sha256=sha(R/'THRESHOLD_FROZEN.json'),unlabeled_gt_reads=len(train_gt_reads)))
 assert not train_gt_reads
 # Only now permit audit-only access to hidden train labels. Never re-select.
 audit_enabled=True
 train_by={r['merged_id']:r for r in train_meta};audit=[]
 for r in manifest:
  gtpath=train_by[r['target_id']]['mask_file_name']
  gt=np.asarray(Image.open(gtpath).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
  audit.append(dict(target_id=r['target_id'],route_id=r['route_id'],**metrics(binary(r['pseudo_mask_path']),gt)))
 jsonl(R/'pool_a_gt_audit.jsonl',audit)
 d=np.array([r['dice'] for r in audit]);j=np.array([r['iou'] for r in audit])
 previous={r['target_id'] for r in read(S/'data/train_manifest.jsonl') if not r['is_gt']};current={r['target_id'] for r in manifest}
 result=dict(pool_a_pseudo_count=len(manifest),unlabeled_train_count=792,coverage=len(manifest)/792,gt_anchors=8,training_total_if_gt_added=len(manifest)+8,
    b7_threshold=choice['threshold'],validation_retained_count=choice['count'],validation_retained_mean_dice=choice['mean_dice'],validation_quality_target_met=target_met,
    pool_a_mean_dice=float(d.mean()),pool_a_mean_iou=float(j.mean()),pool_a_median_dice=float(np.median(d)),pool_a_dice_p10=float(np.quantile(d,.1)),pool_a_min_dice=float(d.min()),
    pool_a_below_08=int((d<.8).sum()),pool_a_below_09=int((d<.9).sum()),pool_a_below_095=int((d<.95).sum()),
    previous620_retained=len(previous&current),previous620_removed=len(previous-current),newly_added=len(current-previous),
    metric='Macro mean per-image binary Dice at 256x256; pseudo labels only, excluding 8 GT anchors',
    audit_only=True,pool_changed_after_gt_audit=False,sam3_training_started=False,test_used=False)
 save(R/'results.json',result)
 assert sha(pool/'manifest.jsonl')==json.loads((R/'POOL_A_FROZEN.json').read_text())['manifest_sha256']
 assert sha(checkpoint)==source['sha256']
 assert all(sha(p)==h for p,h in json.loads((R/'input_hashes.json').read_text()).items())
 save(R/'completion_audit.json',dict(train_targets=792,train_candidates=5544,validation_targets=100,validation_candidates=700,
   validation_student_predictions_reproduced=True,validation_b7_selection_reproduced=True,threshold_chosen_on_validation_only=True,
   train_gt_reads_before_freeze=0,train_gt_reads_after_freeze=len(train_gt_reads),unique_audited_gt=len(set(train_gt_reads)),
   manifest_unchanged_after_gt_audit=True,inputs_unchanged=True))
 lines=['# Round2：B7 重新筛选池 A', '', 'Round1 截止到新 B7。本次仅完成冻结 Round1、全量训练图 B7 选择和 validation 阈值校准，尚未微调 SAM3。', '',
 '## 冻结来源', '',f"- SAM3：原始 base，SHA256 `{policy['sam3_sha256']}`。",f"- 学生：Round1 620 张伪标签训练的 soft validation-best，epoch576，SHA256 `{source['sha256']}`。",
 '- 每图 TP b0–b6 共 7 张候选。B7 使用原公式和 tie-break，取最高分候选；接纳时要求该 mask 非空。',
 '- 全部 792 张无标注训练图重新处理；不默认沿用原 620 张，不引入 tracker 微调试验。', '',
 '## Validation 校准', '', '在读取本轮阈值表前固定：平均 Dice≥0.95、至少保留20张时取覆盖率最高的阈值；若无阈值满足，使用至少20张中平均 Dice 最高者并明确标注未达标。',
 '这是经验质量与覆盖率选择规则；validation 子集均值不代表独立泛化结果或质量保证。', '',
 '| B7 最低分 | 保留数 /100 | 平均 Dice | 平均 IoU |','|---|---:|---:|---:|']
 for t in table:
  ds=f"{t['mean_dice']:.6f}" if t['mean_dice'] is not None else '—';js=f"{t['mean_iou']:.6f}" if t['mean_iou'] is not None else '—'
  lines.append(f"| {t['threshold']:.2f} | {t['count']} | {ds} | {js} |")
 lines += ['',f"选定阈值：**{choice['threshold']:.2f}**；validation 保留 {choice['count']}/100，平均 Dice **{choice['mean_dice']:.6f}**；是否达到预设0.95目标：{'是' if target_met else '否'}。",'',
 '## 池 A 结果', '', '| 项目 | 数值 |','|---|---:|',f"| 池 A 伪标签数 | {len(manifest)} /792 |",f"| 覆盖率 | {len(manifest)/792:.2%} |",
 f"| **池 A 平均 Dice** | **{d.mean():.6f}** |",f"| 平均 IoU | {j.mean():.6f} |",f"| Dice 中位数 | {np.median(d):.6f} |",f"| Dice 第10百分位 | {np.quantile(d,.1):.6f} |",f"| Dice 最低值 | {d.min():.6f} |",
 f"| Dice<0.8 / <0.9 的图数 | {int((d<.8).sum())} / {int((d<.9).sum())} |",f"| 加入8张GT后的训练总数 | {len(manifest)+8} |",'',
 '**平均 Dice 是池 A 中伪标签对训练集真实 mask 的逐图宏平均，排除8张GT；属于冻结池后的事后审计，不是 SAM3 或学生的 test Dice。**',
 '未用训练图 GT 选候选、选阈值或删掉低 Dice 样本。此次用户明确要求池 A 平均 Dice，因此只在 THRESHOLD_FROZEN 与 POOL_A_FROZEN 写入后开放训练 GT 审计。', '',
 '## 与上一轮620张训练池的关系', '', f"- 保留 {len(previous&current)} 张，移除 {len(previous-current)} 张，新增 {len(current-previous)} 张。", '',
 '## 归档', '', f'实验目录：`{R}`。', '池 A：`pool_a/manifest.jsonl` 和 `pool_a/masks/`。',
 '逐候选分数、792张全量选择、validation阈值表、冻结时间点文件、逐图GT审计和输入哈希均已保存。', '']
 (R/'report.md').write_text('\n'.join(lines),encoding='utf-8')
 shutil.copy2(R/'report.md',P/'new_project/round2_pool_a_report.md')
 (R/'COMPLETE').write_text('complete\n')
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  R.mkdir(parents=True,exist_ok=True);(R/'FAILED.txt').write_text(traceback.format_exc());raise
