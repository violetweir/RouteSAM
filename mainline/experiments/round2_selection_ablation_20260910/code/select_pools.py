import collections,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
A=P/'new_project/experiments/round2_pool_a_20260910'
R=P/'new_project/experiments/round2_selection_ablation_20260910'
TP=P/'work/kvasir_tp_filterfirst_students_20260909'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def jsonl(p,x):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
def select(rows,name):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['target_id']].append(r)
 out=[]
 for target,g in sorted(groups.items()):
  if name!='A0':g=[r for r in g if r['nonempty'] and r['q_return']>=.95]
  if name=='A2':g=[r for r in g if r['q_multi']>=.95 and r['q_model']>=.95]
  if g:out.append(dict(max(g,key=lambda r:(r['b7_score'],r['q_multi'],r['q_return'],r['route_id']))))
 return out
def summarize(rows):
 if not rows:return dict(count=0)
 d=np.array([r['dice'] for r in rows]);j=np.array([r['iou'] for r in rows])
 return dict(count=len(rows),mean_dice=float(d.mean()),mean_iou=float(j.mean()),median_dice=float(np.median(d)),below08=int((d<.8).sum()),below09=int((d<.9).sum()))
def main():
 R.mkdir(exist_ok=False);(R/'code').mkdir()
 policy=dict(variants={'A0':'Original max B7 then nonempty, calibrated scalar threshold; preserve exact428 baseline',
                      'A1':'Before max B7 require nonempty and R>=.95',
                      'A2':'Before max B7 require nonempty and R>=.95,C>=.95,S>=.95'},
  threshold_grid=json.loads((A/'policy.json').read_text())['threshold_grid'],
  calibration='validation mean retained Dice>=.95 and n>=20; maximum count, lower threshold ties; if infeasible maximize mean among n>=20',
  training_arm_selection='Among distinct A1/A2 calibrated pools, prefer validation quality target met, then validation count, mean Dice, then simpler A1; no train GT used',
  matched_coverage='Common K=min(428, available nonempty eligible targets of A0/A1/A2); top K by B7, then target_id; diagnostic only',
  training=dict(epochs=10,batch_size=1,weights='all GT/pseudo samples1; once/image/epoch; no GT oversampling',
     labels='hard selected mask',prompt='colon polyp',lora='image-model full-module LoRA rank16 alpha32 dropout.1',
     optimizer='AdamW lr5e-5 weight_decay.01 constant',seed=2026,train_precision='FP32',validation_precision='BF16',
     source_canvas=256,model_input=1008,metric_canvas=256,
     selection='best mean direct validation Dice over epochs0..10; tie keeps earlier',
     test='after both runs finish and both val-best checkpoints frozen, evaluate both direct test100; fixed text, class>=.5, mask>=.5, union',
     comparison_limit='same epochs means different update counts if pool sizes differ; evaluate practical pool recipe; matched-K is offline diagnostic'),
  no_pool_b_or_new_student_training=True,no_tracker_pilot_weights=True,train_gt_audit='after all pools and training arm frozen only')
 save(R/'policy.json',policy)
 train=read(A/'b7/train/candidate_scores.jsonl');val=read(A/'b7/validation/candidate_scores.jsonl')
 vg={r['route_id']:r for r in read(TP/'quality/anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl')}
 frozen={};tables={};pools={};all_selected={}
 for name in ['A0','A1','A2']:
  vs=select(val,name);ts=select(train,name);all_selected[name]=ts;tab=[]
  for threshold in policy['threshold_grid']:
   kept=[r for r in vs if r['nonempty'] and r['b7_score']>=threshold]
   tab.append(dict(threshold=threshold,count=len(kept),mean_dice=float(np.mean([vg[r['route_id']]['gt_dice_evaluation_only'] for r in kept])) if kept else None))
  feasible=[r for r in tab if r['count']>=20 and r['mean_dice']>=.95]
  if feasible:best=max(feasible,key=lambda r:(r['count'],-r['threshold']))
  else:best=max([r for r in tab if r['count']>=20],key=lambda r:(r['mean_dice'],r['count'],-r['threshold']))
  frozen[name]={**best,'quality_target_met':bool(feasible)};tables[name]=tab
  pools[name]=[r for r in ts if r['nonempty'] and r['b7_score']>=best['threshold']]
 assert {(r['target_id'],r['route_id']) for r in pools['A0']}=={(r['target_id'],r['route_id']) for r in read(A/'pool_a/manifest.jsonl')}
 candidates=[n for n in ['A1','A2'] if {(r['target_id'],r['route_id']) for r in pools[n]}!={(r['target_id'],r['route_id']) for r in pools['A0']}]
 assert candidates,'Both alternatives exactly duplicate baseline; do not launch duplicate trainings'
 chosen=max(candidates,key=lambda n:(frozen[n]['quality_target_met'],frozen[n]['count'],frozen[n]['mean_dice'],n=='A1'))
 save(R/'validation_thresholds.json',tables);save(R/'FROZEN_CALIBRATION.json',dict(variants=frozen,chosen_training_alternative=chosen))
 K=min([428]+[sum(r['nonempty'] for r in all_selected[n]) for n in all_selected])
 matched={n:sorted([r for r in rs if r['nonempty']],key=lambda r:(-r['b7_score'],r['target_id']))[:K] for n,rs in all_selected.items()}
 manifests={}
 for name,rows in pools.items():
  dest=R/'pools'/name;(dest/'masks').mkdir(parents=True)
  manifest=[]
  for r in rows:
   f=dest/'masks'/(r['target_id'].replace('::','__')+'.png');shutil.copy2(r['source_mask_path'],f)
   assert sha(f)==r['source_mask_sha256']
   manifest.append({**r,'pseudo_mask_path':str(f),'pseudo_mask_sha256':sha(f),'sample_weight':1.})
  jsonl(dest/'manifest.jsonl',manifest);manifests[name]=manifest
  jsonl(dest/'matched_coverage.jsonl',matched[name])
 save(R/'POOLS_FROZEN.json',dict(training_arms=['A0',chosen],matched_K=K,
   manifest_hashes={n:sha(R/'pools'/n/'manifest.jsonl') for n in pools},policy_sha256=sha(R/'policy.json'),unlabeled_gt_used_for_selection=False))
 # Audit only after all selection rules, members and training arms are frozen.
 meta={r['merged_id']:r for r in read(P/'work/kvasir_1pct_anchors/baseline_data/train/metadata.jsonl')}
 cache={}
 def audit(rows):
  out=[]
  for r in rows:
   if r['target_id'] not in cache:
    cache[r['target_id']]=np.asarray(Image.open(meta[r['target_id']]['mask_file_name']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
   g=cache[r['target_id']];m=np.asarray(Image.open(r['source_mask_path']).convert('L'))>127
   i=int((g&m).sum());total=int(g.sum())+int(m.sum());union=total-i
   out.append(dict(target_id=r['target_id'],route_id=r['route_id'],dice=2*i/total if total else 1.,iou=i/union if union else 1.))
  return out
 results={}
 for name in pools:
  aa=audit(pools[name]);mm=audit(matched[name]);jsonl(R/'pools'/name/'audit.jsonl',aa);jsonl(R/'pools'/name/'matched_audit.jsonl',mm)
  results[name]=dict(validation=frozen[name],pool=summarize(aa),matched_coverage=summarize(mm),train_total=len(aa)+8)
 save(R/'screening_results.json',dict(variants=results,training_arms=['A0',chosen],matched_K=K))
 assert all(sha(R/'pools'/n/'manifest.jsonl')==h for n,h in json.loads((R/'POOLS_FROZEN.json').read_text())['manifest_hashes'].items())
 lines=['# Round2 池 A 筛选消融与 SAM3 单图微调对比','','## 目标','','比较候选前置约束能否改善B7训练池，以及是否改善SAM3的直接分割。本轮不使用传播微调权重，不训练池B或新学生。','',
 '| 方案 | 筛选方式 |','|---|---|','| A0 | 原B7最高候选，再检查非空和总分阈值 |','| A1 | 先要求非空、返回一致性R≥0.95，再选B7最高 |','| A2 | 先要求非空，R、候选一致性C、学生一致性S均≥0.95，再选B7最高 |','',
 '三种方案使用相同B7公式与validation总分阈值网格。质量目标为validation保留均值Dice≥0.95且至少20张，优先最大覆盖率。规则、训练组和全部池成员冻结后才读取训练GT进行审计。','',
 '## 筛选结果','','| 方案 | B7阈值 | Val保留数 | Val均值Dice | 训练伪标签数 | 池均值Dice | Dice<0.8 |','|---|---:|---:|---:|---:|---:|---:|']
 for n,v in results.items():
  x=v['validation'];y=v['pool'];lines.append(f"| {n} | {x['threshold']:.2f} | {x['count']} | {x['mean_dice']:.6f} | {y['count']} | {y['mean_dice']:.6f} | {y['below08']} |")
 lines+=['',f'按validation预设规则选中 **{chosen}** 与A0进行训练。此决定先于训练GT审计。', '',f'## 相同数量诊断（K={K}）','','| 方案 | 平均Dice | 平均IoU |','|---|---:|---:|']
 for n,v in results.items():lines.append(f"| {n} | {v['matched_coverage']['mean_dice']:.6f} | {v['matched_coverage']['mean_iou']:.6f} |")
 lines += ['', '## 双GPU训练方案','',f'- GPU0：A0，{len(pools["A0"])}张硬伪标签+8张GT。',f'- GPU1：{chosen}，{len(pools[chosen])}张硬伪标签+8张GT。',
 '- 两组从同一原始SAM3-base和同一LoRA初始化出发；使用同一训练程序与固定文本colon polyp。',
 '- 首轮10个epoch，batch1，每图每epoch一次，GT/伪标签等权，不重复采样GT。池大小不同时更新步数不同；这是训练池配方的实际比较，不是固定计算量的因果消融。',
 '- 采用现有SAM3图像模型全模块LoRA配置：rank16、alpha32、dropout0.1；AdamW lr5e-5、weight decay0.01、恒定学习率；seed2026。',
 '- 沿用已有训练器的分类、存在性、框和mask损失及其系数；等权指图像样本权重相同，不代表各损失项系数相同。',
 '- RGB先固定为256×256，与伪标签坐标一致，再输入模型的1008×1008尺寸。验证Dice统一在256×256计算，GT从原图最近邻直接缩放，避免重复缩放影响指标。',
 '- 每epoch完整validation100；按直接Dice在epoch0..10中选best，平分保留较早权重。两组权重选定后再统一直接test100。',
 '- 推理只有单图+固定文本，不提供GT点/框，不使用TP、Router、B7选择测试mask。类别分数≥0.5的查询mask取并集，mask概率阈值0.5。',
 '- 先验证单步训练、有限梯度、冻结参数、零更新输出一致性及checkpoint恢复，再启动双卡正式训练。','',
 '## 解释边界','','训练池GT均值只用于事后描述，已经看过的训练审计结果不作为新的阈值调参目标。validation也用于筛选校准和checkpoint选择，最终独立性能以冻结权重后的test报告为准。','',
 f'实验目录：`{R}`。完整策略、校准表、池清单、匹配数量诊断和逐图审计已归档。训练结果完成后追加。','']
 (R/'experiment.md').write_text('\n'.join(lines),encoding='utf-8');shutil.copy2(R/'experiment.md',P/'new_project/round2_selection_ablation.md')
 print(json.dumps(dict(variants=results,training_arms=['A0',chosen],matched_K=K),indent=2))
if __name__=='__main__':main()
