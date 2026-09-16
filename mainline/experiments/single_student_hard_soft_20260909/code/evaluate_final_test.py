"""User-requested final-epoch test audit, separate from validation-best results."""
import argparse
import json
from pathlib import Path
import torch
from train_student import read,save,sha,StudentDataset,loader,SamUnet,evaluate

R=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/new_project/experiments/single_student_hard_soft_20260909')
assert (R/'COMPLETE').exists()
OUT=R/'final_checkpoint_test';OUT.mkdir(exist_ok=False)
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
best=json.loads((R/'results.json').read_text())
frozen={v:dict(checkpoint=str(R/f'runs/{v}/student_final.pth'),sha256=sha(R/f'runs/{v}/student_final.pth'),epoch=816,iteration=39984) for v in ['hard','soft']}
save(OUT/'frozen_checkpoints.json',frozen)
results={}
for variant in ['hard','soft']:
 print('Evaluating',variant,'final',flush=True)
 model=SamUnet(argparse.Namespace(in_channels=3,num_classes=2))
 model.load_state_dict(torch.load(frozen[variant]['checkpoint'],map_location='cpu'));model=model.cuda().eval()
 val=evaluate(model,loader(StudentDataset(read(R/'data/validation_manifest.jsonl'),'hard',False),1,workers=1))
 expected=json.loads((R/f'runs/{variant}/summary.json').read_text())['final_validation']
 assert abs(val['dice']-expected['dice'])<1e-10
 test=evaluate(model,loader(StudentDataset(read(R/'data/test_manifest.jsonl'),'hard',False),1,workers=1),R/f'runs/{variant}/test_final')
 assert sha(frozen[variant]['checkpoint'])==frozen[variant]['sha256']
 metrics=read(R/f'runs/{variant}/test_final/per_target_metrics.jsonl')
 assert len(metrics)==len({x['target_id'] for x in metrics})==100
 for row in metrics:assert sha(row['mask_path'])==row['mask_sha256']
 results[variant]=dict(**frozen[variant],validation=val,test=test,best_checkpoint_test=best[variant]['test'],final_minus_best_test_dice=test['dice']-best[variant]['test']['dice'],final_validation_reproduced=True,all_100_saved_masks_hash_verified=True)
 save(OUT/'results.json',results);print(json.dumps(results[variant]),flush=True)
 del model;torch.cuda.empty_cache()
lines=['# 等权单学生：final checkpoint 的 test 补充评估','','按用户要求评估既有第 816 epoch / 39,984 iter 的 final checkpoint；未重新训练、未按 test 重新选择模型。输入预处理、阈值和逐图指标与此前 best 评估相同。','','| 版本 | Best test Dice | Final test Dice | Final test IoU | Final validation Dice |','|---|---:|---:|---:|---:|']
for v,x in results.items():lines.append(f"| {v} | {x['best_checkpoint_test']['dice']:.6f} | {x['test']['dice']:.6f} | {x['test']['iou']:.6f} | {x['validation']['dice']:.6f} |")
lines+=['','两组均覆盖完整 100 张 test。两个 final checkpoint 均复现训练末期的 validation Dice；checkpoint 哈希和全部 200 张输出 mask 哈希核查通过。','',f'输出目录：`{OUT}`；逐图结果及 mask 位于 `runs/hard/test_final` 和 `runs/soft/test_final`。']
report='\n'.join(lines)+'\n';(OUT/'report.md').write_text(report);(R.parents[1]/'single_student_final_test_results.md').write_text(report)
(OUT/'COMPLETE').write_text('complete\n')
