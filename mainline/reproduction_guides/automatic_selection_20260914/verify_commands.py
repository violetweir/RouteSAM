from pathlib import Path
import os,subprocess,json,shutil
p=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');g=p/'new_project/reproduction_guides/automatic_selection_20260914';e=p/'new_project/experiments'
py='/home/violet/anaconda3/envs/sam3/bin/python';helper=g/'reproduce_automatic_selection.py';mode='sam3enc_anchor_conditioned_target_pooling'
os.environ.update(PYTHONPATH='/Data_8TB/lht/sam3:'+str(p/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
checks={}
for name,base in [('kvasir','automatic_anchor_tp_test_20260913/kvasir'),('isic2018','automatic_anchor_tp_test_20260913/isic2018'),('busi','busi_auto5_tp_1pct_20260913')]:
 out=g/('verify_tp_'+name)
 subprocess.run([py,str(helper),'prepare','--dataset',name,'--selection',str(g/('verify_'+name)),'--out',str(out)],check=True)
 subprocess.run([py,str(out/'code/stage1_feature_knn_routes.py'),'--mode',mode,'--feature-source','sam3_base','--feature-size','256','--knn-feature','patch_mean','--beam-width','32','--min-bridge','0','--max-bridge','6','--split','test','--protocol-root',str(out/'protocol'),'--output-root',str(out/'quality_root')],check=True)
 rel=Path('quality_root')/mode/'test_pool0_stage1/routes.jsonl'
 rd=lambda f:[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
 orig=rd(e/base/rel);new=rd(out/rel);assert {r['route_id'] for r in orig}=={r['route_id'] for r in new},name
 # Scoring validation deliberately reuses existing frozen predictions, not a new GPU run.
 rel=Path('quality_root')/mode/'propagation_quality_test/propagation_quality.jsonl';dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(e/base/rel,dest)
 subprocess.run([py,str(helper),'evaluate','--dataset',name,'--out',str(out),'--split','test'],check=True)
 m=json.loads((out/'test_results.json').read_text());assert m['max_abs_fixed_dice_difference']<1e-12
 checks[name]=dict(routes=len(new),route_ids_exact_match=True,scoring_existing_masks=True,new_GPU_predictions=False,max_abs_fixed_dice_difference=m['max_abs_fixed_dice_difference'])
 (g/'command_checks.json').write_text(json.dumps(checks,indent=2))
print('ALL_CHECKS_COMPLETE')
