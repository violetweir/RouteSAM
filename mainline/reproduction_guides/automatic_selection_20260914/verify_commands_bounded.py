from pathlib import Path
import os,sys,subprocess,json,shutil,importlib.util
p=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');g=p/'new_project/reproduction_guides/automatic_selection_20260914';e=p/'new_project/experiments'
py='/home/violet/anaconda3/envs/sam3/bin/python';helper=g/'reproduce_automatic_selection.py';mode='sam3enc_anchor_conditioned_target_pooling'
os.environ.update(PYTHONPATH='/Data_8TB/lht/sam3:'+str(p/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
sys.path[:0]=['/Data_8TB/lht/sam3',str(p/'src')]
rd=lambda f:[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
checks=json.loads((g/'command_checks.json').read_text())
for name,base in [('isic2018','automatic_anchor_tp_test_20260913/isic2018'),('busi','busi_auto5_tp_1pct_20260913')]:
 out=g/('verify_tp_'+name)
 if not out.exists():subprocess.run([py,str(helper),'prepare','--dataset',name,'--selection',str(g/('verify_'+name)),'--out',str(out)],check=True)
 spec=importlib.util.spec_from_file_location('route_'+name,out/'code/stage1_feature_knn_routes.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 rr=rd(out/'protocol/merged_manifest.jsonl');support=rd(out/'protocol/support_manifest.jsonl')
 state=m.build_mode_state(mode,rr,support,out/'quality_root/features','sam3_base','patch_mean',256)
 target=sorted(r['merged_id'] for r in rr if r['split']=='test')[0]
 # Same cache row order and same train; limit test targets only in this integration check.
 rr=[dict(r,split='check_excluded') if r['split']=='test' and r['merged_id']!=target else r for r in rr]
 rows=m.freeze_routes(mode,out/'sample_route_check',rr,support,state,'test',0,6,32,True,False)
 orig=rd(e/base/f'quality_root/{mode}/test_pool0_stage1/routes.jsonl')
 assert {x['route_id'] for x in rows}=={x['route_id'] for x in orig if x['target_id']==target}
 rel=Path('quality_root')/mode/'propagation_quality_test/propagation_quality.jsonl';dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(e/base/rel,dest)
 subprocess.run([py,str(helper),'evaluate','--dataset',name,'--out',str(out),'--split','test'],check=True)
 results=json.loads((out/'test_results.json').read_text());assert results['max_abs_fixed_dice_difference']<1e-12
 checks[name]=dict(route_check_targets=1,routes=7,route_ids_exact_match=True,scoring_existing_masks=True,new_GPU_predictions=False,max_abs_fixed_dice_difference=results['max_abs_fixed_dice_difference'])
 (g/'command_checks.json').write_text(json.dumps(checks,indent=2))
print('BOUNDED_CHECKS_COMPLETE')
