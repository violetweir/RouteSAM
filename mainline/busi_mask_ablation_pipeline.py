from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,time,traceback,importlib.util,collections
import numpy as np
from PIL import Image
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');E=P/'new_project/experiments';OLD=E/'busi_auto5_tp_1pct_20260913';R=E/'busi_mask_prompt_ablation_20260913';PY='/home/violet/anaconda3/envs/sam3/bin/python';MODE='sam3enc_anchor_conditioned_target_pooling'
VARIANTS={'mask_no_text':None,'mask_text':'breast lesion'}
def save(p,a):Path(p).write_text(json.dumps(a,indent=2,ensure_ascii=False)+'\n')
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmodule(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m

def prepare():
 assert (R/'SMOKE_COMPLETE').exists() and not (R/'PREPARED').exists()
 shutil.copy2(__file__,R/'pipeline.py');shutil.copy2(P/'new_project/busi_mask_prompt.py',R/'mask_prompt.py')
 hashes={}
 for v,text in VARIANTS.items():
  d=R/v;d.mkdir();(d/'logs').mkdir();(d/'code').mkdir();(d/'automatic').mkdir()
  for fn in ['router.py']:shutil.copy2(OLD/'code'/fn,d/'code'/fn)
  for split in ['validation','test']:
   f=OLD/f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl';dest=d/f'quality_root/{MODE}/{split}_pool0_stage1';dest.mkdir(parents=True);shutil.copy2(f,dest/'routes.jsonl');hashes[str(f)]=sha(f)
  rs=(OLD/'code/run_router.py').read_text().replace("R=E/'busi_auto5_tp_router_20260913'",f"R=E/'busi_mask_prompt_ablation_20260913/{v}_router'").replace("SOURCES={'busi':'busi_auto5_tp_1pct_20260913'}",f"SOURCES={{'busi':'busi_mask_prompt_ablation_20260913/{v}'}}")
  (d/'code/run_router.py').write_text(rs)
  summary=(OLD/'pipeline.py').read_text().replace("R=E/'busi_auto5_tp_1pct_20260913'",f"R=E/'busi_mask_prompt_ablation_20260913/{v}'")
  (d/'code/summarize.py').write_text(summary)
  save(d/'policy.json',dict(text=text,spatial_prompt='exact selected anchor GT mask, no boxes or points',return_prompt='forward predicted mask, reversed RGB path, no target GT',same_paths=True,canvas=256,track='follow seeded object 0; full SAM3 grounding+tracking; begin after exact mask frame',no_text_detections='native allow_new_detections false when no text/geometry',text_detections='native allow_new_detections true; same seeded object followed',no_test_gt=True))
  save(d/'status.json',dict(stage='prepared'))
 support=read(OLD/'protocol/support_manifest.jsonl')
 for row in support:
  for k in ['image_path','mask_path']:hashes[row[k]]=sha(row[k])
 for p in [OLD/'SELECTED_SUPPORT_FROZEN.json',R/'mask_prompt.py',R/'adapter_source.py']:hashes[str(p)]=sha(p)
 save(R/'input_hashes.json',hashes);save(R/'status.json',dict(stage='prepared'));(R/'PREPARED').touch()

def worker(v,split,limit=0):
 import torch
 core=loadmodule('mask_core',R/'mask_prompt.py');q=core.quality;t21=core.t21
 d=R/v;rows=sorted(read(d/f'quality_root/{MODE}/{split}_pool0_stage1/routes.jsonl'),key=q.route_sort_key)
 if limit:rows=[rows[0],rows[6]]
 output=d/f'quality_root/{MODE}/propagation_quality_{split}';output=output if not limit else d/'smoke';output.mkdir(exist_ok=False,parents=True);(output/'forward_masks').mkdir()
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60)
 model=core.install(core.build_sam3_video_model(checkpoint_path=core.BASE,load_from_HF=False,device='cuda',compile=False));model.eval()
 for i,row in enumerate(rows,1):
  start=time.time();paths=[row['anchor_image_path'],*row['bridge_image_paths'],row['target_image_path']];seed=t21.load_mask(row['anchor_mask_path'],256);pred=core.propagate(model,paths,seed,VARIANTS[v]);m=pred.pop('final_mask');f=output/'forward_masks'/(row['route_id']+'.png');Image.fromarray(m.astype(np.uint8)*255).save(f)
  if m.any():
   back=core.propagate(model,list(reversed(paths)),m,VARIANTS[v]);cycle=t21.dice(back['final_mask'],seed);success=bool(back['final_candidate_count']);cycle_score=back['final_sam_score'];cycle_audit=back['prompt_audit'];failure=None if success else 'seeded_object_missing'
  else:cycle=0.;success=False;cycle_score=0.;cycle_audit=None;failure='empty_forward_mask'
  result=dict(row,status='success',forward_mask_path=str(f),forward_mask_sha256=sha(f),q_cycle=cycle,cycle_success=success,cycle_failure_reason=failure,cycle_candidate_count=int(success),cycle_sam_score=cycle_score,cycle_prompt_audit=cycle_audit,seconds=time.time()-start,**pred)
  t21.append_fsync(output/'propagation_quality.jsonl',result);print(f'{v} {split} {i}/{len(rows)} cycle={cycle:.4f}',flush=True)
  save(d/'status.json',dict(stage=split+'_propagation',count=i,total=len(rows)))
 if limit:save(d/'SMOKE_COMPLETE.json',dict(n=len(rows),forward_return_exact_mask=True))

def stage(v,name,cmd):
 d=R/v;save(R/'status.json',dict(variant=v,stage=name));env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONPATH='/Data_8TB/lht/sam3:'+str(P/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
 with (d/'logs'/f'{name}.log').open('x') as f:
  p=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);save(d/f'{name}_process.json',dict(pid=p.pid,cmd=cmd));rc=p.wait()
 if rc:raise RuntimeError(f'{v}/{name} failed {rc}')

def report():
 result={};old=json.loads((OLD/'test_results.json').read_text());oldrouter=json.loads((E/'busi_auto5_tp_router_20260913/results.json').read_text())['busi']
 for v in VARIANTS:result[v]=dict(test=json.loads((R/v/'test_results.json').read_text()),router=json.loads((R/(v+'_router')/'results.json').read_text())['busi'])
 equal={}
 for split in ['validation','test']:
  maps=[]
  for v in VARIANTS:
   maps.append({r['route_id']:r['forward_mask_sha256'] for r in read(R/v/f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl')})
  assert maps[0].keys()==maps[1].keys();equal[split]=dict(equal=sum(maps[0][k]==maps[1][k] for k in maps[0]),total=len(maps[0]))
 save(R/'results.json',dict(variants=result,old_box_test=old,old_box_router=oldrouter,text_vs_no_text_identical=equal))
 lines=['# BUSI：完整mask提示有/无文本对照','','固定自动5张参考图与原val64/test66路径。空间提示为所选参考GT完整mask，文本固定breast lesion。无target GT参与预测。两轮均跟踪mask指定的同一对象；原框版曾逐帧选择最高分对象，因此与原框版的比较同时包含提示和对象跟踪方式的改变。新两轮之间只切换文本。','','|桥长|原框无文本Test Dice|mask无文本|mask＋文本|','|---|---:|---:|---:|']
 for b in range(7):lines.append('|b'+str(b)+'|'+'|'.join(f'{x:.6f}' for x in [old['fixed_bridge'][str(b)]['dice'],*[result[v]['test']['fixed_bridge'][str(b)]['dice'] for v in VARIANTS]])+'|')
 for label,vals in [('Router',[oldrouter['test']['selected_router']['dice'],*[result[v]['router']['test']['selected_router']['dice'] for v in VARIANTS]]),('Oracle',[old['oracle'],*[result[v]['test']['oracle'] for v in VARIANTS]])]:lines.append('|'+label+'|'+'|'.join(f'{x:.6f}' for x in vals)+'|')
 lines+=['','Router每轮分别按相同的预设配置和validation图像分组交叉验证选择、拟合。test不参与选参。Oracle只是事后候选上限。',f'\n两轮逐候选mask完全一致数：{equal}', '\n原SAM3和历史预测均保留；原始GT数据不修改。', '\n实验根目录：`'+str(R)+'`']
 (R/'report.md').write_text('\n'.join(lines)+'\n')
 for p,h in json.loads((R/'input_hashes.json').read_text()).items():assert sha(p)==h,(p,'input changed')
 save(R/'status.json',dict(stage='complete'));(R/'COMPLETE').touch()

def run():
 assert (R/'PREPARED').exists();(R/'STARTED').open('x').close()
 # Validate forward AND return interfaces for both variants before full runs.
 for v in VARIANTS:stage(v,'smoke',[PY,str(R/'pipeline.py'),'worker',v,'validation','2'])
 for v in VARIANTS:
  for split in ['validation','test']:stage(v,split+'_propagation',[PY,str(R/'pipeline.py'),'worker',v,split])
  stage(v,'summarize',[PY,str(R/v/'code/summarize.py'),'summarize']);stage(v,'router',[PY,str(R/v/'code/run_router.py')]);save(R/v/'status.json',dict(stage='complete'));(R/v/'COMPLETE').touch()
 report()
if __name__=='__main__':
 try:
  action=sys.argv[1]
  if action=='prepare':prepare()
  elif action=='worker':worker(sys.argv[2],sys.argv[3],int(sys.argv[4]) if len(sys.argv)>4 else 0)
  elif action=='report':report()
  else:run()
 except BaseException:
  if R.exists():save(R/'status.json',dict(stage='failed',error=traceback.format_exc()))
  raise