from pathlib import Path
import os,json,subprocess,shutil,hashlib,collections,traceback
R=Path(__file__).resolve().parent
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
PY='/home/violet/anaconda3/envs/sam3/bin/python'
CPU='/home/violet/anaconda3/envs/mkunet_mamba/bin/python'
MODE='sam3enc_anchor_conditioned_target_pooling'
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def run(name,cmd):
 save(R/'status.json',dict(stage=name));print(name,flush=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/Data_8TB/lht/sam3:'+str(P/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
 with (R/f'{name}.log').open('x') as log:
  p=subprocess.Popen(cmd,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT)
  save(R/f'{name}_process.json',dict(pid=p.pid,command=cmd));rc=p.wait()
 if rc:raise RuntimeError(f'{name} failed, exit={rc}')
def main():
 run('selection_features',[PY,str(R/'code/isic_auto_extract.py')])
 run('automatic_selection',[CPU,str(R/'code/anchor_selection.py')])
 F=R/'selection';f=json.loads((F/'SELECTIONS_FROZEN.json').read_text());ids=f['selected']['global_local_facility'][:21]
 records=[json.loads(s) for s in (R/'protocol/merged_manifest.jsonl').read_text().splitlines()];byid={r['merged_id']:r for r in records};support=[]
 for ident in ids:
  r=byid[ident].copy();assert r['split']=='train';r.update(frozen_image_path=r['image_path'],frozen_mask_path=r['mask_path']);support.append(r)
 (R/'protocol/support_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in support))
 save(R/'SELECTED_SUPPORT_FROZEN.json',dict(ids=ids,count=21,selection_sha256=sha(F/'SELECTIONS_FROZEN.json')))
 run('features_and_routes',[PY,str(R/'code/stage1_feature_knn_routes.py'),'--mode',MODE,'--feature-source','sam3_base','--feature-size','256','--knn-feature','patch_mean','--beam-width','32','--min-bridge','0','--max-bridge','6','--split','validation','--protocol-root',str(R/'protocol'),'--output-root',str(R/'quality_root')])
 file=R/f'quality_root/{MODE}/validation_pool0_stage1/routes.jsonl';routes=[json.loads(s) for s in file.read_text().splitlines()]
 train={r['merged_id'] for r in records if r['split']=='train'};val={r['merged_id'] for r in records if r['split']=='validation'}
 assert len(routes)==1813 and len({r['route_id'] for r in routes})==1813
 assert collections.Counter((r['target_id'],r['bridge_count']) for r in routes)==collections.Counter({(t,b):1 for t in val for b in range(7)})
 for r in routes:assert r['anchor_id'] in ids and set(r['bridge_ids'])<=train and not r['target_gt_used_for_search_or_inference']
 save(R/'ROUTES_FROZEN.json',dict(count=1813,sha256=sha(file),source_counts=dict(collections.Counter(r['anchor_id'] for r in routes))))
 run('propagation',[PY,str(R/'code/eval_route_propagation_quality.py'),'--checkpoint','/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt','--mode',MODE,'--root',str(R/'quality_root'),'--split','validation','--canvas','256','--no-target-gt'])
 run('summarize',[PY,str(R/'code/summarize.py')])
 save(R/'status.json',dict(stage='complete'));(R/'COMPLETE').write_text('complete\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  (R/'FAILED.txt').write_text(traceback.format_exc());save(R/'status.json',dict(stage='failed',error=traceback.format_exc()));raise
