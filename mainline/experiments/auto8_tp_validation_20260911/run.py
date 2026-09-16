from pathlib import Path
import os,sys,json,hashlib,subprocess,shutil,traceback,collections,time
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'new_project/experiments/auto8_tp_validation_20260911'
S=P/'new_project/experiments/automatic_anchor_selection_pilot_20260911'
MODE='sam3enc_anchor_conditioned_target_pooling'
PY='/home/violet/anaconda3/envs/sam3/bin/python'
read=lambda p:json.loads(p.read_text())
lines=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def status(stage,**kw):save(R/'status.json',dict(stage=stage,**kw));print(stage,kw,flush=True)
def run(name,cmd):
 status(name,command=cmd)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONPATH='/Data_8TB/lht/sam3:'+str(P/'src'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 with (R/f'{name}.log').open('x') as f:
  proc=subprocess.Popen(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT)
  save(R/f'{name}_process.json',dict(pid=proc.pid,command=cmd,gpu=1))
  rc=proc.wait()
 assert rc==0,(name,rc)
def prepare():
 R.mkdir(exist_ok=False)
 for name in ['code','protocol','baseline']:(R/name).mkdir()
 frozen=read(S/'SELECTIONS_FROZEN.json');ids=frozen['selected']['global_local_facility'][:8]
 assert len(set(ids))==8 and (S/'COMPLETE.json').exists()
 original=P/'work/kvasir_1pct_anchors/protocol'
 allrows=lines(original/'merged_manifest.jsonl');byid={r['merged_id']:r for r in allrows}
 records=[r for r in allrows if r['split'] in ['train','validation']]
 assert collections.Counter(r['split'] for r in records)==dict(train=800,validation=100)
 support=[]
 for ident in ids:
  r=byid[ident].copy();assert r['split']=='train'
  r.update(frozen_image_path=r['image_path'],frozen_mask_path=r['mask_path'])
  support.append(r)
 for name,rr in [('merged_manifest.jsonl',records),('support_manifest.jsonl',support)]:
  (R/'protocol'/name).write_text(''.join(json.dumps(r)+'\n' for r in rr))
 (R/'protocol/frozen_labeled_images.txt').write_text('\n'.join(r['image_path'] for r in support)+'\n')
 for name in ['stage1_feature_knn_routes.py','eval_route_propagation_quality.py','run_t21_dynamic_pseudovideo.py']:
  text=(P/'scripts'/name).read_text()
  if name!='run_t21_dynamic_pseudovideo.py':text=text.replace('T21_PATH = ROOT / "scripts/run_t21_dynamic_pseudovideo.py"','T21_PATH = Path(__file__).resolve().parent / "run_t21_dynamic_pseudovideo.py"')
  (R/'code'/name).write_text(text)
 baseline=P/'work/kvasir_tp_filterfirst_students_20260909'
 shutil.copy2(baseline/'router.json',R/'router.json')
 shutil.copy2(baseline/'code/router.py',R/'code/router.py')
 shutil.copy2(baseline/'quality/anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl',R/'baseline/propagation_quality.jsonl')
 shutil.copy2(Path(__file__),R/'run.py')
 shutil.copy2(Path(__file__).with_name('auto8_tp_val_summarize.py'),R/'code/summarize.py')
 ckpt=Path('/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt')
 policy=dict(selected_ids=ids,selection_frozen_sha256=sha(S/'SELECTIONS_FROZEN.json'),feature_size=256,propagation_canvas=256,selection_feature_size=1008,beam_width=32,knn='patch_mean',routes=700,targets=100,split='validation',checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),gpu=1,router_sha256=sha(R/'router.json'),router_training_caveat='Original frozen router fitted on same val100; diagnostic, not held-out performance',scope='replace 8 anchors only; no student, B7, router refitting, test, or change to active A0 training',gt_policy='route search uses selected train GT only; propagation no-target-gt; freeze all masks and router decisions before GT scoring',features='fresh SAM3-base256 descriptors for train800+val100; no test images loaded')
 save(R/'policy.json',policy)
 save(R/'input_hashes.json',{str(p):sha(p) for p in [S/'SELECTIONS_FROZEN.json',baseline/'router.json',baseline/'quality/anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl',original/'support_manifest.jsonl']})
 (R/'experiment.md').write_text('''# 自动选出的8张参考图：TP验证集实验

固定上一轮global_local_facility前8张名单。保持原train800/val100/test100划分，本轮只提取train800+val100特征并传播val100，不读取test图像或GT。

与原TP对照保持SAM3-base、检索特征256、patch_mean KNN、beam32、b0-b6共7候选、传播画布256、原独立Router不变。选图算法使用1008特征，这与传播特征尺度不同；当前目的是在原基线设置下单独检验新参考图。

新8张GT仅用于参考图原型、初始化框和返回一致性；生成700条路由后冻结，传播时不读取val GT。全部预测及Router选择冻结后才计算val100逐图宏平均Dice、Oracle、固定b0-b6分数与源头分布。

原Router曾在完整val100拟合，本轮不重拟合，Router结果只能作为开发集描述性对照。没有足够随机参考集传播重复，不能把单次优势直接归因为自动选图普遍有效。

GPU1顺序执行特征提取/路径生成、700条前向+返回传播、冻结Router选mask与评分。GPU0的A0训练继续运行。所有结果独立保存。
''',encoding='utf-8')
 status('prepared')
def main():
 policy=read(R/'policy.json')
 run('features_and_routes',[PY,str(R/'code/stage1_feature_knn_routes.py'),'--mode',MODE,'--feature-source','sam3_base','--feature-size','256','--knn-feature','patch_mean','--beam-width','32','--min-bridge','0','--max-bridge','6','--split','validation','--protocol-root',str(R/'protocol'),'--output-root',str(R/'quality_root')])
 routefile=R/f'quality_root/{MODE}/validation_pool0_stage1/routes.jsonl';rr=lines(routefile)
 records=lines(R/'protocol/merged_manifest.jsonl');train={r['merged_id'] for r in records if r['split']=='train'};val={r['merged_id'] for r in records if r['split']=='validation'}
 assert len(rr)==700 and len({r['route_id'] for r in rr})==700
 assert collections.Counter((r['target_id'],r['bridge_count']) for r in rr)==collections.Counter({(t,b):1 for t in val for b in range(7)})
 for r in rr:
  assert r['anchor_id'] in policy['selected_ids'] and set(r['bridge_ids'])<=train and not r['target_gt_used_for_search_or_inference']
 save(R/'ROUTES_FROZEN.json',dict(route_sha256=sha(routefile),count=700,anchors=dict(collections.Counter(r['anchor_id'] for r in rr)),time=time.time()))
 run('propagation',[PY,str(R/'code/eval_route_propagation_quality.py'),'--checkpoint',policy['checkpoint'],'--mode',MODE,'--root',str(R/'quality_root'),'--split','validation','--canvas','256','--no-target-gt'])
 run('summarize',[PY,str(R/'code/summarize.py')])
 for p,h in read(R/'input_hashes.json').items():assert sha(Path(p))==h
 status('complete',results=read(R/'results.json'))
 (R/'COMPLETE').write_text('complete\n')
if __name__=='__main__':
 if sys.argv[1:]==['prepare']:
  prepare()
  with (R/'pipeline.log').open('x') as log:
   p=subprocess.Popen([PY,str(R/'run.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (R/'pipeline.pid').write_text(str(p.pid));print('started',p.pid)
 else:
  try:main()
  except BaseException:
   (R/'FAILED.txt').write_text(traceback.format_exc());status('failed',error=traceback.format_exc());raise
