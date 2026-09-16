from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np,json,hashlib
R=Path(__file__).resolve().parent;E=R.parent;B=E/'tn3k_busi_factorial_20260915';D=E/'tn3k_failure_diagnosis_20260916'
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
records=[json.loads(l) for l in (B/'protocol/merged_manifest.jsonl').read_text().splitlines()]
idx={r['merged_id']:i for i,r in enumerate(records)};train=[i for i,r in enumerate(records) if r['split']=='train']
z=np.load(B/'quality_root/features/sam3_base_s256_features.npz');aids=z['anchor_ids'].tolist();ai=[idx[a] for a in aids]
rows=read(D/'anchors_results.json');tids=[r['id'] for r in rows];ti=[idx[t] for t in tids];N=len(tids);A=len(aids)
y=np.array([[r['results'][a]['dice'] for a in aids] for r in rows]);foldmap=read(B/'folds_frozen.json');folds=np.array([foldmap[t] for t in tids])
raw=z['cond_target'][:,ti].T.astype(float);tr=z['cond_target'][:,train].astype(float);mu=tr.mean(1);sd=np.maximum(tr.std(1),1e-6);center=raw-mu;zs=center/sd
pc=z['cond_correspondence'][:,ti].T.astype(float);pctr=z['cond_correspondence'][:,train].astype(float);pcz=(pc-pctr.mean(1))/np.maximum(pctr.std(1),1e-6)
pm=z['patch_mean'].astype(float);glob=pm[ti]@pm[ai].T;gtr=pm[train]@pm[ai].T;gz=(glob-gtr.mean(0))/np.maximum(gtr.std(0),1e-6)
pct=np.stack([np.searchsorted(np.sort(tr[a]),raw[:,a],side='right')/len(train) for a in range(A)],1)
au={x['id']:x for x in read(D/'anchor_audit.json')};area=np.array([au[a]['area'] for a in aids]);patch=np.array([au[a]['foreground_patch_count_18'] for a in aids]);bc=lambda v:np.broadcast_to(v,(N,A))
features=np.stack([raw,center,zs,pct,pc,pcz,glob,gz,bc(mu),bc(sd),bc(area),bc(np.log(np.maximum(area,1e-6))),bc(patch/324),raw-glob,pc-raw,zs*gz],-1)
names=['raw','centered','zscore','train_percentile','pc','pc_zscore','global_cosine','global_zscore','anchor_train_mean','anchor_train_std','anchor_area','log_anchor_area','anchor_patch_fraction','raw_minus_global','pc_minus_raw','zscore_times_global_zscore']
scores={'raw':raw,'centered':center,'zscore':zs,'train_percentile':pct,'global_cosine':glob,'half_zscore_global':.5*(zs+gz)}
fitinfo={}
for method in ['anchor_quality_prior_oof','ridge10_numeric_oof','ridge10_numeric_anchor_oof']:
 pred=np.zeros_like(y);info=[]
 for fold in sorted(set(folds)):
  tridx=np.flatnonzero(folds!=fold);teidx=np.flatnonzero(folds==fold)
  if method=='anchor_quality_prior_oof':
   prior=(y[tridx].sum(0)+10*y[tridx].mean())/(len(tridx)+10);pred[teidx]=prior;info.append({'fold':int(fold),'train_targets':len(tridx),'validation_targets':len(teidx),'prior':prior.tolist()});continue
  xtr=features[tridx].reshape(-1,len(names));xte=features[teidx].reshape(-1,len(names));mean=xtr.mean(0);std=np.maximum(xtr.std(0),1e-6);xtr=(xtr-mean)/std;xte=(xte-mean)/std
  if method.endswith('anchor_oof'):
   xtr=np.concatenate([xtr,np.tile(np.eye(A),(len(tridx),1))],1);xte=np.concatenate([xte,np.tile(np.eye(A),(len(teidx),1))],1)
  yt=y[tridx].reshape(-1);ym=yt.mean();w=np.linalg.solve(xtr.T@xtr+10*np.eye(xtr.shape[1]),xtr.T@(yt-ym));pred[teidx]=(xte@w+ym).reshape(len(teidx),A)
  info.append({'fold':int(fold),'train_targets':len(tridx),'validation_targets':len(teidx),'feature_mean':mean.tolist(),'feature_std':std.tolist(),'weights':w.tolist(),'intercept':float(ym)})
 scores[method]=pred;fitinfo[method]=info
ks=[1,2,4,8,16,23];oracle=y.max(1);out={};per={};rng=np.random.default_rng(20260916);bootidx=rng.integers(0,N,(10000,N))
for name,sc in scores.items():
 orders=np.array([sorted(range(A),key=lambda a:(-sc[i,a],aids[a])) for i in range(N)]);chosen=y[np.arange(N),orders[:,0]]
 detail={'selected_dice':float(chosen.mean()),'topK':{}};p={'selected':chosen.tolist(),'ranked_anchor_ids':[[aids[a] for a in order] for order in orders]}
 for k in ks:
  vals=np.take_along_axis(y,orders[:,:k],axis=1).max(1);detail['topK'][str(k)]={'oracle_dice':float(vals.mean()),'oracle_gap':float((oracle-vals).mean()),'best_reference_recall':float(np.mean(vals>=oracle-1e-12)),'targets_without_dice_ge_05':int(np.sum(vals<.5))};p['oracle_k'+str(k)]=vals.tolist()
 out[name]=detail;per[name]=p
comparisons={}
for name in scores:
 diff=np.array(per[name]['selected'])-np.array(per['raw']['selected']);boot=diff[bootidx].mean(1);comparisons[name+'_minus_raw_top1']={'delta':float(diff.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
res={'n':N,'anchors':A,'fold_sizes':{str(f):int((folds==f).sum()) for f in set(folds)},'all23_b0_oracle':float(oracle.mean()),'methods':out,'paired_exploratory_comparisons':comparisons,'warning':'All results on previously inspected validation64; learned methods target-grouped OOF; no test evaluation, no proof of generalization.'}
save(R/'results.json',res);save(R/'per_target.json',{'target_ids':tids,'anchor_ids':aids,'true_dice':y.tolist(),'methods':per});save(R/'fold_models.json',{'feature_names':names,'models':fitinfo});np.savez_compressed(R/'features_and_scores.npz',features=features,labels=y,folds=folds,**scores)
lines=['# TN3K 参考图排序与Top-K诊断','', '固定原64张验证图、23张参考、全部b0候选；不读取test。监督方法沿用原validation图像级5折，每个目标的23候选必须在同一折。仅使用传播前特征，不把目标GT面积、候选Dice或预测mask特征作为输入。训练折候选Dice作为排序监督，属于额外validation监督，不是仅1%训练标注。','', '本实验在已查看过的验证子集上开发，所有比较为探索性；不据此选择test最优。岭回归alpha固定10，无超参数搜索；参考质量先验向训练折总平均收缩10个目标。','', '| 方法 | 实际Top-1 Dice | Top-2 Oracle | Top-4 Oracle | Top-8 Oracle | Top-16 Oracle |','|---|---:|---:|---:|---:|---:|']
for name,r in out.items():lines.append('| '+name+' | '+f"{r['selected_dice']:.6f}"+' | '+' | '.join(f"{r['topK'][str(k)]['oracle_dice']:.6f}" for k in [2,4,8,16])+' |')
lines+=['',f'全部23参考b0 Oracle：{oracle.mean():.6f}。Top-K Oracle仅表示候选保留能力，不能作为实际部署Dice。','', '## 方法定义','- raw / centered：原始TP / TP减参考图训练均值。','- zscore：再除参考图训练分数标准差。','- train_percentile：该目标分数在该参考训练分数分布的百分位。','- global_cosine：目标与参考的全图特征余弦。','- half_zscore_global：标准化TP与标准化全图相似度各一半。','- anchor_quality_prior_oof：仅使用训练折每张参考的平均传播质量，判断是否主要是参考通用质量问题。','- ridge10_numeric_oof：16维参考-目标特征学习候选Dice，按预测值排序。','- ridge10_numeric_anchor_oof：在上述特征上加入参考身份one-hot；同一固定参考库中使用。','', '## 复现','使用 /home/violet/anaconda3/envs/sam3/bin/python 本目录/run.py。输入来自 tn3k_failure_diagnosis_20260916/anchors_results.json 以及原冻结TN3K特征和folds。无需新增GPU推理。policy_frozen.json固定实验范围，fold_models.json保存各折模型及标准化。','', '注意：Top-K扩大后的实际最终选择仍需独立评估；不能用Oracle补齐实际方法。']
(R/'参考排序诊断.md').write_text('\n'.join(lines)+'\n');(R/'COMPLETE').touch();print(json.dumps(res,indent=2))
