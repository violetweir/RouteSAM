from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import json,hashlib,time,shutil,html,csv
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'new_project/experiments/automatic_anchor_selection_pilot_20260911'
F=P/'work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features'
R.mkdir(exist_ok=False)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def status(s):save(R/'status.json',dict(stage=s));print(s,flush=True)
policy=dict(primary_budget=8,budget_curve=[1,2,4,8,16,32],seed=2026,random_replicates=100,selection_inputs='only train RGB SAM3-base patch_mean and raw tokens; no GT, pseudo masks, conditioned features, val/test features or metrics',methods=['global_facility','global_local_facility'],local_features='64 visual token clusters learned from 64 spatially uniform tokens/image; Hellinger histogram, train-only IDF',similarity='0.5 max(global_cosine,0)+0.5 histogram_cosine',objective='greedy maximize mean nearest-selected similarity; stable id tie-break',duplicates='exact image-file SHA256 duplicates excluded from candidate set; no manual exclusion',mask_access='only after all selections and stopping results frozen; export selected GT for post-selection audit',automatic_stop='exploratory: gain/initial_single_anchor_distortion <0.01 for three consecutive additions, range4..32; no claim this ensures segmentation saturation',test_used=False)
save(R/'policy.json',policy)
status('loading_train_only_features')
protocol=P/'work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl'
allrows=[json.loads(s) for s in protocol.read_text().splitlines()]
cacheids=np.load(F/'sam3_base_s1008_patches.npz',allow_pickle=False)['ids'].tolist()
assert cacheids==[r['merged_id'] for r in allrows]
positions=sorted([i for i,r in enumerate(allrows) if r['split']=='train'],key=lambda i:allrows[i]['merged_id'])
assert len(positions)==800
# Deliberately remove mask paths before selection. They are reloaded only after freezing.
rows=[dict(id=allrows[i]['merged_id'],image_path=allrows[i]['image_path'],cache_index=i) for i in positions]
del allrows
old_manifest=json.loads((P/'new_project/kvasir_labeled_8/manifest.json').read_text())
oldids=[r['target_id'] for r in old_manifest['items']]
idmap={r['id']:i for i,r in enumerate(rows)};old=[idmap[i] for i in oldids]
del old_manifest
hashes=[sha(Path(r['image_path'])) for r in rows]
seen=set();eligible=[]
for i,h in enumerate(hashes):
    if h not in seen:eligible.append(i);seen.add(h)
assert len(eligible)>=32
g=norm(np.load(F/'sam3_base_s1008_features.npz',allow_pickle=False)['patch_mean'][positions].astype(np.float32))
raw=np.load(F/'sam3_base_s1008_patches_raw.npy',mmap_mode='r')
assert raw.shape==(1000,5184,1024)
grid=np.linspace(4,67,8).round().astype(int)
sample_idx=np.array([a*72+b for a in grid for b in grid])
sample=norm(np.stack([raw[i,sample_idx].astype(np.float32) for i in positions]))
# Verify raw-token cache and global descriptor share the same ordering/content.
for j in [0,199,399,599,799]:assert float(norm(raw[positions[j]].astype(np.float32).mean(0))@g[j])>.9999
save(R/'train_images_only.json',rows)
save(R/'input_provenance.json',dict(protocol_sha256=sha(protocol),raw_shape=list(raw.shape),raw_size_bytes=(F/'sam3_base_s1008_patches_raw.npy').stat().st_size,global_cache_sha256=sha(F/'sam3_base_s1008_features.npz'),train_image_hashes=dict(zip([r['id'] for r in rows],hashes)),eligible_count=len(eligible),ordering_checked=True))
status('fitting_train_only_local_dictionary')
km=MiniBatchKMeans(n_clusters=64,random_state=2026,n_init=3,batch_size=2048,max_iter=100).fit(sample.reshape(-1,1024))
assign=km.predict(sample.reshape(-1,1024)).reshape(800,64)
h=np.stack([np.bincount(x,minlength=64) for x in assign]).astype(np.float32)/64
idf=np.log((801)/(1+(h>0).sum(0)))+1
h=norm(np.sqrt(h*idf))
sg=np.clip(g@g.T,0,1);sl=np.clip(h@h.T,0,1);sm=(sg+sl)*.5
np.savez_compressed(R/'train_descriptors.npz',ids=np.array([r['id'] for r in rows]),global_features=g,local_histogram=h,local_dictionary=km.cluster_centers_,idf=idf)
def greedy(s):
    chosen=[];best=np.zeros(800,dtype=np.float32);trace=[]
    for k in range(32):
        gains=np.maximum(s-best[:,None],0).mean(0)
        allowed=np.zeros(800,dtype=bool);allowed[eligible]=True;allowed[chosen]=False
        gains[~allowed]=-np.inf
        j=int(np.argmax(gains));gain=float(gains[j]);chosen.append(j);best=np.maximum(best,s[:,j])
        trace.append(dict(k=k+1,index=j,id=rows[j]['id'],gain=gain,coverage=float(best.mean())))
    return chosen,trace
status('selecting_and_freezing_without_masks')
sets={};traces={};stops={}
for name,s in [('global_facility',sg),('global_local_facility',sm)]:
    chosen,trace=greedy(s);sets[name]=chosen;traces[name]=trace
    distortion=max(1-trace[0]['coverage'],1e-12);streak=0;stop=None
    for item in trace[1:]:
        streak=streak+1 if item['gain']/distortion<.01 else 0
        if streak>=3:stop=item['k'];break
    stops[name]=dict(k=stop,reached_within_32=stop is not None,threshold=.01,normalizer=distortion)
rng=np.random.default_rng(2026)
random_sets=[rng.choice(eligible,32,replace=False).tolist() for _ in range(100)]
frozen=dict(policy=policy,selected={name:[rows[j]['id'] for j in ix] for name,ix in sets.items()},greedy_traces=traces,stopping=stops,random_sets=[[rows[j]['id'] for j in ix] for ix in random_sets],mask_pixels_opened_before_freeze=0,time=time.time())
save(R/'SELECTIONS_FROZEN.json',frozen)
freeze_sha=sha(R/'SELECTIONS_FROZEN.json')
def metric(ix):
    out={}
    for name,s in [('global',sg),('local',sl),('combined',sm)]:
        nearest=s[:,ix].max(1);non=np.ones(800,dtype=bool);non[ix]=False
        assignment=np.argmax(s[:,ix],axis=1);counts=np.bincount(assignment,minlength=len(ix))
        pair=s[np.ix_(ix,ix)];v=pair[np.triu_indices(len(ix),1)]
        out[name]=dict(mean_coverage=float(nearest.mean()),unselected_mean_coverage=float(nearest[non].mean()),p10_coverage=float(np.quantile(nearest[non],.1)),pairwise_similarity=float(v.mean()) if len(v) else None,max_assignment_share=float(counts.max()/800),assignment_counts=counts.tolist())
    return out
metrics={'existing8':metric(old)};curves={};random_summary={}
for name,ix in sets.items():
    metrics[name]=metric(ix[:8]);curves[name]={str(k):metric(ix[:k]) for k in policy['budget_curve']}
for k in policy['budget_curve']:
    rr=[metric(ix[:k]) for ix in random_sets]
    random_summary[str(k)]={view:{key:dict(mean=float(np.mean([r[view][key] for r in rr])),p025=float(np.quantile([r[view][key] for r in rr],.025)),p975=float(np.quantile([r[view][key] for r in rr],.975))) for key in ['mean_coverage','unselected_mean_coverage','p10_coverage','max_assignment_share']} for view in ['global','local','combined']}
    if k==8:
        for name in metrics:
            metrics[name]['combined_percentile_vs_random']=float(np.mean([r['combined']['mean_coverage']<=metrics[name]['combined']['mean_coverage'] for r in rr]))
save(R/'coverage_metrics.json',dict(budget8=metrics,budget_curves=curves,random_summary=random_summary,stopping=stops,warning='Optimized feature coverage is not independent validation and is not segmentation Dice. No propagation performed.'))
status('post_freeze_selected_label_audit')
assert sha(R/'SELECTIONS_FROZEN.json')==freeze_sha
allrows=[json.loads(s) for s in protocol.read_text().splitlines()];byid={r['merged_id']:r for r in allrows}
audits={};gallery=[]
for name,ix in [('existing8',old)]+[(name,ix[:8]) for name,ix in sets.items()]:
    folder=R/name;(folder/'images').mkdir(parents=True);(folder/'masks').mkdir();items=[];cards=[]
    for number,j in enumerate(ix,1):
        r=byid[rows[j]['id']];assert r['split']=='train'
        image_src=Path(r['image_path']);mask_src=Path(r['mask_path']);stem=f'{number:02d}_{r["sample_id"]}.png'
        shutil.copy2(image_src,folder/'images'/stem);shutil.copy2(mask_src,folder/'masks'/stem)
        with Image.open(image_src) as im,Image.open(mask_src) as ma:
            m=np.array(ma.convert('L'))>127;ys,xs=np.where(m)
            item=dict(number=number,id=r['merged_id'],size=list(im.size),foreground_fraction=float(m.mean()),image_sha256=sha(image_src),mask_sha256=sha(mask_src),bbox_aspect=float((xs.max()-xs.min()+1)/(ys.max()-ys.min()+1)) if len(xs) else None)
        items.append(item)
        ip=f'{name}/images/{stem}';mp=f'{name}/masks/{stem}'
        cards.append(f'<article><h3>{number:02d} · {r["sample_id"]}</h3><p>GT占比 {item["foreground_fraction"]:.2%}</p><div class="pair"><img src="{ip}"><img src="{mp}"></div></article>')
    audits[name]=items;save(folder/'manifest.json',items)
    gallery.append(f'<section><h2>{name}</h2>'+''.join(cards)+'</section>')
save(R/'selected_gt_audit.json',audits)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自动参考图选择：8张预算</title><style>body{background:#111923;color:#edf3fa;font:16px system-ui;margin:25px}section{margin:30px 0;border-top:1px solid #567}article{display:inline-block;vertical-align:top;width:46%;padding:1%;margin:1%;background:#202b38;box-sizing:border-box}h3{font-size:14px;overflow-wrap:anywhere}.pair{display:flex;align-items:flex-start;gap:5px}.pair img{width:49%;height:auto}p{color:#b7c8dc}@media(max-width:800px){article{width:98%}}</style><h1>自动参考图选择：固定8张预算</h1><p>仅根据训练图像特征选图，名单冻结后才读取GT。GT用于本页展示，未用于选图。这是特征覆盖实验，不是分割性能结论。</p><p><a href="report.md">实验报告</a></p>'''+''.join(gallery)+'</html>'
(R/'index.html').write_text(page,encoding='utf-8')
report='''# 自动参考样本选择首轮实验

## 范围与协议

在原800张train内自动选择，val/test划分不变。原有8张标注方案只作对照，新名单独立保存，不替换当前A0训练。先冻结所有自动/随机选择名单与停止结果，再打开被选样本GT；未用隐藏GT、伪mask、原anchor条件特征或val/test特征选图。

## 方法

SAM3-base原图1008冻结特征。global_facility使用整图patch均值余弦；global_local_facility将整图余弦与局部视觉词直方图余弦各占一半。局部词典由800张train每图固定64个空间采样patch拟合64簇，训练集IDF加权后取平方根并归一化。它是无mask局部外观描述，不是病灶形态识别。每步选使全部train的最近参考相似度均值增加最多的图，按ID稳定破除并列。

去除完全相同文件SHA256的重复候选，不用人工挑选，也没有可靠的患者/视频ID来排除同病例近重复。100组随机对照固定seed2026；预算曲线1/2/4/8/16/32。缓存含全数据的独立图像编码，但算法只读取train的行；未加载anchor条件特征。

## 固定8张结果

覆盖指标越高表示在对应特征空间中越接近某个参考。下表使用排除所选样本自身后的792张均值。它不是Dice，优化目标上的改善也不是独立泛化证据。

|方法|整图覆盖|局部覆盖|综合覆盖|综合空间最大参考分配占比|
|---|---:|---:|---:|---:|
'''
for name,v in metrics.items():report+=f"|{name}|{v['global']['unselected_mean_coverage']:.6f}|{v['local']['unselected_mean_coverage']:.6f}|{v['combined']['unselected_mean_coverage']:.6f}|{v['combined']['max_assignment_share']:.2%}|\n"
rr=random_summary['8'];report+=f"|随机100组均值|{rr['global']['unselected_mean_coverage']['mean']:.6f}|{rr['local']['unselected_mean_coverage']['mean']:.6f}|{rr['combined']['unselected_mean_coverage']['mean']:.6f}|{rr['combined']['max_assignment_share']['mean']:.2%}|\n"
report+='\n注意：这里的参考分配是描述子最近邻，不是实际TP视频链源头比例。综合特征方法优化综合覆盖，不能把它在该指标上的优势直接解释为分割更好。\n\n## 自动停止探索\n\n预设规则：新增覆盖收益低于单张参考剩余平均距离的1%，连续3次才停止；最多观察32张。阈值未经分割验证，不能当作“最少足够标注数”。\n'
for name,x in stops.items():report+=f"- {name}: {x['k'] if x['k'] is not None else '32张内未达到停止条件'}。\n"
report+='\n## 冻结后GT形态审计\n\n'
for name,a in audits.items():
    z=[x['foreground_fraction'] for x in a];report+=f'- {name}：GT面积占比范围 {min(z):.2%}–{max(z):.2%}，中位数 {float(np.median(z)):.2%}。\n'
report+='\n## 当前结论的边界\n\n完成自动选图、随机对照、覆盖曲线与选后GT预览；尚未运行新参考集TP传播、Router或SAM3/学生训练，因此没有新的Val/Test分割Dice。下一步应冻结本轮主方案，在原val100检验候选Oracle与实际选择Dice，再决定是否优化选图策略；test留待方案固定。自动停止还需与传播可靠性结合。\n'
(R/'report.md').write_text(report,encoding='utf-8')
shutil.copy2(Path(__file__),R/'automatic_anchor_pilot.py')
save(R/'COMPLETE.json',dict(selection_frozen_sha256=freeze_sha,images_selected_per_method=8,train_only=True,old_experiments_unchanged=True))
status('complete')
print(report,flush=True)
