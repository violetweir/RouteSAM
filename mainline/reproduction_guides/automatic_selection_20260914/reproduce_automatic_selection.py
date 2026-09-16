"""三个数据集的自动选图复现入口；仅写入显式指定的新目录。"""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import argparse, collections, hashlib, json, shutil, sys, time
import numpy as np

P = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
E = P / 'new_project/experiments'
MODE = 'sam3enc_anchor_conditioned_target_pooling'
SPECS = {
    'kvasir': ('automatic_anchor_tp_test_20260913/kvasir', 'automatic_anchor_selection_pilot_20260911', 8, (800,100,100)),
    'isic2018': ('automatic_anchor_tp_test_20260913/isic2018', 'isic2018_auto21_tp_validation_20260911', 21, (2075,259,260)),
    'busi': ('busi_auto5_tp_1pct_20260913', 'busi_auto5_tp_1pct_20260913', 5, (517,64,66)),
}
def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def norm(x): return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def source(name):
    base,selection,k,counts=SPECS[name]
    return E/base,E/selection,k,counts

def select(args):
    from sklearn.cluster import MiniBatchKMeans
    base,src,k,counts=source(args.dataset)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((src/'train_images_only.json').read_text())
    assert len(rows)==counts[0]
    ids=[r['id'] for r in rows]
    assert ids==sorted(ids) and len(set(ids))==len(ids)
    if args.features:
        z=np.load(args.features); assert z['ids'].tolist()==ids
        g=norm(z['global_features'].astype(np.float32)); sample=norm(z['sampled_patches'].astype(np.float32))
        desc=src/('train_descriptors.npz' if args.dataset=='kvasir' else 'selection/descriptors.npz' if args.dataset=='busi' else 'selection/train_descriptors.npz')
        frozen=src/('SELECTIONS_FROZEN.json' if args.dataset=='kvasir' else 'selection/SELECTIONS_FROZEN.json')
    elif args.dataset=='kvasir':
        f=P/'work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features'
        positions=[r['cache_index'] for r in rows]
        cacheids=np.load(f/'sam3_base_s1008_patches.npz')['ids'].tolist()
        assert [cacheids[i] for i in positions]==ids
        g=norm(np.load(f/'sam3_base_s1008_features.npz')['patch_mean'][positions].astype(np.float32))
        raw=np.load(f/'sam3_base_s1008_patches_raw.npy',mmap_mode='r')
        grid=np.linspace(4,67,8).round().astype(int)
        idx=np.array([a*72+b for a in grid for b in grid])
        sample=norm(np.stack([raw[i,idx].astype(np.float32) for i in positions]))
        desc=src/'train_descriptors.npz'
        frozen=src/'SELECTIONS_FROZEN.json'
    else:
        z=np.load(src/'selection_features.npz')
        assert z['ids'].tolist()==ids
        g=norm(z['global_features'].astype(np.float32)); sample=norm(z['sampled_patches'].astype(np.float32))
        desc=src/('selection/descriptors.npz' if args.dataset=='busi' else 'selection/train_descriptors.npz')
        frozen=src/'selection/SELECTIONS_FROZEN.json'
    n=len(rows); assert g.shape==(n,1024) and sample.shape==(n,64,1024)
    hashes=[sha(r['image_path']) for r in rows]; seen=set(); eligible=[]
    for i,h in enumerate(hashes):
        if h not in seen: eligible.append(i); seen.add(h)
    km=MiniBatchKMeans(n_clusters=64,random_state=2026,n_init=3,batch_size=2048,max_iter=100).fit(sample.reshape(-1,1024))
    labels=km.predict(sample.reshape(-1,1024)).reshape(n,64)
    h=np.stack([np.bincount(x,minlength=64) for x in labels]).astype(np.float32)/64
    idf=np.log((n+1)/(1+(h>0).sum(0)))+1; h=norm(np.sqrt(h*idf))
    sim=(np.clip(g@g.T,0,1)+np.clip(h@h.T,0,1))*.5
    best=np.zeros(n,dtype=np.float32); chosen=[]; trace=[]
    for step in range(k):
        gain=np.maximum(sim-best[:,None],0).mean(0)
        allowed=np.zeros(n,dtype=bool); allowed[eligible]=True; allowed[chosen]=False; gain[~allowed]=-np.inf
        j=int(np.argmax(gain)); chosen.append(j); best=np.maximum(best,sim[:,j])
        trace.append(dict(k=step+1,id=ids[j],gain=float(gain[j]),coverage=float(best.mean())))
    selected=[ids[i] for i in chosen]; f=json.loads(frozen.read_text())
    expected=f['selected_ids'] if args.dataset=='busi' else f['selected']['global_local_facility'][:k]
    old=np.load(desc)
    result=dict(dataset=args.dataset,selected_ids=selected,expected_ids=expected,exact_match=selected==expected,
                trace=trace,eligible_count=len(eligible),train_images=len(rows),mask_pixels_read=0,
                dictionary_max_abs=float(np.max(abs(old['local_dictionary']-km.cluster_centers_))),
                histogram_max_abs=float(np.max(abs(old['local_histogram']-h))),
                source_frozen_sha256=sha(frozen),source_descriptor_sha256=sha(desc),
                train_image_hashes=dict(zip(ids,hashes)))
    save(out/'selection_replay.json',result)
    save(out/'SELECTED_IDS_FROZEN.json',dict(selected_ids=selected,time=time.time(),test_or_val_used=False))
    np.savez_compressed(out/'descriptors.npz',ids=np.array(ids),global_features=g,local_histogram=h,local_dictionary=km.cluster_centers_,idf=idf)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trace','train_image_hashes')},ensure_ascii=False),flush=True)
    assert selected==expected, '选图名单与历史不一致；先检查环境、输入顺序及浮点差异，不要继续传播。'

def extract(args):
    # 仅重新编码训练 RGB，不读取 GT。显卡由外部 CUDA_VISIBLE_DEVICES 指定。
    import gc, torch
    base,src,_,counts=source(args.dataset)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(base/'code'))
    import stage1_feature_knn_routes as route
    from sam3.model_builder import build_sam3_video_model
    rows=json.loads((src/'train_images_only.json').read_text()); assert len(rows)==counts[0]
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.40)
    model=build_sam3_video_model(checkpoint_path=route.SAM3_CKPT,load_from_HF=False,device='cpu',compile=False)
    trunk=model.detector.backbone.vision_backbone.trunk; del model; gc.collect(); trunk=trunk.cuda().eval()
    grid=np.linspace(4,67,8).round().astype(int); idx=np.array([a*72+b for a in grid for b in grid])
    means=[]; samples=[]
    with torch.no_grad():
        for i,r in enumerate(rows):
            x=((route.load_rgb_tensor(r['image_path'],1008)-.5)/.5).unsqueeze(0).cuda()
            feat=trunk(x)[0]; tok=torch.nn.functional.normalize(feat.flatten(2).permute(0,2,1),dim=-1)[0]
            assert tok.shape==(5184,1024)
            means.append(torch.nn.functional.normalize(tok.mean(0),dim=0).float().cpu().numpy())
            samples.append(tok[idx].float().cpu().numpy()); del x,feat,tok
            if (i+1)%25==0: print(i+1,len(rows),flush=True)
    np.savez_compressed(out/'selection_features.npz',ids=np.array([r['id'] for r in rows]),global_features=np.stack(means),sampled_patches=np.stack(samples))
    save(out/'extraction.json',dict(train_images=len(rows),size=1008,mask_pixels_read=0,checkpoint=route.SAM3_CKPT,checkpoint_sha256=sha(route.SAM3_CKPT)))

def prepare(args):
    base,_,k,counts=source(args.dataset); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    for d in ['code','protocol','logs','quality_root/features']: (out/d).mkdir(parents=True)
    for fn in ['stage1_feature_knn_routes.py','eval_route_propagation_quality.py','run_t21_dynamic_pseudovideo.py']:
        shutil.copy2(base/'code'/fn,out/'code'/fn)
    for fn in ['merged_manifest.jsonl','support_manifest.jsonl']: shutil.copy2(base/'protocol'/fn,out/'protocol'/fn)
    rr=read(out/'protocol/merged_manifest.jsonl'); support=read(out/'protocol/support_manifest.jsonl')
    assert collections.Counter(r['split'] for r in rr)==dict(zip(['train','validation','test'],counts))
    assert len(support)==k and all(r['split']=='train' for r in support)
    if args.selection:
        selected=json.loads((Path(args.selection)/'SELECTED_IDS_FROZEN.json').read_text())['selected_ids']
        assert selected==[r['merged_id'] for r in support]
    if not args.fresh_features:
        shutil.copy2(base/'quality_root/features/sam3_base_s256_features.npz',out/'quality_root/features/sam3_base_s256_features.npz')
    files=list((out/'code').glob('*.py'))+list((out/'protocol').glob('*'))+list((out/'quality_root/features').glob('*'))
    save(out/'reproduction_inputs.json',dict(dataset=args.dataset,source=str(base),fresh_features=args.fresh_features,
         hashes={str(f.relative_to(out)):sha(f) for f in files},selected_ids=[r['merged_id'] for r in support]))
    print('Prepared',out,flush=True)

def evaluate(args):
    from PIL import Image
    base,_,_,counts=source(args.dataset); out=Path(args.out); split=args.split
    n=counts[1 if split=='validation' else 2]
    q=out/f'quality_root/{MODE}/propagation_quality_{split}/propagation_quality.jsonl'
    rows=read(q); targets={r['merged_id'] for r in read(out/'protocol/merged_manifest.jsonl') if r['split']==split}
    assert len(targets)==n and len(rows)==n*7 and all(r['status']=='success' for r in rows)
    assert collections.Counter((r['target_id'],r['bridge_count']) for r in rows)==collections.Counter({(t,b):1 for t in targets for b in range(7)})
    save(out/f'{split}_PREDICTIONS_FROZEN.json',dict(quality_sha256=sha(q),time=time.time(),target_GT_not_read_yet=True))
    def mask(p):
        p=Path(p); p=p if p.is_absolute() else P/p
        return np.asarray(Image.open(p).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
    scored=[]; gt={}
    for r in rows:
        assert not r['target_gt_used_for_search_or_inference']
        mp=Path(r['forward_mask_path']); mp=mp if mp.is_absolute() else P/mp
        assert sha(mp)==r['forward_mask_sha256']
        if r['target_id'] not in gt: gt[r['target_id']]=mask(r['target_mask_path_evaluation_only'])
        a=mask(mp); b=gt[r['target_id']]; i=int((a&b).sum()); z=int(a.sum())+int(b.sum())
        scored.append(dict(target_id=r['target_id'],route_id=r['route_id'],bridge_count=r['bridge_count'],dice=2*i/z if z else 1.,iou=i/(z-i) if z-i else 1.))
    fixed={str(b):dict(dice=float(np.mean([r['dice'] for r in scored if r['bridge_count']==b])),iou=float(np.mean([r['iou'] for r in scored if r['bridge_count']==b]))) for b in range(7)}
    result=dict(n=n,fixed_bridge=fixed,oracle=float(np.mean([max(r['dice'] for r in scored if r['target_id']==t) for t in targets])))
    save(out/f'{split}_per_candidate_metrics.json',scored); save(out/f'{split}_results.json',result)
    if split=='test':
        if args.dataset=='busi': expected=json.loads((base/'test_results.json').read_text())
        else: expected=json.loads((base/'results.json').read_text())['results']['automatic']
        result['max_abs_fixed_dice_difference']=max(abs(fixed[str(b)]['dice']-expected['fixed_bridge'][str(b)]['dice']) for b in range(7))
        save(out/f'{split}_results.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['extract','select','prepare','evaluate'])
    ap.add_argument('--dataset',required=True,choices=list(SPECS))
    ap.add_argument('--out',required=True)
    ap.add_argument('--selection',help='select 子命令输出目录；prepare 时核验名单')
    ap.add_argument('--features',help='select 时改用新提取的1008特征 npz；默认用历史原始特征缓存')
    ap.add_argument('--fresh-features',action='store_true',help='不复制256缓存，后续路径命令重新提取')
    ap.add_argument('--split',choices=['validation','test'],default='test')
    args=ap.parse_args(); globals()[args.action](args)
