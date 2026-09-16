import collections,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'new_project/experiments/tp_tracker_endpoint_20260910'
S=P/'new_project/experiments/tp_student_rescreen_20260910'
TP=P/'work/kvasir_tp_filterfirst_students_20260909'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def jsonl(p,x):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
def main():
 assert json.loads((R/'preflight_public_v3/summary.json').read_text())['passed']
 dest=R/'pilot';dest.mkdir(exist_ok=False)
 labels=read(S/'data/train_manifest.jsonl');assert len(labels)==628
 support=read(TP/'protocol/support_manifest.jsonl');assert len(support)==8
 protocol=read(TP/'protocol/merged_manifest.jsonl');train_ids={r['merged_id'] for r in protocol if r['split']=='train'}
 groups=collections.defaultdict(list)
 q=TP/'quality/anchor_conditioned_target_pooling/propagation_quality_train/propagation_quality.jsonl'
 keep=['target_id','route_id','bridge_count','anchor_id','bridge_ids','anchor_image_path','bridge_image_paths',
       'target_image_path','anchor_mask_path','anchor_box_xywh_normalized','forward_mask_path','forward_mask_sha256']
 for row in read(q):groups[row['target_id']].append({k:row[k] for k in keep})
 support=sorted(support,key=lambda r:r['merged_id']);by_support={r['merged_id']:r for r in support}
 rows=[];excluded_empty=0
 pseudo_index=0
 for label in sorted(labels,key=lambda r:r['target_id']):
  assert sha(label['hard_label_path'])==label['hard_sha256']
  assert sha(label['soft_label_path'])==label['soft_sha256']
  target=label['target_id']
  if not label['is_gt']:
   candidates=sorted(groups[target],key=lambda r:r['bridge_count']);assert len(candidates)==7
   offset=pseudo_index%7;pseudo_index+=1;candidates=candidates[offset:]+candidates[:offset]
   accepted=[]
   for c in candidates:
    assert sha(c['forward_mask_path'])==c['forward_mask_sha256']
    if (np.asarray(Image.open(c['forward_mask_path']).convert('L'))>127).any():accepted.append(c)
    else:excluded_empty+=1
   candidates=accepted
  else:
   target_meta=by_support[target];others=[r for r in support if r['merged_id']!=target];candidates=[]
   for anchor in others:
    m=np.asarray(Image.open(anchor['mask_path']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
    ys,xs=np.where(m);x0,x1=xs.min(),xs.max();y0,y1=ys.min(),ys.max()
    candidates.append(dict(target_id=target,route_id='gt_endpoint_'+hashlib.sha256((anchor['merged_id']+target).encode()).hexdigest()[:24],
       bridge_count=0,anchor_id=anchor['merged_id'],bridge_ids=[],anchor_image_path=anchor['image_path'],bridge_image_paths=[],
       target_image_path=target_meta['image_path'],anchor_mask_path=anchor['mask_path'],
       anchor_box_xywh_normalized=[float(x0/256),float(y0/256),float((x1-x0+1)/256),float((y1-y0+1)/256)]))
  assert candidates
  for c in candidates:
   frame_ids=[c['anchor_id'],*c['bridge_ids'],target]
   assert set(frame_ids)<=train_ids and c['anchor_id']!=target
   assert c['anchor_id'] in by_support
  rows.append(dict(target_id=target,is_gt=label['is_gt'],label_path=label['hard_label_path'] if label['is_gt'] else label['soft_label_path'],
                   label_sha256=label['hard_sha256'] if label['is_gt'] else label['soft_sha256'],paths=candidates,weight=1.))
 jsonl(dest/'train_targets.jsonl',rows)
 valq=TP/'quality/anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl'
 validation=[{k:v for k,v in row.items() if 'evaluation_only' not in k and not k.startswith('gt_')} for row in read(valq)]
 assert len(validation)==700
 jsonl(dest/'validation_routes.jsonl',validation)
 cfg=dict(epochs=1,train_targets=628,gt=8,pseudo=620,seed=2026,batch_sequences=1,weights='all 1',
   train_pool='Frozen existing 620 soft labels plus 8 GT; isolate training-interface change; no new B7 threshold',
   terminal_labels='existing soft target for pseudo; hard for GT; same label across routes',
   path_policy='each target once; pseudo primary bridges balanced by sorted index modulo7, skip frozen-base empty paths; GT uses distinct GT anchor b0',
   fallback='if current public output has no tracked selectable object, try next predeclared path; one optimizer update per target; fail if none',
   initialization='/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt',
   lora=dict(rank=4,alpha=8,dropout=0.,policy='memory_attention_decoder_lora'),
   optimizer='AdamW',lr=1e-5,weight_decay=.01,grad_clip=1.,loss='terminal per-image BCE + SoftDice smooth1; no auxiliary loss',
   forward='original public TP API in eval behavior with current adapter; terminal replay on detached current-model history',
   backpropagation='terminal step only (truncated BPTT length1); memory attention and decoder receive gradients; no full-chain BPTT',
   precision='BF16 autocast, cache disabled to avoid inference casts severing gradients or stale updated weights',
   validation='all100 targets x7 fixed TP paths; recompute forward and cycle, frozen student best and original B7; compare with base',
   validation_checkpoint_selection='higher full validation B7 Dice between epoch0 base and epoch1; ties keep base',
   test='not evaluated during this pilot',
   base_validation_b7=.851807170563205,base_validation_oracle=.869544177933916,
   excluded_base_empty_paths=excluded_empty,primary_bridge_counts=dict(collections.Counter(r['paths'][0]['bridge_count'] for r in rows)),
   input_hashes={str(p):sha(p) for p in [q,valq,S/'data/train_manifest.jsonl',S/'runs/soft/student_best.pth',S/'runs/soft/validation_best/per_target_metrics.jsonl']})
 save(dest/'config.json',cfg);print(json.dumps(cfg,indent=2))
 (dest/'PREPARED').write_text('ready\n')
if __name__=='__main__':main()
