"""Paired SAM3 direct image LoRA; pure text inference and separate GT metrics."""
import argparse,copy,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True);parser.add_argument('--gpu',required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from PIL import Image as PIL
import train_sam3_lora_kvasir_e50 as t
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');R=P/'new_project/experiments/round2_A0_A1_repeat_seed2027_20260910'
D=R/('smoke' if args.smoke else 'runs')/args.arm
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def append(p,x):
 with p.open('a') as f:f.write(json.dumps(x)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def tensor_hash(items):
 h=hashlib.sha256()
 for n,p in sorted(items):h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def adapter(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
def load_adapter(model,state):
 params=dict(model.named_parameters());assert set(state)=={n for n,p in params.items() if p.requires_grad}
 with torch.no_grad():
  for n,v in state.items():params[n].copy_(v.to(params[n].device))
 torch.clear_autocast_cache()
def checkpoint(model,path):
 tmp=path.with_suffix('.tmp');torch.save(adapter(model),tmp);tmp.replace(path)
def move(x):
 if torch.is_tensor(x):return x.cuda(non_blocking=True)
 if isinstance(x,list):return [move(v) for v in x]
 if isinstance(x,tuple):return tuple(move(v) for v in x)
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 if hasattr(x,'__dataclass_fields__'):
  for k in x.__dataclass_fields__:setattr(x,k,move(getattr(x,k)))
 return x
def assert_text_only(batch):
 assert list(batch.find_text_batch)==[TEXT_PROMPT]
 for q in batch.find_inputs:
  assert q.input_boxes is None or q.input_boxes.numel()==0
  assert q.input_points is None or q.input_points.numel()==0

TEXT_PROMPT = 'colon polyp'

class DirectDataset(Dataset):
 def __init__(self,rows):self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  image=PIL.open(r['file_name']).convert('RGB').resize((1008,1008),PIL.Resampling.BILINEAR)
  a=np.array(image,dtype=np.float32)/255
  x=torch.from_numpy(np.ascontiguousarray(((a-.5)/.5).transpose(2,0,1)))
  # Constant dummy target satisfies the training dataclass, never actual GT.
  dummy=torch.zeros(1008,1008,dtype=torch.bool);dummy[504,504]=True
  obj=t.Object(bbox=torch.tensor([.5,.5,.01,.01]),area=.0001,object_id=0,segment=dummy)
  query=t.FindQueryLoaded(query_text=TEXT_PROMPT,image_id=0,object_ids_output=[0],is_exhaustive=True,query_processing_order=0,
      inference_metadata=t.InferenceMetadata(coco_image_id=i,original_image_id=i,original_category_id=0,original_size=(256,256),object_id=-1,frame_index=-1))
  return t.Datapoint(find_queries=[query],images=[t.Image(data=x,objects=[obj],size=(1008,1008))],raw_images=[image])
def collate(batch):return t.collate_fn_api(batch,dict_key='input',with_seg_masks=True)
def outputs(model,batch,grad=False):
 assert_text_only(batch);assert model.num_interactive_steps_val==0
 if grad:return model(batch)
 with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return model(batch)
def final_output(out):
 with t.SAM3Output.iteration_mode(out,iter_mode=t.SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as it:return list(it)[-1][-1]
def prediction(final):
 scores=final['pred_logits'].float().squeeze(-1)[0].sigmoid();masks=final['pred_masks'].float()[0].sigmoid();valid=scores>=.5
 prob=masks[valid].amax(0) if bool(valid.any()) else torch.zeros_like(masks[0])
 return (F.interpolate(prob[None,None],size=(256,256),mode='bilinear',align_corners=False)[0,0]>=.5).cpu().numpy()

@torch.no_grad()
def evaluate(model,split,tag,limit=None):
 model.eval();rows=json.loads((R/f'data/shared/{split}.json').read_text());rows=rows[:limit] if limit else rows
 dest=D/tag;(dest/'masks').mkdir(parents=True,exist_ok=False)
 loader=DataLoader(DirectDataset(rows),batch_size=1,shuffle=False,num_workers=0 if limit else 2,collate_fn=collate)
 masks=[]
 for i,bat in enumerate(loader):
  m=prediction(final_output(outputs(model,move(bat['input']))));path=dest/'masks'/(rows[i]['merged_id'].replace('::','__')+'.png')
  PIL.fromarray((m*255).astype(np.uint8)).save(path);masks.append(dict(target_id=rows[i]['merged_id'],mask_path=str(path),mask_sha256=sha(path)))
 save(dest/'PREDICTIONS_FROZEN.json',masks)
 # Evaluation labels are opened only after every prediction is saved.
 metrics=[]
 for r,p in zip(rows,masks):
  m=np.array(PIL.open(p['mask_path']))>127;g=np.array(PIL.open(r['mask_file_name']).convert('L').resize((256,256),PIL.Resampling.NEAREST))>127
  inter=int((m&g).sum());total=int(m.sum())+int(g.sum());union=total-inter
  metrics.append({**p,'dice':2*inter/total if total else 1.,'iou':inter/union if union else 1.})
 save(dest/'per_target_metrics.json',metrics)
 result=dict(count=len(metrics),dice=float(np.mean([r['dice'] for r in metrics])),iou=float(np.mean([r['iou'] for r in metrics])))
 save(dest/'summary.json',result);return result

def train_batch(trainer,bat):
 model=trainer.model;batch=move(bat['input']);out=outputs(model,batch,grad=True)
 targets=[model.back_convert(x) for x in batch.find_targets]
 with t.SAM3Output.iteration_mode(out,iter_mode=t.SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as it:
  for stage,target in zip(it,targets):
   for o in stage:
    o['indices']=trainer.matcher(o,target)
    for aux in o.get('aux_outputs',[]):aux['indices']=trainer.matcher(aux,target)
 losses=trainer.loss_wrapper(out,targets);loss=losses[t.CORE_LOSS_KEY]
 assert torch.isfinite(loss),'nonfinite loss'
 trainer.optimizer.zero_grad(set_to_none=True);loss.backward()
 grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
 assert grads and all(torch.isfinite(g).all() for g in grads),'nonfinite or absent gradients'
 trainer.optimizer.step();torch.clear_autocast_cache()
 return float(loss.detach())

def summarize_if_ready(arms):
 if not all((R/'runs'/a/'results.json').exists() for a in arms):return
 results={a:json.loads((R/'runs'/a/'results.json').read_text()) for a in arms}
 baseline={x['target_id']:x for x in json.loads((R/'runs'/arms[0]/'test_best/per_target_metrics.json').read_text())}
 alternative={x['target_id']:x for x in json.loads((R/'runs'/arms[1]/'test_best/per_target_metrics.json').read_text())}
 assert len(baseline)==100 and set(baseline)==set(alternative)
 delta=np.array([alternative[k]['dice']-baseline[k]['dice'] for k in sorted(baseline)])
 rng=np.random.default_rng(2026);ci=np.quantile(delta[rng.integers(0,100,size=(10000,100))].mean(1),[.025,.975])
 comparison=dict(mean_dice_delta=float(delta.mean()),paired_ci95=ci.tolist(),wins=int((delta>1e-12).sum()),losses=int((delta< -1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()))
 save(R/'training_results.json',dict(arms=results,comparison=comparison))
 lines=['# Round2 SAM3 直接分割：训练池对比结果','','两组采用相同SAM3-base、LoRA初始化、10个epoch和硬标签；各自用完整validation选择epoch0..10中的best，均冻结后才评估直接test。','',
  '| 训练组 | 训练图总数（含8GT） | Best epoch | Val Dice | Test Dice | Test IoU |','|---|---:|---:|---:|---:|---:|']
 for a,r in results.items():lines.append(f"| {a} | {r['training_pool_size']} | {r['best_epoch']} | {r['validation']['dice']:.6f} | {r['test']['dice']:.6f} | {r['test']['iou']:.6f} |")
 lines += ['',f"{arms[1]} 相对 {arms[0]} 的test Dice差：{delta.mean():+.6f}；配对bootstrap95%区间 [{ci[0]:+.6f}, {ci[1]:+.6f}]。",f"逐图胜/负/平：{comparison['wins']}/{comparison['losses']}/{comparison['ties']}。",'',
 '推理仅使用图像与统一colon polyp文本，未使用GT点/框、TP路径、Router或B7。预测保存后再读取GT评估；指标为256×256逐图宏平均。',
 '训练池数量不同，相同epoch对应不同更新次数；结果是池配方的实际效果比较，不能单独归因为某一筛选条件。','']
 text='\n'.join(lines);(R/'training_report.md').write_text(text,encoding='utf-8');(P/'new_project/round2_selection_training_results.md').write_text(text,encoding='utf-8')

def main():
 D.mkdir(parents=True,exist_ok=False);save(D/'status.json',dict(stage='initializing',arm=args.arm,gpu=args.gpu))
 torch.set_num_threads(4);random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
 trainer=t.SAM3TrainerNative(str(R/f'{args.arm}.yaml'));model=trainer.model;model.num_interactive_steps_val=0
 torch.set_autocast_cache_enabled(False);torch.clear_autocast_cache()
 initial=adapter(model);initial_hash=tensor_hash(initial.items());frozen_hash=tensor_hash((n,p) for n,p in model.named_parameters() if not p.requires_grad)
 save(D/'initialization.json',dict(adapter_hash=initial_hash,frozen_hash=frozen_hash,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),gpu=args.gpu))
 dataset=t.COCOSegmentDataset(R/'data'/args.arm,'train');records=json.loads((R/'data'/args.arm/'records.json').read_text())
 assert len(dataset)==len(records) and sum(r['is_gt'] for r in records)==8
 if args.smoke:
  before=evaluate(model,'validation','zero_update',limit=1)
  first=next(iter(DataLoader(dataset,batch_size=1,num_workers=0,collate_fn=collate)))
  model.train();loss=train_batch(trainer,first)
  assert tensor_hash(adapter(model).items())!=initial_hash
  assert tensor_hash((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen_hash
  checkpoint(model,D/'updated.pt');saved=adapter(model);load_adapter(model,initial)
  load_adapter(model,torch.load(D/'updated.pt',map_location='cpu',weights_only=True));assert tensor_hash(adapter(model).items())==tensor_hash(saved.items())
  after=evaluate(model,'validation','updated',limit=1)
  save(D/'smoke.json',dict(passed=True,loss=loss,zero_update=before,updated=after,gradients_finite=True,frozen_unchanged=True,checkpoint_roundtrip=True))
  save(D/'status.json',dict(stage='smoke_complete'));return
 arms=json.loads((R/'POOLS_FROZEN.json').read_text())['training_arms'];other=next(a for a in arms if a!=args.arm)
 deadline=time.time()+600
 while not (R/'runs'/other/'initialization.json').exists():
  if time.time()>deadline:raise RuntimeError('Other arm initialization unavailable')
  time.sleep(5)
 oi=json.loads((R/'runs'/other/'initialization.json').read_text());assert oi['adapter_hash']==initial_hash and oi['frozen_hash']==frozen_hash
 checkpoint(model,D/'epoch0.pt');best=evaluate(model,'validation','validation_epoch0');best_epoch=0
 checkpoint(model,D/'best.pt');save(D/'best.json',dict(epoch=0,**best));step=0;start=time.time()
 for epoch in range(1,11):
  order=np.random.default_rng(2026+epoch).permutation(len(dataset)).tolist()
  loader=DataLoader(dataset,batch_size=1,sampler=order,num_workers=2,collate_fn=collate,pin_memory=True)
  model.train();losses=[]
  for j,bat in enumerate(loader):
   value=train_batch(trainer,bat);losses.append(value);step+=1
   if j==0 or step%20==0:
    status=dict(stage='training',arm=args.arm,epoch=epoch,epochs=10,epoch_step=j+1,epoch_size=len(dataset),step=step,loss=value,elapsed_seconds=time.time()-start,best_validation_dice=best['dice'])
    save(D/'status.json',status);print(json.dumps(status),flush=True)
  assert len(losses)==len(dataset) and len(set(order))==len(dataset)
  checkpoint(model,D/f'epoch{epoch}.pt')
  torch.save(dict(epoch=epoch,step=step,adapter=adapter(model),optimizer=trainer.optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),D/'latest_training.pt')
  save(D/'status.json',dict(stage='validation',epoch=epoch,step=step))
  val=evaluate(model,'validation',f'validation_epoch{epoch}')
  if val['dice']>best['dice']:
   best=val;best_epoch=epoch;checkpoint(model,D/'best.pt');save(D/'best.json',dict(epoch=epoch,**best))
  append(D/'epochs.jsonl',dict(epoch=epoch,step=step,train_count=len(dataset),gt_count=8,pseudo_count=len(dataset)-8,loss=float(np.mean(losses)),validation=val,best_epoch=best_epoch))
 assert tensor_hash((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen_hash
 save(D/'WEIGHTS_FROZEN.json',dict(best_epoch=best_epoch,validation=best,best_sha256=sha(D/'best.pt'),frozen_parameters_unchanged=True,total_steps=step))
 save(D/'status.json',dict(stage='waiting_for_other_arm',best_epoch=best_epoch,validation=best))
 model.cpu();trainer.optimizer.state.clear();torch.cuda.empty_cache()
 while not (R/'runs'/other/'WEIGHTS_FROZEN.json').exists():
  if (R/'runs'/other/'FAILED.txt').exists():raise RuntimeError('Other arm failed; test held until comparison can finish')
  time.sleep(30)
 model.cuda();load_adapter(model,torch.load(D/'best.pt',map_location='cpu',weights_only=True));save(D/'status.json',dict(stage='test',best_epoch=best_epoch))
 result=evaluate(model,'test','test_best')
 save(D/'results.json',dict(arm=args.arm,best_epoch=best_epoch,validation=best,test=result,training_pool_size=len(dataset),test_direct=True))
 save(D/'status.json',dict(stage='complete',best_epoch=best_epoch,test=result));(D/'COMPLETE').write_text('complete\n')
 summarize_if_ready(arms)

if __name__=='__main__':
 try:main()
 except BaseException:
  D.mkdir(parents=True,exist_ok=True);(D/'FAILED.txt').write_text(traceback.format_exc());save(D/'status.json',dict(stage='failed',error=traceback.format_exc()));raise
