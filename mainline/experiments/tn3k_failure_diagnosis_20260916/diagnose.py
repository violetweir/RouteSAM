from pathlib import Path
import os,sys,json,time,importlib.util
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np,torch
from PIL import Image
from sam3.model_builder import build_sam3_video_model
R=Path(__file__).resolve().parent
B=R.parent/'tn3k_busi_factorial_20260915'
BASE='/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
spec=importlib.util.spec_from_file_location('quality',R/'code/eval_route_propagation_quality.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);t21=q.t21
mode=sys.argv[1]
def save(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def metrics(pred,gt):
 p=np.asarray(Image.fromarray(pred.astype(np.uint8)*255).resize((256,256),Image.Resampling.NEAREST))>127
 i=int((p&gt).sum());a=int(p.sum());b=int(gt.sum())
 return dict(dice=2*i/max(a+b,1),precision=i/max(a,1),recall=i/max(b,1),area_ratio=a/max(b,1))
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.7)
 model=build_sam3_video_model(checkpoint_path=BASE,load_from_HF=False,device='cuda',compile=False);model.eval()
 val=json.loads((R/'sample_frozen.json').read_text())['rows'];supports=q.read_jsonl(B/'protocol/support_manifest.jsonl')
 cache={(r['target_id'],r['anchor_id']):r for r in q.read_jsonl(B/'quality_root/sam3enc_anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl') if r['bridge_count']==0 and r['status']=='success'}
 out=[];maskdir=R/(mode+'_masks');maskdir.mkdir(exist_ok=True)
 for j,row in enumerate(val):
  gt=t21.load_mask(row['mask_path'],256);res={};tid=row['merged_id'];short=tid.split('::')[-1]
  if mode=='direct':
   for size in [256,1008]:
    box=t21.tight_box(t21.load_mask(row['mask_path'],size))
    state=model.init_state(resource_path=[t21.load_rgb(row['image_path'],size)],offload_video_to_cpu=False,offload_state_to_cpu=False,async_loading_frames=False)
    _,output=model.add_prompt(state,frame_idx=0,text_str=None,boxes_xywh=[box],box_labels=[1]);selected=t21.select_top(output,size)
    res[str(size)]={**metrics(selected['mask'],gt),'candidates':selected['candidate_count']}
    Image.fromarray(selected['mask'].astype(np.uint8)*255).save(maskdir/f'{short}_{size}.png');del state;torch.cuda.empty_cache()
  else:
   for a in supports:
    aid=a['merged_id'];old=cache.get((tid,aid))
    if old:
     pred=t21.load_mask(old['forward_mask_path'],256);path=old['forward_mask_path']
    else:
     box=t21.tight_box(t21.load_mask(a['mask_path'],512));trace=q.propagate_with_trace(model,[a['image_path'],row['image_path']],box,256);pred=trace['final_mask']
     path=str(maskdir/f"{short}_{aid.split('::')[-1]}.png");Image.fromarray(pred.astype(np.uint8)*255).save(path)
    res[aid]={**metrics(pred,gt),'reused':bool(old),'mask_path':path}
  out.append({'id':tid,'results':res});save(R/(mode+'_partial.json'),out);save(R/(mode+'_progress.json'),{'done':j+1,'total':len(val),'time':time.time()});print(mode,j+1,len(val),flush=True)
 save(R/(mode+'_results.json'),out);(R/(mode+'_COMPLETE')).touch()
if __name__=='__main__':main()
