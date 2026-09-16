from pathlib import Path
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import sys,importlib.util,json,time,collections
import numpy as np,torch
from PIL import Image
from sam3.model_builder import build_sam3_video_model
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');E=P/'new_project/experiments';B=E/'busi_auto5_tp_1pct_20260913';R=E/'busi_failure_diagnosis_20260913'
spec=importlib.util.spec_from_file_location('quality',B/'code/eval_route_propagation_quality.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);t21=q.t21
BASE='/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
def save(p,a):Path(p).write_text(json.dumps(a,indent=2,ensure_ascii=False)+'\n')
def evaluate(pred,gt):
 m=np.array(Image.fromarray(pred.astype(np.uint8)*255).resize((256,256),Image.Resampling.NEAREST))>127;return t21.metrics(m,gt)['dice']
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.70);model=build_sam3_video_model(checkpoint_path=BASE,load_from_HF=False,device='cuda',compile=False);model.eval()
 records=q.read_jsonl(B/'protocol/merged_manifest.jsonl');val=sorted([r for r in records if r['split']=='validation'],key=lambda r:r['merged_id']);supports=q.read_jsonl(B/'protocol/support_manifest.jsonl');out=[]
 for i,row in enumerate(val):
  gt=t21.load_mask(row['mask_path'],256);per={}
  for name,size,text,gtbox in [('target_gtbox_256',256,None,True),('target_gtbox_native_to1008',1008,None,True),('text_only_breast_lesion_1008',1008,'breast lesion',False)]:
   box=t21.tight_box(t21.load_mask(row['mask_path'],size)) if gtbox else None
   state=model.init_state(resource_path=[t21.load_rgb(row['image_path'],size)],offload_video_to_cpu=False,offload_state_to_cpu=False,async_loading_frames=False)
   _,output=model.add_prompt(state,frame_idx=0,text_str=text,boxes_xywh=[box] if box else None,box_labels=[1] if box else None);selected=t21.select_top(output,size);per[name]=dict(dice=evaluate(selected['mask'],gt),candidates=selected['candidate_count'],score=selected['sam_score']);del state;torch.cuda.empty_cache()
  peranchor={}
  for a in supports:
   anchor_mask=t21.load_mask(a['mask_path'],512);box=t21.tight_box(anchor_mask);trace=q.propagate_with_trace(model,[a['image_path'],row['image_path']],box,256);peranchor[a['merged_id']]=dict(dice=evaluate(trace['final_mask'],gt),score=trace['final_sam_score'])
  out.append(dict(id=row['merged_id'],direct=per,per_anchor_b0=peranchor));save(R/'gpu_progress.json',dict(count=i+1,total=len(val)));print(i+1,len(val),per,flush=True);save(R/'gpu_per_target_partial.json',out)
 z=np.load(B/'quality_root/features/sam3_base_s256_features.npz');ids=[r['merged_id'] for r in records];ix=[ids.index(r['id']) for r in out];train=[i for i,r in enumerate(records) if r['split']=='train'];c=z['cond_target'].astype(float);aids=z['anchor_ids'].tolist();cm=c[:,train].mean(1,keepdims=True);cs=np.maximum(c[:,train].std(1,keepdims=True),1e-8)
 selection={}
 for name,scores in [('raw',c),('centered',c-cm),('zscore',(c-cm)/cs)]:
  chosen=scores[:,ix].argmax(0);selection[name]=dict(dice=float(np.mean([r['per_anchor_b0'][aids[j]]['dice'] for r,j in zip(out,chosen)])),counts=dict(collections.Counter(aids[j] for j in chosen)))
 result=dict(n=len(val),direct={k:dict(dice=float(np.mean([r['direct'][k]['dice'] for r in out])),no_candidates=sum(r['direct'][k]['candidates']==0 for r in out)) for k in out[0]['direct']},anchor_b0={a:float(np.mean([r['per_anchor_b0'][a]['dice'] for r in out])) for a in aids},anchor_b0_oracle=float(np.mean([max(x['dice'] for x in r['per_anchor_b0'].values()) for r in out])),anchor_score_selection=selection,warning='Validation-only diagnosis. Target GT box tests use extra location supervision and are NOT 1% results; alternative selection scores not frozen independent evaluations.')
 save(R/'gpu_results.json',result);save(R/'gpu_per_target.json',out);(R/'GPU_COMPLETE').touch();print(json.dumps(result),flush=True)
if __name__=='__main__':main()