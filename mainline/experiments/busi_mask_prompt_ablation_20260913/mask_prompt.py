from pathlib import Path
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import sys,inspect,textwrap,types,json,time,hashlib,importlib.util
import numpy as np, torch
from PIL import Image
from sam3.model_builder import build_sam3_video_model
import sam3.model.sam3_video_inference as api
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');E=P/'new_project/experiments';OLD=E/'busi_auto5_tp_1pct_20260913';R=E/'busi_mask_prompt_ablation_20260913';MODE='sam3enc_anchor_conditioned_target_pooling';BASE='/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
spec=importlib.util.spec_from_file_location('quality',OLD/'code/eval_route_propagation_quality.py');quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality);t21=quality.t21

def install(model):
 # Retain official object registration/cache/memory bookkeeping. Replace only prompt decoder call.
 src=textwrap.dedent(inspect.getsource(api.Sam3VideoInferenceWithInstanceInteractivity.add_tracker_new_points))
 src=src.replace('def add_tracker_new_points(', 'def add_tracker_exact_mask(').replace('    points,\n    labels,','    mask,')
 start=src.index('self.tracker.add_new_points(');end=src.index('\n        )',start)
 old=src[start:end]
 assert 'points=points' in old
 replacement='self.tracker.add_new_mask(\n                inference_state=tracker_state, frame_idx=frame_idx, obj_id=obj_id,\n                mask=mask, add_mask_to_memory=True,\n            )'
 src=src[:start]+replacement+src[end:]
 src=src.replace('        self.clear_detector_added_cond_frame_in_tracker(\n            tracker_state, obj_id, frame_idx\n        )', '        # Preserve user mask conditioning frame.')
 ns=dict(vars(api));exec(compile(src,'exact_mask_adapter','exec'),ns)
 model.add_tracker_exact_mask=types.MethodType(ns['add_tracker_exact_mask'],model)
 (R/'adapter_source.py').write_text(src)
 return model

@torch.inference_mode()
def propagate(model,paths,seed,text):
 frames=[t21.load_rgb(p,256) for p in paths];state=model.init_state(resource_path=frames,offload_video_to_cpu=False,offload_state_to_cpu=False,async_loading_frames=False)
 # Configure native text inputs without any box/point or initial detector-generated mask.
 state['text_prompt']=text
 state['input_batch'].find_text_batch[0]=text if text else '<text placeholder>'
 for t in range(len(frames)):state['input_batch'].find_inputs[t].text_ids[...]=model.TEXT_ID_FOR_TEXT if text else model.TEXT_ID_FOR_VISUAL
 _,out=model.add_tracker_exact_mask(state,frame_idx=0,obj_id=0,mask=torch.from_numpy(seed.astype(np.float32)))
 ts=state['tracker_inference_states'][0];stored=ts['mask_inputs_per_obj'][0][0].cpu().numpy().squeeze();assert np.array_equal(stored,seed)
 assert not ts['point_inputs_per_obj'][0]
 assert all(x is None for x in state['per_frame_raw_box_input'])
 meta=state['tracker_metadata'];rm=meta['rank0_metadata'];rm['obj_first_frame_idx'][0]=0
 if 'masklet_confirmation' in rm:
  rm['masklet_confirmation']['status']=np.ones(1,dtype=np.int64)
  rm['masklet_confirmation']['consecutive_det_num']=np.full(1,model.masklet_confirmation_consecutive_det_thresh,dtype=np.int64)
 state['action_history'].clear() # run full SAM3 grounding+tracking, not instance-only path that ignores text detections
 masks=[seed];scores=[1.];counts=[1];detections=[]
 # Full propagation starts after the exact seed frame so it cannot replace the input mask.
 for idx,output in model.propagate_in_video(state,start_frame_idx=1,max_frame_num_to_track=len(frames)-1,reverse=False):
  assert idx==len(masks),(idx,len(masks));ids=np.asarray(output['out_obj_ids']);loc=np.flatnonzero(ids==0);detections.append(len(ids))
  if len(loc):
   j=int(loc[0]);masks.append(np.asarray(output['out_binary_masks'][j],dtype=bool));scores.append(float(output['out_probs'][j]));counts.append(1)
  else:masks.append(np.zeros_like(seed));scores.append(0.);counts.append(0)
 assert len(masks)==len(paths)
 audit=dict(seed_memory_exact=True,box_prompt_count=0,point_prompt_count=0,text=text,text_feature_cache_keys=[list(k) for k in state['feature_cache'].get('text',{})],objects_per_frame=detections)
 del state;torch.cuda.empty_cache()
 return dict(final_mask=masks[-1],final_candidate_count=counts[-1],final_sam_score=scores[-1],**quality.trace_features(masks,scores,counts),prompt_audit=audit)

def main():
 R.mkdir(exist_ok=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60)
 model=install(build_sam3_video_model(checkpoint_path=BASE,load_from_HF=False,device='cuda',compile=False));model.eval()
 rows=quality.read_jsonl(OLD/f'quality_root/{MODE}/validation_pool0_stage1/routes.jsonl')
 chosen=[r for r in rows if r['bridge_count'] in [0,6]][:2];out=[]
 for route in chosen:
  paths=[route['anchor_image_path'],*route['bridge_image_paths'],route['target_image_path']];seed=t21.load_mask(route['anchor_mask_path'],256)
  for text in [None,'breast lesion']:
   result=propagate(model,paths,seed,text);print(route['route_id'],text,result['prompt_audit'],float(result['final_mask'].mean()),flush=True)
   out.append(dict(route_id=route['route_id'],bridge=route['bridge_count'],text=text,audit=result['prompt_audit'],mask_sha256=hashlib.sha256(result['final_mask'].tobytes()).hexdigest(),area=float(result['final_mask'].mean())))
 (R/'smoke_results.json').write_text(json.dumps(out,indent=2));(R/'SMOKE_COMPLETE').touch()
if __name__=='__main__':main()