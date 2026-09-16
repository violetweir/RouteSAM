from pathlib import Path
import os,gc,json,time
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch,numpy as np
import stage1_feature_knn_routes as route
from sam3.model_builder import build_sam3_video_model
R=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.40)
rows=json.loads((R/'train_images_only.json').read_text())
assert len(rows)==517 and all(r['split']=='train' for r in rows)
model=build_sam3_video_model(checkpoint_path=route.SAM3_CKPT,load_from_HF=False,device='cpu',compile=False)
trunk=model.detector.backbone.vision_backbone.trunk
del model;gc.collect()
trunk=trunk.cuda().eval()
idx=np.array([a*72+b for a in np.linspace(4,67,8).round().astype(int) for b in np.linspace(4,67,8).round().astype(int)])
means=[];samples=[]
start=time.time()
with torch.no_grad():
 for i,r in enumerate(rows):
  x=((route.load_rgb_tensor(r['image_path'],1008)-.5)/.5).unsqueeze(0).cuda()
  feat=trunk(x)[0];tok=torch.nn.functional.normalize(feat.flatten(2).permute(0,2,1),dim=-1)[0]
  means.append(torch.nn.functional.normalize(tok.mean(0),dim=0).float().cpu().numpy())
  samples.append(tok[idx].float().cpu().numpy())
  del x,feat,tok
  if i==0 or (i+1)%25==0:
   info=dict(stage='selection_features',count=i+1,total=len(rows),elapsed_seconds=time.time()-start,allocated_gb=torch.cuda.memory_allocated()/2**30,reserved_gb=torch.cuda.memory_reserved()/2**30)
   (R/'feature_progress.json').write_text(json.dumps(info));print(json.dumps(info),flush=True)
np.savez_compressed(R/'selection_features.npz',ids=np.array([r['id'] for r in rows]),global_features=np.stack(means),sampled_patches=np.stack(samples))
print('FEATURES_COMPLETE',flush=True)
