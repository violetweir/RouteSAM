from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,argparse,gc
import numpy as np
import torch
from torchvision.transforms import functional as TF
from PIL import Image
from sam3.model_builder import _create_vit_backbone
import sam3.model.vitdet as vitdet
# The repository's inference fused MLP unconditionally casts to bfloat16.
# Use its training-path mathematical equivalent in this isolated CPU process.
def cpu_addmm_act(activation,linear,x):
    y=linear(x)
    if activation in [torch.nn.functional.gelu,torch.nn.GELU]:return torch.nn.functional.gelu(y)
    if activation in [torch.nn.functional.relu,torch.nn.ReLU]:return torch.nn.functional.relu(y)
    raise ValueError(activation)
vitdet.addmm_act=cpu_addmm_act

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');R=P/'work/kvasir_pc_evidence_20260908'
CKPT='/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt'
ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['validation','test'],default='validation');args=ap.parse_args()
torch.set_num_threads(4);torch.set_num_interop_threads(1)
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
records=read(P/'work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl');support=read(P/'work/kvasir_1pct_anchors/protocol/support_manifest.jsonl')
jobs={r['merged_id']:r for r in records if r['split']==args.split}
jobs.update({r['merged_id']:r for r in support});assert len(jobs)==108
R.mkdir(parents=True,exist_ok=True);dest=R/'tokens';dest.mkdir(exist_ok=True)
pending=[r for r in jobs.values() if not (dest/(r['merged_id'].replace('::','__')+'.npy')).exists()]
print('CPU-only extraction pending',len(pending),flush=True)
if pending:
    t0=time.time();model=_create_vit_backbone().eval()
    state=torch.load(CKPT,map_location='cpu',weights_only=True,mmap=True)
    if 'model' in state:state=state['model']
    prefix='detector.backbone.vision_backbone.trunk.'
    weights={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    assert len(weights)>100
    model.load_state_dict(weights,strict=True);del weights,state;gc.collect()
    for blk in model.blocks:
        attn=blk.attn
        if attn.use_rope and attn.freqs_cis is not None and blk.window_size==0:
            scale=attn.rope_pt_size[0]/18. if attn.rope_interp else 1.
            attn.register_buffer('freqs_cis',attn.compute_cis(end_x=18,end_y=18,scale_pos=scale).cpu())
    print('model ready seconds',time.time()-t0,flush=True)
    with torch.inference_mode():
        for i,r in enumerate(pending):
            x=TF.to_tensor(TF.resize(Image.open(r['image_path']).convert('RGB'),[256,256],interpolation=TF.InterpolationMode.BICUBIC,antialias=True))
            z=torch.nn.functional.normalize(model(((x-.5)/.5)[None])[0].flatten(2).permute(0,2,1),dim=-1)[0]
            a=z.numpy();assert a.shape==(324,1024) and np.isfinite(a).all()
            np.save(dest/(r['merged_id'].replace('::','__')+'.npy'),a)
            print('tokens',i+1,'/',len(pending),'elapsed',round(time.time()-t0,1),flush=True)
audit={'split':args.split,'target_count':100,'train_support_count':8,'device':'CPU float32, 4 threads','checkpoint':CKPT,'strict_trunk_weights':True,'token_shape':[324,1024],'cpu_mlp':'isolated replacement of forced-bfloat16 fused op with equivalent float32 Linear + GELU/ReLU','note':'New verification features use CPU float32. Historical candidate masks/router inputs unchanged; not a claim of bitwise reproduction of old GPU bfloat16 features.'}
(R/f'extraction_{args.split}.json').write_text(json.dumps(audit,indent=2)+'\n')
print('COMPLETE',args.split,flush=True)
