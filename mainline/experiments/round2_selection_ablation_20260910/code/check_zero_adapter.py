import train_direct as run
from sam3.model_builder import build_sam3_image_model
import numpy as np
import torch
from PIL import Image
import json
torch.set_num_threads(4)
model=build_sam3_image_model(device='cuda',compile=False,
 checkpoint_path='/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt',
 load_from_HF=False,bpe_path='/Data_8TB/lht/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz',eval_mode=False).eval()
model.num_interactive_steps_val=0
torch.set_autocast_cache_enabled(False)
run.evaluate(model,'validation','bare_base',limit=1)
bare=json.loads((run.D/'bare_base/PREDICTIONS_FROZEN.json').read_text())[0]
initial=json.loads((run.D/'zero_update/PREDICTIONS_FROZEN.json').read_text())[0]
a=np.asarray(Image.open(bare['mask_path']))>127;b=np.asarray(Image.open(initial['mask_path']))>127
n=int(a.sum())+int(b.sum());dice=2*int((a&b).sum())/n if n else 1.
result=dict(bare_vs_zero_adapter_dice=dice,differing_pixels=int((a!=b).sum()),passed=dice>.995)
run.save(run.R/'zero_adapter_audit.json',result)
assert result['passed'],result
print(result)
