from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');R=P/'work/kvasir_pc_prompt_refine_20260908'
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def path(p):
    p=Path(p);return p if p.is_absolute() else P/p
rows=read(R/'per_target.jsonl');assert len(rows)==100
quality=read(P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6/sam3enc_anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl');qi={r['route_id']:r for r in quality}
ordered=sorted(rows,key=lambda r:r['outputs']['fgbg_box']['raw_dice']-r['tp_dice']);selected=[ordered[0],ordered[50],ordered[-1]]
fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');font=ImageFont.truetype(str(fontpath),15) if fontpath.exists() else ImageFont.load_default()
canvas=Image.new('RGB',(1280,3*306+45),'white');draw=ImageDraw.Draw(canvas)
draw.text((12,10),'GT-selected diagnostic examples: worst / median / best raw FG-BG change; not representative performance.',fill='black',font=font)
for i,r in enumerate(selected):
    q=qi[r['tp_route_id']];y=45+i*306;draw.text((5,y),r['target_id'],fill='black',font=font)
    sources=[q['target_image_path'],q['target_mask_path_evaluation_only'],q['forward_mask_path'],r['outputs']['tp_box_control']['mask_path'],r['outputs']['fgbg_box']['mask_path']]
    titles=['Input','GT',f"TP {r['tp_dice']:.4f}",f"TP-box {r['outputs']['tp_box_control']['raw_dice']:.4f}",f"FG-BG-box {r['outputs']['fgbg_box']['raw_dice']:.4f}"]
    for j,(src,title) in enumerate(zip(sources,titles)):
        im=Image.open(path(src)).convert('RGB').resize((256,256),Image.Resampling.BICUBIC if j==0 else Image.Resampling.NEAREST)
        canvas.paste(im,(j*256,y+45));draw.text((j*256+5,y+23),title,fill='black',font=font)
canvas.save(R/'examples.png')
(R/'example_selection.json').write_text(json.dumps({'policy':'worst/median/best raw FG-BG Dice difference vs TP, GT diagnostic only','target_ids':[r['target_id'] for r in selected]},indent=2)+'\n')
