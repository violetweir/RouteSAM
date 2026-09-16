import json,traceback
import train_direct as d
old=d.R.parent/'round2_A1_A2_cosine_clip_original1008_20260910'
d.D=d.R/'inference_repeat'
def main():
 d.D.mkdir(exist_ok=False)
 src=old/'runs/A2/best.pt';digest=d.sha(src)
 d.save(d.D/'FROZEN.json',dict(checkpoint=str(src),sha256=digest,prompt='colon polyp',input='original1008',metric=256))
 d.torch.set_num_threads(4)
 trainer=d.t.SAM3TrainerNative(str(d.R/'A2.yaml'));model=trainer.model;model.num_interactive_steps_val=0
 d.torch.set_autocast_cache_enabled(False)
 d.load_adapter(model,d.torch.load(src,map_location='cpu',weights_only=True))
 result=d.evaluate(model,'test','test')
 assert d.sha(src)==digest
 new=json.loads((d.D/'test/per_target_metrics.json').read_text());prior=json.loads((old/'runs/A2/test_best/per_target_metrics.json').read_text())
 assert len(new)==len(prior)==100
 identical=0
 for a,b in zip(new,prior):
  assert a['target_id']==b['target_id']
  assert d.sha(d.Path(a['mask_path']))==a['mask_sha256']
  identical+=int(a['mask_sha256']==b['mask_sha256'])
 previous=json.loads((old/'runs/A2/results.json').read_text())['test']
 d.save(d.D/'results.json',dict(test=result,previous=previous,identical_mask_count=identical,dice_delta=result['dice']-previous['dice']))
 print(json.dumps(dict(result=result,identical_masks=identical)),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  if d.D.exists():(d.D/'FAILED.txt').write_text(traceback.format_exc())
  raise
