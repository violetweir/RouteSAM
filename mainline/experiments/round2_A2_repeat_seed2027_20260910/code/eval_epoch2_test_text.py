import json,shutil,traceback
import train_direct as d
d.D=d.R/'audits/epoch2_test_text'
def main():
 d.D.mkdir(parents=True,exist_ok=False)
 src=d.R/'runs/A2/epoch2.pt';dst=d.D/'epoch2.pt';shutil.copy2(src,dst);digest=d.sha(dst);assert digest==d.sha(src)
 val=json.loads((d.R/'runs/A2/validation_epoch2/summary.json').read_text())
 d.save(d.D/'FROZEN.json',dict(epoch=2,seed=2027,sha256=digest,validation=val,prompt='colon polyp',input='original1008',metric=256,reason='user requested current validation best during training'))
 d.torch.set_num_threads(4)
 trainer=d.t.SAM3TrainerNative(str(d.R/'A2.yaml'));model=trainer.model;model.num_interactive_steps_val=0
 d.torch.set_autocast_cache_enabled(False)
 d.load_adapter(model,d.torch.load(dst,map_location='cpu',weights_only=True))
 result=d.evaluate(model,'test','test')
 assert result['count']==100 and d.sha(src)==digest
 d.save(d.D/'results.json',dict(epoch=2,seed=2027,validation=val,test=result,checkpoint_sha256=digest))
 (d.D/'COMPLETE').write_text('complete\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  if d.D.exists():(d.D/'FAILED.txt').write_text(traceback.format_exc())
  raise
