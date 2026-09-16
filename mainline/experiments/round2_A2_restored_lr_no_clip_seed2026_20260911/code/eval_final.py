import json, shutil, traceback
import train_direct as d
d.D=d.R/'audits/epoch10_test'

def main():
    d.D.mkdir(parents=True,exist_ok=False)
    src=d.R/'runs/seed2026/epoch10.pt'
    dst=d.D/'epoch10.pt'
    shutil.copy2(src,dst)
    digest=d.sha(dst)
    assert digest==d.sha(src)
    val=json.loads((d.R/'runs/seed2026/validation_epoch10/summary.json').read_text())
    d.save(d.D/'FROZEN.json',dict(epoch=10,seed=2026,checkpoint_sha256=digest,validation=val,prompt='colon polyp',input='original1008',metric=256,reason='User requested final checkpoint test; diagnostic, does not replace validation-selected best'))
    d.torch.set_num_threads(4)
    trainer=d.t.SAM3TrainerNative(str(d.R/'seed2026.yaml'))
    model=trainer.model
    model.num_interactive_steps_val=0
    d.torch.set_autocast_cache_enabled(False)
    d.load_adapter(model,d.torch.load(dst,map_location='cpu',weights_only=True))
    result=d.evaluate(model,'test','test_text')
    d.TEXT_PROMPT=''
    empty=d.evaluate(model,'test','test_empty')
    d.TEXT_PROMPT='colon polyp'
    assert result['count']==empty['count']==100 and d.sha(src)==digest
    d.save(d.D/'results.json',dict(epoch=10,seed=2026,validation=val,test=result,test_empty=empty,checkpoint_sha256=digest))
    (d.D/'COMPLETE').write_text('complete\n')
    print(json.dumps(dict(test=result,test_empty=empty)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if d.D.exists():(d.D/'FAILED.txt').write_text(traceback.format_exc())
        raise
