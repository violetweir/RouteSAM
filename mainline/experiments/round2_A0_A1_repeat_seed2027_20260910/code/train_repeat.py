"""A1/A2 retraining: original RGB to 1008, cosine per update, gradient norm clip 1."""
import json, math, random, time, traceback
import train_direct as d

def update(trainer, batch, scheduler):
    model = trainer.model
    out = d.outputs(model, d.move(batch['input']), grad=True)
    targets = [model.back_convert(x) for x in batch['input'].find_targets]
    with d.t.SAM3Output.iteration_mode(out, iter_mode=d.t.SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as it:
        for stage, target in zip(it, targets):
            for o in stage:
                o['indices'] = trainer.matcher(o, target)
                for aux in o.get('aux_outputs', []):
                    aux['indices'] = trainer.matcher(aux, target)
    loss = trainer.loss_wrapper(out, targets)[d.t.CORE_LOSS_KEY]
    assert d.torch.isfinite(loss)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert params
    norm = d.torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, error_if_nonfinite=True)
    after = d.torch.linalg.vector_norm(d.torch.stack([d.torch.linalg.vector_norm(p.grad.detach().float()) for p in params]))
    assert float(after) <= 1.0001, 'gradient clipping failed'
    lr = trainer.optimizer.param_groups[0]['lr']
    trainer.optimizer.step()
    scheduler.step()
    d.torch.clear_autocast_cache()
    return dict(loss=float(loss.detach()), learning_rate=lr, next_learning_rate=trainer.optimizer.param_groups[0]['lr'],
                grad_norm_before=float(norm), grad_norm_after=float(after), clipped=bool(norm > 1.0))

def main():
    D, R = d.D, d.R
    arm = d.args.arm
    assert arm in ['A0','A1']
    expected_count = {'A0':436,'A1':604}[arm]
    total_steps = expected_count*10
    D.mkdir(parents=True, exist_ok=False)
    d.save(D/'status.json', dict(stage='initializing', arm=arm+'_seed2027', gpu=d.args.gpu))
    d.torch.set_num_threads(4)
    trainer = d.t.SAM3TrainerNative(str(R/f'{arm}.yaml'))
    model = trainer.model
    model.num_interactive_steps_val = 0
    d.torch.set_autocast_cache_enabled(False)
    d.torch.clear_autocast_cache()
    initial = d.adapter(model)
    initial_hash = d.tensor_hash(initial.items())
    frozen_hash = d.tensor_hash((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    old = json.loads((R/'reference_initialization.json').read_text())
    assert initial_hash != old['adapter_hash'] and frozen_hash == old['frozen_hash']
    repeat_ref=json.loads((R/'reference_seed2027_initialization.json').read_text())
    assert initial_hash==repeat_ref['adapter_hash'] and frozen_hash==repeat_ref['frozen_hash']
    d.save(D/'initialization.json', dict(adapter_hash=initial_hash, frozen_hash=frozen_hash, same_frozen_base=True, different_lora_seed=True, seed=2027))
    ds = d.t.COCOSegmentDataset(R/f'data/{arm}', 'train')
    records = json.loads((R/f'data/{arm}/records.json').read_text())
    assert len(ds)==len(records)==expected_count and sum(x['is_gt'] for x in records)==8
    scheduler = d.torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=total_steps, eta_min=5e-7)
    if d.args.smoke:
        sample = ds[0]
        raw = d.PIL.open(records[0]['image_path']).convert('RGB').resize((1008,1008), d.PIL.Resampling.BILINEAR)
        expected = ds.transform(raw)
        assert d.torch.equal(sample.images[0].data, expected), 'training RGB differs from original to1008'
        model.train()
        info = update(trainer, d.collate([sample]), scheduler)
        assert info['next_learning_rate'] < info['learning_rate']
        assert d.tensor_hash(d.adapter(model).items()) != initial_hash
        assert d.tensor_hash((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen_hash
        d.checkpoint(model,D/'updated.pt')
        saved=d.tensor_hash(d.adapter(model).items())
        d.load_adapter(model,initial)
        d.load_adapter(model,d.torch.load(D/'updated.pt',map_location='cpu',weights_only=True))
        assert d.tensor_hash(d.adapter(model).items())==saved
        val=d.evaluate(model,'validation','smoke_validation',limit=1)
        d.save(D/'smoke.json', dict(passed=True, original_rgb_tensor_verified=True, frozen_unchanged=True,
                                  adapter_roundtrip=True, actual_update=info, one_image_validation=val))
        d.save(D/'status.json',dict(stage='smoke_complete'))
        print(json.dumps(info),flush=True)
        return
    assert json.loads((R/f'smoke/{arm}/smoke.json').read_text())['passed']
    d.checkpoint(model,D/'epoch0.pt')
    best=d.evaluate(model,'validation','validation_epoch0');best_epoch=0
    d.checkpoint(model,D/'best.pt');d.save(D/'best.json',dict(epoch=0,**best))
    step=0;start=time.time()
    for epoch in range(1,11):
        order=d.np.random.default_rng(2027+epoch).permutation(len(ds)).tolist()
        loader=d.DataLoader(ds,batch_size=1,sampler=order,num_workers=2,collate_fn=d.collate,pin_memory=True)
        model.train();updates=[]
        for j,batch in enumerate(loader):
            info=update(trainer,batch,scheduler);step+=1;updates.append(info)
            expected_lr=5e-7+(5e-5-5e-7)*(1+math.cos(math.pi*(step-1)/total_steps))/2
            assert math.isclose(info['learning_rate'],expected_lr,rel_tol=1e-9)
            d.append(D/'updates.jsonl',dict(epoch=epoch,step=step,**info))
            if j==0 or step%20==0:
                status=dict(stage='training',arm=arm+'_seed2027',epoch=epoch,epochs=10,epoch_step=j+1,epoch_size=expected_count,
                            step=step,elapsed_seconds=time.time()-start,best_validation_dice=best['dice'],**info)
                d.save(D/'status.json',status);print(json.dumps(status),flush=True)
        assert len(updates)==expected_count and len(set(order))==expected_count
        d.checkpoint(model,D/f'epoch{epoch}.pt')
        state=dict(epoch=epoch,step=step,adapter=d.adapter(model),optimizer=trainer.optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   torch_rng=d.torch.get_rng_state(),cuda_rng=d.torch.cuda.get_rng_state_all(),numpy_rng=d.np.random.get_state(),python_rng=random.getstate())
        d.torch.save(state,D/'latest_training.tmp');(D/'latest_training.tmp').replace(D/'latest_training.pt')
        d.save(D/'status.json',dict(stage='validation',epoch=epoch,step=step))
        val=d.evaluate(model,'validation',f'validation_epoch{epoch}')
        if val['dice']>best['dice']:
            best=val;best_epoch=epoch;d.checkpoint(model,D/'best.pt');d.save(D/'best.json',dict(epoch=epoch,**best))
        d.append(D/'epochs.jsonl',dict(epoch=epoch,step=step,validation=val,best_epoch=best_epoch,
            loss=float(d.np.mean([x['loss'] for x in updates])),lr_start=updates[0]['learning_rate'],lr_end=updates[-1]['learning_rate'],
            clipped_fraction=float(d.np.mean([x['clipped'] for x in updates])),
            grad_norm_before_mean=float(d.np.mean([x['grad_norm_before'] for x in updates])),
            grad_norm_before_max=max(x['grad_norm_before'] for x in updates)))
    assert d.tensor_hash((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen_hash
    d.save(D/'WEIGHTS_FROZEN.json',dict(best_epoch=best_epoch,validation=best,total_steps=step,best_sha256=d.sha(D/'best.pt')))
    d.load_adapter(model,d.torch.load(D/'best.pt',map_location='cpu',weights_only=True))
    d.save(D/'status.json',dict(stage='test',best_epoch=best_epoch))
    result=d.evaluate(model,'test','test_best')
    d.TEXT_PROMPT=''
    empty_result=d.evaluate(model,'test','test_best_empty')
    d.TEXT_PROMPT='colon polyp'
    d.save(D/'results.json',dict(arm=arm+'_seed2027',best_epoch=best_epoch,validation=best,test=result,test_empty=empty_result,training_pool_size=expected_count,
        primary_prompt='colon polyp',empty_text_role='predeclared diagnostic, not test-based prompt selection'))
    report='\n'.join([f'# {arm}新设置训练结果','',f'Best epoch: {best_epoch}; Val Dice: {best["dice"]:.9f}',
        f'Test colon polyp Dice: {result["dice"]:.9f}; IoU: {result["iou"]:.9f}',
        f'Test empty text Dice: {empty_result["dice"]:.9f}; IoU: {empty_result["iou"]:.9f}',
        '原图直接1008输入；256逐图宏平均指标；本次seed2027权重按有文本validation冻结后才执行test。空文本为预先约定的补充消融。',''])
    (D/'report.md').write_text(report,encoding='utf-8')
    (D/'COMPLETE').write_text('complete\n');d.save(D/'status.json',dict(stage='complete',best_epoch=best_epoch,test=result))

if __name__=='__main__':
    try:
        main()
    except BaseException:
        if d.D.exists():
            (d.D/'FAILED.txt').write_text(traceback.format_exc())
            d.save(d.D/'status.json',dict(stage='failed',error=traceback.format_exc()))
        raise
