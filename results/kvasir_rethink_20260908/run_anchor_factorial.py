"""Propagate the unique routes for the three fixed-budget validation controls."""
import os
import time
from pathlib import Path
from run_anchor_matrix import P,R as MATRIX,OLD,MODES,CKPT,read,save,jsonl,module,sha,resolve

R=MATRIX.parent/'anchor_factorial'

def main():
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_video_model
    os.chdir(P)
    assert (MATRIX/'INFERENCE_COMPLETE.json').exists() and (R/'GENERATION_COMPLETE.json').exists()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    ev=module('factorial_eval',MATRIX/'code/eval_route_propagation_quality.py')
    allroutes={}
    for variant in ['B','C','D']:
        for r in read(R/variant/'routes.jsonl'):allroutes.setdefault(r['route_id'],r)
    routes=sorted(allroutes.values(),key=lambda r:(r['target_id'],r['bridge_count'],r['route_id']))
    sources={}
    for mode in MODES:
        for r in read(OLD/mode/'propagation_quality_validation/propagation_quality.jsonl'):
            if r['route_id'] in allroutes:
                r['forward_mask_path']=str(resolve(r['forward_mask_path']))
                if r['route_id'] in sources:assert sources[r['route_id']]['forward_mask_sha256']==r['forward_mask_sha256']
                sources[r['route_id']]=r
    for r in read(MATRIX/'quality/propagation_quality.jsonl'):
        if r['route_id'] in allroutes:sources.setdefault(r['route_id'],r)
    out=R/'unique_quality';out.mkdir(exist_ok=True)
    resultpath=out/'propagation_quality.jsonl'
    for r in read(resultpath):sources[r['route_id']]=r
    seeded=[];cycle_pending=[]
    for r in routes:
        old=sources.get(r['route_id'])
        if old is None:continue
        for k in ['anchor_id','anchor_mask_sha256','anchor_box_xywh_normalized','bridge_ids','target_id']:assert r[k]==old[k]
        assert sha(resolve(old['forward_mask_path']))==old['forward_mask_sha256']
        row={**old,**r}
        (seeded if row.get('q_cycle') is not None else cycle_pending).append(row)
    jsonl(resultpath,seeded)
    save(R/'inference_plan.json',dict(unique_routes=len(routes),reused_complete=len(seeded),reused_forward_need_cycle=len(cycle_pending),new_forward=len(routes)-len(seeded)-len(cycle_pending)))
    print((R/'inference_plan.json').read_text(),flush=True)
    model=build_sam3_video_model(checkpoint_path=str(CKPT),load_from_HF=False,device='cuda',compile=False)
    model.eval();assert model.image_size==1008
    # Long-route replay checks both forward prediction and cycle inference at the memory cap.
    reference=next(r for r in read(OLD/MODES[0]/'propagation_quality_validation/propagation_quality.jsonl') if r['bridge_count']==6)
    replay=ev.evaluate(model,[reference],R/'long_route_replay',256,False,True,no_cycle=False)[0]
    oldmask=np.asarray(Image.open(resolve(reference['forward_mask_path'])).convert('L'))>127
    newmask=np.asarray(Image.open(replay['forward_mask_path']).convert('L'))>127
    mismatch=int(np.count_nonzero(oldmask!=newmask))
    audit=dict(route_id=reference['route_id'],differing_pixels=mismatch,old_cycle=reference['q_cycle'],new_cycle=replay['q_cycle'])
    save(R/'long_route_replay_audit.json',audit)
    assert mismatch==0 and abs(reference['q_cycle']-replay['q_cycle'])<1e-12,audit
    for i,row in enumerate(cycle_pending,1):
        paths=[row['anchor_image_path'],*row['bridge_image_paths'],row['target_image_path']]
        mask=np.asarray(Image.open(resolve(row['forward_mask_path'])).convert('L'))>127
        started=time.time();cycle=ev.t21.propagate_return_from_predicted_mask(model,paths,mask,256)
        row.update(q_cycle=ev.t21.dice(cycle['mask'],ev.t21.load_mask(row['anchor_mask_path'],256)),
            cycle_success=cycle['success'],cycle_failure_reason=cycle['failure_reason'],cycle_candidate_count=cycle['candidate_count'],
            cycle_sam_score=cycle['sam_score'],cycle_added_seconds=time.time()-started)
        ev.t21.append_fsync(resultpath,row)
        print('CYCLE_ONLY',i,len(cycle_pending),flush=True)
    rows=ev.evaluate(model,routes,out,256,True,True,no_cycle=False)
    lookup={r['route_id']:r for r in rows}
    for variant in ['B','C','D']:
        quality=[{**lookup[r['route_id']],**r} for r in read(R/variant/'routes.jsonl')]
        assert len(quality)==700 and all(r['q_cycle'] is not None for r in quality)
        jsonl(R/variant/'propagation_quality.jsonl',quality)
    save(R/'INFERENCE_COMPLETE.json',dict(unique_routes=len(rows),per_variant=700,peak_cuda_allocated=torch.cuda.max_memory_allocated(),time=time.time()))
    print('FACTORIAL INFERENCE COMPLETE',flush=True)

if __name__=='__main__':main()
