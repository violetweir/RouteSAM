"""Focused tests for leakage boundaries and exact TP fallback."""
import numpy as np
import run_gain_gate as g

def run():
    rr=[{'target_id':'x','features':[0.]*len(g.FEATURES),'tp_route_id':'tp','pc_route_id':'pc','tp_mask_path':'tp.png','pc_mask_path':'pc.png','tp_dice':.9,'pc_dice':.7,'gain':-.2}]
    fallback=g.decisions(None,None,rr)[0]
    assert fallback['selected_route_id']=='tp' and fallback['delta']==0 and fallback['dice']==.9
    model={'means':[0.]*len(g.FEATURES),'stds':[1.]*len(g.FEATURES),'weights':[.1]+[0.]*len(g.FEATURES)}
    assert g.decisions(model,.1,rr)[0]['selected_route_id']=='tp' # Strict threshold.
    assert g.decisions(model,.05,rr)[0]['selected_route_id']=='pc'
    changed=[dict(rr[0],tp_dice=0.,pc_dice=1.,gain=1.)]
    assert g.decisions(model,.05,changed)[0]['selected_route_id']==g.decisions(model,.05,rr)[0]['selected_route_id']
    ids=[str(i) for i in range(100)];folds=g.split_folds(ids,5,2026)
    assert len(set(sum(folds,[])))==100 and all(len(f)==20 for f in folds)
    for held in folds:assert not set(held)&(set(ids)-set(held))
    assert not any(k.startswith('gt_') or 'evaluation_only' in k for k in g.clean({'q_cycle':.5,'gt_dice_evaluation_only':.9,'target_mask_path_evaluation_only':'secret'}))
    x=np.ones((8,3));fit=g.linear_fit(x,np.arange(8),10.)
    assert np.isfinite(g.predict(fit,x[0])) and abs(g.predict(fit,x[0])-3.5)<1e-10
    print('gain gate focused tests passed')

if __name__=='__main__':run()
