import copy
import numpy as np
import run_candidate_rerank as m
g=m.g

def run():
    group=[]
    for k in range(7):
        x=[0.]*15;x[0]=k*.03
        group.append({'target_id':'a','tp_route_id':'tp','pc_route_id':f'pc{k}','tp_mask_path':'tp.png','pc_mask_path':f'pc{k}.png','pc_bridge_count':k,'pc_original_rank':k+1,'features':x,'tp_dice':.8,'pc_dice':.1*k,'gain':.1*k-.8,'pc_top1_dice':0.})
    model={'means':[0.]*15,'stds':[1.]*15,'weights':[0.,1.]+[0.]*14}
    pick=m.decisions(model,.05,group)[0]
    assert pick['selected_route_id']=='pc6' and pick['pc_original_rank']==7
    changed=copy.deepcopy(group)
    for row in changed:row.update(tp_dice=1.,pc_dice=0.,gain=-1.)
    assert m.decisions(model,.05,changed)[0]['selected_route_id']=='pc6'
    assert m.decisions(model,.18,group)[0]['selected_route_id']=='tp'
    assert m.decisions(None,None,group)[0]['selected_mask_path']=='tp.png'
    # Repeating identical candidates within each target leaves loss/regularization unchanged.
    rng=np.random.default_rng(3);x=rng.normal(size=(20,15));y=rng.normal(size=20)
    one=[{'target_id':str(i),'features':x[i].tolist(),'gain':float(y[i])} for i in range(20)]
    seven=[dict(row) for row in one for _ in range(7)]
    a=m.fit_grouped(one,10.);b=m.fit_grouped(seven,10.);reference=g.linear_fit(x,y,10.)
    assert max(abs(g.predict(a,row)-g.predict(b,row)) for row in x)<1e-10
    assert max(abs(g.predict(a,row)-g.predict(reference,row)) for row in x)<1e-10
    print('candidate rerank tests passed: all-candidate selection, GT independence, exact TP fallback, target-weighted ridge')

if __name__=='__main__':run()
