"""Train-only robust calibration and a NumPy reference for TP-guided PC."""
import numpy as np

def fit_anchor_calibration(scores, train_indices, anchor_indices, eps=1e-6):
    scores=np.asarray(scores,dtype=np.float64)
    medians,mads,counts=[],[],[]
    for a,self_index in enumerate(anchor_indices):
        ids=[int(i) for i in train_indices if int(i)!=int(self_index)]
        x=scores[a,ids]
        if not np.isfinite(x).all():raise ValueError('Non-finite train scores')
        median=float(np.median(x));mad=float(np.median(np.abs(x-median)))
        medians.append(median);mads.append(mad);counts.append(len(ids))
    return {'median':medians,'mad':mads,'epsilon':eps,'counts':counts}

def standardize(scores,cal):
    s=np.asarray(scores,dtype=np.float64)
    return (s-np.asarray(cal['median'])[:,None])/(np.asarray(cal['mad'])[:,None]+cal['epsilon'])

def fit_transition_calibration(sim,train_indices,eps=1e-6):
    t=np.asarray(sim)[np.ix_(train_indices,train_indices)]
    x=t[np.triu_indices(len(train_indices),1)].astype(np.float64)
    median=float(np.median(x));mad=float(np.median(np.abs(x-median)))
    return {'median':median,'mad':mad,'epsilon':eps,'train_pair_count':len(x)}

def guided_reference(target_tokens,anchor_tokens,prototype,tau=10.,top_r=3):
    """All inputs are normalized tokens; target shape P,D; anchor shape M,D."""
    x=np.asarray(target_tokens,dtype=np.float64)
    a=np.asarray(anchor_tokens,dtype=np.float64)
    s=x@np.asarray(prototype,dtype=np.float64)
    weights=np.exp(tau*(s-s.max()));weights/=weights.sum()
    local=np.sort(a@x.T,axis=0)[-min(top_r,len(a)):].mean(axis=0)
    return float(weights@local),weights,local

def self_test():
    # Identical supported tokens yield one, including anchors with fewer than r tokens.
    value,w,c=guided_reference(np.array([[1.,0.],[1.,0.]]),np.array([[1.,0.]]),np.array([1.,0.]))
    assert abs(value-1)<1e-12 and np.allclose(w,[.5,.5])
    # Changing attention towards the matching target region improves correspondence.
    x=np.eye(2);a=np.array([[1.,0.]])
    aligned,_,_=guided_reference(x,a,np.array([1.,0.]))
    opposed,_,_=guided_reference(x,a,np.array([0.,1.]))
    assert aligned>.99 and opposed<.01
    # Held-out values never affect calibration, and self-match is excluded.
    s=np.array([[999.,1.,2.,3.,4.,5.],[1.,999.,2.,3.,4.,5.]])
    cal=fit_anchor_calibration(s,[0,1,2,3],[0,1])
    other=s.copy();other[:,4:]=1e9
    assert cal==fit_anchor_calibration(other,[0,1,2,3],[0,1])
    assert cal['median']==[2.,2.] and cal['counts']==[3,3]
    assert np.isfinite(standardize(np.ones((1,4)),fit_anchor_calibration(np.ones((1,4)),[0,1,2],[0]))).all()
    sim=np.eye(4);tc=fit_transition_calibration(sim,[0,1,2]);assert tc['train_pair_count']==3

if __name__=='__main__':
    self_test();print('joint score tests passed')
