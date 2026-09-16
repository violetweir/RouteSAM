"""Small, explicit PC score ablation; no target masks are accepted here."""
import numpy as np


def adaptive_k(foreground_fraction, patch_count):
    if not 0 <= foreground_fraction <= 1 or patch_count < 1:
        raise ValueError('Invalid foreground fraction or patch count')
    return max(1, min(patch_count, int(np.floor(foreground_fraction * patch_count + 0.5))))


def score_maps(similarities, anchor_fractions, grid=18, spatial_weight=0.05):
    """Input A x N x P normalized cosine maps; output A x N scalars.

    v1: mean of top K, K=round(anchor foreground fraction * P).
    v2: v1 - .05*(1 - K/area(bounding rectangle of selected patches)).
    Occupancy is a simple compactness proxy, not a shape/lesion guarantee.
    Ties use lower flattened patch index. No target GT or pseudo mask enters.
    """
    sims = np.asarray(similarities, dtype=np.float32)
    if sims.ndim != 3 or sims.shape[-1] != grid * grid:
        raise ValueError('Expected A x N x grid^2 scores')
    if len(anchor_fractions) != sims.shape[0]:
        raise ValueError('Anchor fraction count mismatch')
    means, compactness, ks = [], [], []
    for anchor, fraction in enumerate(anchor_fractions):
        k = adaptive_k(fraction, grid * grid)
        ks.append(k)
        order = np.argsort(-sims[anchor], axis=-1, kind='stable')[:, :k]
        top = np.take_along_axis(sims[anchor], order, axis=-1)
        appearance = top.mean(axis=-1)
        yy, xx = order // grid, order % grid
        bbox_area = (yy.max(axis=-1)-yy.min(axis=-1)+1) * (xx.max(axis=-1)-xx.min(axis=-1)+1)
        comp = k / bbox_area.astype(np.float32)
        means.append(appearance)
        compactness.append(comp)
    v1 = np.asarray(means, dtype=np.float32)
    compactness = np.asarray(compactness, dtype=np.float32)
    return {'v1': v1, 'v2': (v1-spatial_weight*(1-compactness)).astype(np.float32),
            'compactness': compactness, 'anchor_k': np.asarray(ks)}


def fast_routes(stage, records, support, patch_mean, cond_scores, split, beam_width=32):
    """Vectorize historical beam scoring while preserving its stable tie order.

    Generates all depths in a single pass because the original beam recurrence
    at depth d does not depend on its requested maximum depth. Must pass the
    complete v0 route identity audit before any new experiment is accepted.
    """
    index = {r['merged_id']: i for i,r in enumerate(records)}
    train = np.asarray([i for i,r in enumerate(records) if r['split']=='train'])
    ids = np.asarray([records[i]['merged_id'] for i in train])
    sim = (patch_mean @ patch_mean.T).astype(np.float64)
    cond = np.asarray(cond_scores, dtype=np.float64)
    anchors = stage.t21.human_pool(support, 512)
    rank_caches = []
    for ai in range(len(anchors)):
        ranks = []
        for tail in range(len(records)):
            order = np.lexsort((ids, cond[ai, train], sim[train,tail]))[::-1]
            ranks.append(train[order])
        rank_caches.append(ranks)
    result = []
    targets = sorted([r for r in records if r['split']==split], key=lambda r:r['merged_id'])
    for no,target in enumerate(targets,1):
        ti = index[target['merged_id']]
        best = [None]*7
        for ai,anchor in enumerate(anchors):
            forbidden = {ti, index[anchor['anchor_id']]}
            paths = np.empty((1,0), dtype=np.int64)
            for depth in range(7):
                if depth:
                    expanded = []
                    for path in paths:
                        used = forbidden | set(path.tolist())
                        tail = ti if len(path)==0 else int(path[0])
                        ranked = [int(i) for i in rank_caches[ai][tail] if int(i) not in used][:beam_width]
                        expanded.extend([[i,*path] for i in ranked])
                    paths = np.asarray(expanded, dtype=np.int64)
                nodes = np.concatenate([paths, np.full((len(paths),1),ti,dtype=np.int64)],axis=1)
                values = np.concatenate([cond[ai,nodes],sim[nodes[:,:-1],nodes[:,1:]]],axis=1)
                mins, means = values.min(axis=1), values.mean(axis=1)
                order = np.lexsort((-means,-mins))[:beam_width]
                paths = paths[order]
                score = (float(mins[order[0]]), float(means[order[0]]))
                candidate = (score, anchor['anchor_id'], anchor, paths[0].tolist())
                if best[depth] is None or candidate[:2] > best[depth][:2]:
                    best[depth] = candidate
        for depth,(score,_,anchor,path) in enumerate(best):
            result.append(stage.make_route(target,depth,anchor,path,score,records))
        if no==1 or no%10==0:
            print(f'routes {split} {no}/{len(targets)}',flush=True)
    return result


def self_test():
    assert adaptive_k(0,324)==1 and adaptive_k(1,324)==324
    assert adaptive_k(0.05,324)==16
    # Equal appearance, equal K: compact 2x2 patch block must beat four corners.
    x=np.zeros((1,2,16),dtype=np.float32)
    x[0,0,[0,1,4,5]]=1
    x[0,1,[0,3,12,15]]=1
    s=score_maps(x,[.25],grid=4)
    assert s['v1'][0,0]==s['v1'][0,1]==1
    assert s['compactness'][0,0]==1 and s['compactness'][0,1]==.25
    assert s['v2'][0,0]>s['v2'][0,1]
    assert np.allclose(score_maps(x,[.25],grid=4,spatial_weight=0)['v2'],s['v1'])
    assert score_maps(x,[0],grid=4)['anchor_k'][0]==1


if __name__=='__main__':
    self_test()
    print('core tests passed')
