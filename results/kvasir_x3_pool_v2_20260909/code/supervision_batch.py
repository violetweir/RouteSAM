"""Align every per-image field before a loss that slices GT/pseudo streams."""
import torch

def align_supervision_batch(batch, expected_labeled):
    labels=batch['is_labeled'].to(dtype=torch.bool,device='cpu')
    if labels.ndim!=1 or int(labels.sum())!=expected_labeled:
        raise ValueError('Unexpected labeled membership in training batch')
    order=torch.cat((torch.where(labels)[0],torch.where(~labels)[0]))
    indices=order.tolist();n=len(indices);out={}
    for key,value in batch.items():
        if isinstance(value,torch.Tensor) and value.ndim and len(value)==n:
            out[key]=value.index_select(0,order.to(value.device))
        elif isinstance(value,(list,tuple)) and len(value)==n:
            out[key]=[value[i] for i in indices]
        else:
            raise ValueError(f'Unrecognized per-image field: {key}')
    assert bool(out['is_labeled'][:expected_labeled].all())
    assert not bool(out['is_labeled'][expected_labeled:].any())
    return out
