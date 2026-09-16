"""Validation-only structural controls for TP-main / PC-local auxiliary masks."""
from pathlib import Path
import json,collections,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
R=P/'work/kvasir_tp_main_pc_aux_20260908'
O=P/'work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6'
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(name,x):(R/name).write_text(json.dumps(x,indent=2)+'\n')
def path(p):
    p=Path(p);return p if p.is_absolute() else P/p
def mask(p):
    a=np.asarray(Image.open(path(p)).convert('L'))>127;assert a.shape==(256,256);return a
def dice(a,b):
    n=a.sum()+b.sum();return float(2*(a&b).sum()/n) if n else 1.
def metrics(d,base):
    delta=np.array(d)-np.array(base);rng=np.random.default_rng(2026)
    ci=np.quantile(delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1),[.025,.975])
    return {'dice':float(np.mean(d)),'delta_vs_TP':float(delta.mean()),'paired_ci95':[float(x) for x in ci],
      'win_tie_loss':[int((delta>1e-9).sum()),int((abs(delta)<=1e-9).sum()),int((delta< -1e-9).sum())]}
R.mkdir(parents=True,exist_ok=True)
save('structure_protocol.json',{'stage':'CPU structural probe; no new correspondence features or SAM3 inference',
 'split':'original validation only; no test inputs',
 'TP_main':'existing five-fold target-level OOF independent TP Router outputs; preserve anchor/path/ranking',
 'PC_aux':'existing seven PC masks; no student',
 'band':'dilation(TP,8 iterations,3x3) minus erosion(TP,8 iterations,3x3); TP unchanged outside band',
 'fixed_controls':['TP','unrestricted_PC_5of7','local_PC_5of7','local_PC_7of7'],
 'vote_rule':'add where >=k PC masks foreground; remove where >=k PC masks background; otherwise retain TP',
 'oracle_diagnostics':['fixed_TP_plus_raw_PC7','fixed_TP_plus_local_PC7','full_TP7_PC7','full_TP7_PC7_plus_local_PC7'],
 'GT_policy':'construct every output before loading GT; oracle and edit-quality statistics diagnostic only',
 'acceptance':'Exploratory structural check only; no test promotion from this script',
 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
tp={r['target_id']:r for r in read(P/'work/kvasir_tp_guided_pc_joint_20260907/tp_baseline/validation_oof.jsonl')}
groups={}
for mode in ['target_pooling','patch_correspondence']:
    g=collections.defaultdict(list)
    for r in read(O/f'sam3enc_anchor_conditioned_{mode}/propagation_quality_validation/propagation_quality.jsonl'):g[r['target_id']].append(r)
    groups[mode]=g
assert len(tp)==100
rows=[]
for target,b in sorted(tp.items()):
    base=mask(b['source_mask_path']);pr=sorted(groups['patch_correspondence'][target],key=lambda r:r['bridge_count']);tr=groups['target_pooling'][target]
    pc=[mask(r['forward_mask_path']) for r in pr];tm=[mask(r['forward_mask_path']) for r in tr]
    assert len(pc)==len(tm)==7
    dil=binary_dilation(base,structure=np.ones((3,3)),iterations=8)
    ero=binary_erosion(base,structure=np.ones((3,3)),iterations=8,border_value=0)
    band=dil & ~ero;votes=np.sum(pc,axis=0)
    candidates=[np.where(band,m,base) for m in pc]
    outputs={'TP':base}
    for name,k,local in [('unrestricted_PC_5of7',5,False),('local_PC_5of7',5,True),('local_PC_7of7',7,True)]:
        out=base.copy();allowed=band if local else np.ones_like(base)
        out[allowed & (votes>=k)]=True;out[allowed & (votes<=7-k)]=False
        if local:assert np.array_equal(out[~band],base[~band])
        outputs[name]=out
        dest=R/name/'validation_masks';dest.mkdir(parents=True,exist_ok=True)
        Image.fromarray(out.astype('uint8')*255).save(dest/(target.replace('::','__')+'.png'))
    # No labels used above. Read GT solely to evaluate the already-constructed masks.
    gt=np.asarray(Image.open(pr[0]['target_mask_path_evaluation_only']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
    bd=dice(base,gt);assert abs(bd-b['dice'])<1e-12
    ds={n:dice(m,gt) for n,m in outputs.items()};raw=[dice(m,gt) for m in pc];local=[dice(m,gt) for m in candidates];td=[dice(m,gt) for m in tm]
    row={'target_id':target,'tp_route_id':b['route_id'],'band_fraction':float(band.mean()),'dice':ds,
      'oracle_diagnostic_only':{'fixed_TP_plus_raw_PC7':max([bd]+raw),'fixed_TP_plus_local_PC7':max([bd]+local),
        'full_TP7_PC7':max(td+raw),'full_TP7_PC7_plus_local_PC7':max(td+raw+local)},
      'edits':{n:{'pixels_changed':int((m!=base).sum()),'wrong_to_right':int(((m!=base)&(m==gt)).sum()),'right_to_wrong':int(((m!=base)&(m!=gt)).sum())} for n,m in outputs.items() if n!='TP'}}
    rows.append(row)
base=[r['dice']['TP'] for r in rows]
result={'count':100,'methods':{n:metrics([r['dice'][n] for r in rows],base) for n in rows[0]['dice']},
 'oracle_diagnostic_only':{n:float(np.mean([r['oracle_diagnostic_only'][n] for r in rows])) for n in rows[0]['oracle_diagnostic_only']},
 'mean_band_fraction':float(np.mean([r['band_fraction'] for r in rows])),
 'edits':{n:{k:sum(r['edits'][n][k] for r in rows) for k in ['pixels_changed','wrong_to_right','right_to_wrong']} for n in rows[0]['edits']},
 'test_evaluated':False,'new_SAM3_correspondence_implemented':False}
assert abs(result['methods']['TP']['dice']-.8517552851672318)<1e-12
save('structure_results.json',result)
(R/'structure_per_target.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print(json.dumps(result,indent=2),flush=True)
