from pathlib import Path
import sys,json,hashlib,numpy as np
from PIL import Image
p=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7');r=p/'work/kvasir_tp_filterfirst_students_20260909'
sys.path.insert(0,str(r/'code'))
from probability_io import load_probability
import run_t24_student as training
checks={}
for name,manifest in [('S2',r/'pseudo_manifest_original.jsonl'),('S3',r/'S3_consensus/pseudo_consensus.jsonl')]:
    sys.argv=['preflight','--data-path',str(p/'work/kvasir_1pct_anchors/baseline_data'),'--labeled-list',str(r/'protocol/frozen_labeled_images.txt'),'--pseudo-manifest',str(manifest),'--output-dir',str(r/'preflight'/name),'--experiment',name,'--defer-test']
    args=training.parse_args();ds=training.T22StudentDataset(args,'train',None)
    assert ds.labeled_count==8 and len(ds.rows)==588
    assert all(row['target']!=row['gt'] for row in ds.rows if not row['is_labeled'])
    for row in ds.rows:
        assert row['target'].is_file()
        if row.get('pixel_weight_path'):assert row['pixel_weight_path'].is_file()
    sample=ds[ds.labeled_count]
    assert sample['soft_target'].min()>=0 and sample['soft_target'].max()<=1
    entry={'labeled':ds.labeled_count,'pseudo':len(ds.rows)-ds.labeled_count,'no_unlabeled_gt_targets':True,'sample_soft_unique':sample['soft_target'].unique().tolist()[:16],'sample_pixel_weight_unique':sample['pixel_weight'].unique().tolist()[:16]}
    if name=='S3':
        assert len(entry['sample_soft_unique'])>2 and sample['pixel_weight'].min()<1
        assert np.array_equal(load_probability(ds.rows[8]['target']),sample['soft_target'].numpy())
    checks[name]=entry
f=r/'preflight';f.mkdir(exist_ok=True)
a=np.array([[0,9362,32768,65535]],np.uint16);Image.fromarray(a).save(f/'uint16_fixture.png')
assert np.allclose(load_probability(f/'uint16_fixture.png'),a.astype(np.float32)/65535)
b=np.array([[0,128,255]],np.uint8);Image.fromarray(b).save(f/'uint8_fixture.png')
assert np.array_equal(load_probability(f/'uint8_fixture.png'),b.astype(np.float32)/255)
checks['decoder_tests']={'uint16_values_preserved':True,'uint8_behavior_preserved':True}
(r/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
c=json.loads((r/'config.json').read_text());c['code_sha256']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (r/'code').glob('*.py')};c['pre_training_correction']='Fixed uint16 consensus/weight PNG decoding before any training. Historical T24 convert(L) saturated S3 soft labels and weights. S2 uint8 unchanged. Shared decoder also used for X3 and committee.';(r/'config.json').write_text(json.dumps(c,indent=2)+'\n')
print(json.dumps(checks,indent=2))
