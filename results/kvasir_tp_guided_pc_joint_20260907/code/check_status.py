"""Read-only experiment progress snapshot."""
import collections
import datetime
import json
from pathlib import Path

r=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_tp_guided_pc_joint_20260907')
out={'time':datetime.datetime.now().astimezone().isoformat(),'complete':(r/'COMPLETE').exists()}
if (r/'FAILED').exists():out['failed']=(r/'FAILED').read_text()
for name in ['feature_replay_audit','ridge_implementation_audit']:
    p=r/(name+'.json')
    if p.exists():out[name]=json.loads(p.read_text())
out['variants']={}
for v in ['tp_baseline','joint_v1','joint_v2']:
    d={'complete':(r/v/'COMPLETE').exists()}
    if (r/v/'FAILED.json').exists():d['failed']=json.loads((r/v/'FAILED.json').read_text())
    for split in ['validation','test']:
        p=r/v/v/f'propagation_quality_{split}/propagation_quality.jsonl'
        if p.exists():
            rows=[]
            for line in p.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
            d[split]={'quality_rows':len(rows),'status':dict(collections.Counter(x.get('status') for x in rows))}
        p=r/v/f'{split}_summary.json'
        if p.exists():d[split+'_summary']=json.loads(p.read_text())
        p=r/v/f'reuse_audit_{split}.json'
        if p.exists():d[split+'_reuse']=json.loads(p.read_text())
    out['variants'][v]=d
out['pipeline_tail']=(r/'pipeline.log').read_text().splitlines()[-6:]
print(json.dumps(out,ensure_ascii=False,indent=2))
