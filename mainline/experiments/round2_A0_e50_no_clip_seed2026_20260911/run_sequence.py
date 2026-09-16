from pathlib import Path
import os, subprocess, json
R=Path(__file__).resolve().parent
env=os.environ.copy();env['PYTHONPATH']='/Data_8TB/lht/sam3';env['PYTHONUNBUFFERED']='1'
cmd=['/home/violet/anaconda3/envs/sam3/bin/python',str(R/'code/train_no_clip.py'),'--arm','seed2026','--gpu','0']
for name,extra in [('smoke',['--smoke']),('train',[])]:
    with (R/(name+'_seed2026.log')).open('x') as out:
        p=subprocess.Popen(cmd+extra,cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT)
        (R/('pid_'+name+'.txt')).write_text(str(p.pid))
        rc=p.wait()
    if rc:
        (R/'LAUNCH_FAILED.json').write_text(json.dumps(dict(stage=name,returncode=rc)))
        raise SystemExit(rc)
