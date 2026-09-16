"""Finish and archive this single experiment after its existing inference process."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

P=Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7')
ROOT=P/'work/kvasir_rethink_20260908'
R=ROOT/'anchor_factorial'

def status(state, **kw):
    (R/'pipeline_status.json').write_text(json.dumps(dict(state=state,updated_at=datetime.datetime.now().astimezone().isoformat(),**kw),indent=2)+'\n')

def main():
    pid=int((R/'inference.pid').read_text())
    try:
        while not (R/'INFERENCE_COMPLETE.json').exists():
            process=Path('/proc')/str(pid)/'status'
            if not process.exists() or '\nState:\tZ' in process.read_text():
                raise RuntimeError('Inference process exited without completion marker; see inference.log')
            q=R/'unique_quality/propagation_quality.jsonl'
            count=sum(1 for line in q.open() if line.strip()) if q.exists() else 0
            status('inference',pid=pid,completed_unique_routes=count,total_unique_routes=1669)
            time.sleep(15)
        status('analysis')
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (R/'analysis.log').open('a') as log:
            subprocess.run(['/home/violet/anaconda3/envs/mkunet_mamba/bin/python','-u',str(ROOT/'analyze_anchor_factorial.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (R/'COMPLETE').exists()
        shutil.copy2(R/'report.md',P/'reproduction_reports/Kvasir_anchor_retrieval_path_factorial_validation_20260908.md')
        status('complete',results=str(R/'results.json'),report=str(R/'report.md'))
    except BaseException:
        error=traceback.format_exc();(R/'PIPELINE_FAILED.txt').write_text(error);status('failed',error=error);raise

if __name__=='__main__':main()
