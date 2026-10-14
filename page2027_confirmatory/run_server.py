"""Run the staged pipeline, then one isolated worker per explicitly selected GPU."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='1', help='Physical GPU IDs, e.g. 1 or 1,0; default leaves GPU0 alone')
    p.add_argument('--out', default='runs/confirmatory')
    p.add_argument('--pilot', type=int, default=0, help='Run N questions/dataset/policy and estimate runtime')
    args = p.parse_args()
    gpus = [s.strip() for s in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or any(not s.isdigit() for s in gpus) or args.pilot < 0:
        p.error('Use distinct numeric GPU IDs and a nonnegative pilot size')
    out = Path(args.out).resolve()
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, '-u', str(ROOT / 'scripts/step6_confirmatory_review.py'), '--out', str(out)]
    def environment(gpu):
        return {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'TOKENIZERS_PARALLELISM': 'false',
                'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
    env = environment(gpus[0])
    for stage in ['prepare', 'fit', 'calibrate', 'smoke']:
        print(f'=== {stage} (GPU {gpus[0]} for GPU stages) ===', flush=True)
        subprocess.run(base + ['--stage', stage], env=env, cwd=ROOT, check=True)
    children, handles = [], []
    try:
        for shard, gpu in enumerate(gpus):
            handle = (logs / f'shard-{shard}.log').open('a', encoding='utf-8')
            handles.append(handle)
            cmd = base + ['--stage', 'run', '--shard', str(shard), '--num-shards', str(len(gpus))]
            if args.pilot:
                cmd += ['--limit', str(args.pilot)]
            children.append(subprocess.Popen(cmd, env=environment(gpu), cwd=ROOT, stdout=handle,
                                              stderr=subprocess.STDOUT))
            print(f'Shard {shard} -> GPU {gpu}; log {handle.name}', flush=True)
        while any(c.poll() is None for c in children):
            for i, child in enumerate(children):
                if child.poll() not in [None, 0]:
                    raise RuntimeError(f'Shard {i} failed; inspect {logs / ("shard-"+str(i)+".log")}')
            time.sleep(2)
        if any(c.returncode for c in children):
            raise RuntimeError('A shard failed. Inspect logs and rerun the same command to resume.')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()
    if args.pilot:
        subprocess.run(base + ['--stage', 'estimate'], env=env, cwd=ROOT, check=True)
    else:
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/step7_review_stats.py'),
                        '--out', str(out)], env={**env, 'CUDA_VISIBLE_DEVICES': ''},
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
