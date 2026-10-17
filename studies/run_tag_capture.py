# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Run selected full-response conditions sequentially; default to plain only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from status_io import write_status

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python_qwen', type=Path, required=True)
    parser.add_argument('--python_kanana', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/tag_outputs')
    parser.add_argument('--models', nargs='+', choices=['qwen3.5_4b', 'kanana'], default=['qwen3.5_4b', 'kanana'])
    parser.add_argument('--modes', nargs='+', choices=['plain', 'c', 'dc'], default=['plain'])
    parser.add_argument('--split', choices=['all', 'val', 'test'], default='all')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'run.lock').open('a+b')
    lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        if lock.read(1) == b'':
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior_path = output / 'pipeline_status.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8')) if prior_path.exists() else {}
    state = {'stage': 'running', 'pid': os.getpid(), 'started': time.time(), 'jobs': prior.get('jobs', {})}

    def save():
        write_status(prior_path, state, required=True)

    try:
        for model, script, python, alpha, extra in [
            ('qwen3.5_4b', 'qwen_tag_generation.py', args.python_qwen, .09, ['--model_size', '4b']),
            ('kanana', 'kanana_tag_generation.py', args.python_kanana, .71, [])]:
            if model not in args.models:
                continue
            for mode in args.modes:
                name = f'{model}/{mode}'
                command = [str(python.resolve()), str(ROOT / 'studies' / script), *extra,
                           '--ratio_thres', str(alpha), '--output_dir', str(output)]
                if mode != 'plain':
                    command.append('--constrained')
                if args.split != 'all':
                    command += ['--split', args.split]
                if mode == 'dc':
                    command.append('--debiasing')
                if args.limit:
                    command += ['--limit', str(args.limit)]
                if args.resume:
                    command.append('--resume')
                state['current'] = name
                state['jobs'][name] = {'stage': 'running', 'started': time.time(), 'command': command}
                save()
                log = output / model / mode / 'inference.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('a' if args.resume else 'w', encoding='utf-8') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                state['jobs'][name].update(stage='complete' if result.returncode == 0 else 'failed',
                                           returncode=result.returncode, finished=time.time())
                save()
                if result.returncode:
                    raise RuntimeError(f'{name} failed; see {log}')
        state.update(stage='generation_complete', current=None, finished=time.time())
        save()
    except Exception as exc:
        state.update(stage='failed', error=str(exc))
        save()
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
