# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Run the reference conditions sequentially in explicitly selected environments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ('qwen3.5_2b', 'qwen_n.py', 'qwen', '2b', .55, 'qw'),
    ('qwen3.5_4b', 'qwen_n.py', 'qwen', '4b', .09, 'qw'),
    ('qwen3_2b', 'qwen_n_v3.py', 'qwen3vl', '2b', .35, 'q3'),
    ('qwen3_4b', 'qwen_n_v3.py', 'qwen3vl', '4b', .31, 'q3'),
    ('kanana', 'kanana_n.py', 'kanana', None, .71, 'ka'),
]


def main():
    parser = argparse.ArgumentParser()
    for name in ['qwen', 'qwen3vl', 'kanana']:
        parser.add_argument('--python_' + name, type=Path, required=True)
    parser.add_argument('--phase', choices=['test', 'search', 'final'], default='test')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproduction')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    jobs = []
    for name, script, env, size, alpha, search in MODELS:
        python = str(getattr(args, 'python_' + env).resolve())
        extra = ['--model_size', size] if size else []
        if args.phase == 'search':
            target = output / 'search/val'
            jobs.append((name, [python, str(ROOT / f'seomun_train/search_thres_{search}.py'), *extra, '--output_dir', str(target)], None))
            continue
        if args.phase == 'final':
            best = json.loads((output / f'search/val/{name}/abs2_thres_best_result_None.json').read_text(encoding='utf-8'))
            alpha = best['best_n']
        for mode in (['plain', 'c', 'dc'] if args.phase == 'test' else ['dc']):
            target = output / ('test' if args.phase == 'test' else 'final/test')
            flags = [] if mode == 'plain' else ['--constrained']
            if mode == 'dc': flags += ['--debiasing', '--ratio_thres', str(alpha)]
            cmd = [python, str(ROOT / 'seomun_test' / script), *extra, *flags, '--output_dir', str(target)]
            jobs.append((name + '_' + mode, cmd, target / name / f'{name}_test_{mode}.txt'))
    if args.phase in ['test', 'final']:
        alpha = .09 if args.phase == 'test' else json.loads((output / 'search/val/qwen3.5_4b/abs2_thres_best_result_None.json').read_text())['best_n']
        for variant in ['a', 'b', 'c']:
            for mode in (['plain','dc'] if args.phase == 'test' else ['dc']):
                target = output / ('test' if args.phase == 'test' else 'final/test')
                cmd = [str(args.python_qwen.resolve()), str(ROOT / 'seomun_test/qwen_n.py'), '--model_size', '4b', '--prompt_variant', variant, '--output_dir', str(target)]
                if mode == 'dc':cmd += ['--constrained','--debiasing','--ratio_thres',str(alpha)]
                jobs.append((f'prompt_{variant}_{mode}',cmd,target/'qwen3.5_4b'/f'qwen3.5_4b_test_prompt[{variant}]_{mode}.txt'))
        if args.phase == 'test':
            target = output / 'test'
            jobs.append(('vcd', [str(args.python_qwen.resolve()),str(ROOT/'seomun_test/qwen_vcd.py'),'--output_dir',str(target),'--audit_visual_inputs'],target/'qwen3.5_4b/qwen4b_vcd.txt'))
            jobs.append(('vcd_trie', [str(args.python_qwen.resolve()),str(ROOT/'seomun_test/qwen_vcd.py'),'--output_dir',str(target),'--audit_visual_inputs','--constrained'],target/'qwen3.5_4b/qwen4b_vcd_trie.txt'))
    logs = output / 'logs' / args.phase
    logs.mkdir(parents=True, exist_ok=True)
    status_path = output / f'{args.phase}_status.json'
    prior = json.loads(status_path.read_text(encoding='utf-8')) if args.resume and status_path.exists() else {}
    state = {'phase': args.phase, 'jobs': prior.get('jobs', {}), 'started': datetime.now(timezone.utc).isoformat()}
    for name, cmd, predictions in jobs:
        script_hash = hashlib.sha256(Path(cmd[1]).read_bytes()).hexdigest()
        previous = state['jobs'].get(name, {})
        if args.resume and previous.get('returncode') == 0 and previous.get('script_sha256') == script_hash and previous.get('command') == cmd:
            if predictions is None or predictions.exists():continue
        state['current'] = name
        state['jobs'][name] = {'command': cmd, 'script_sha256': script_hash, 'started': datetime.now(timezone.utc).isoformat()}
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        started = time.monotonic()
        with (logs / f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        state['jobs'][name].update(returncode=result.returncode, seconds=time.monotonic()-started)
        if result.returncode == 0 and predictions is not None:
            lines = predictions.read_text(encoding='utf-8').splitlines()[2:]
            if len(lines) != 294:
                state['jobs'][name]['returncode'] = 1
                state['jobs'][name]['error'] = f'Expected 294 predictions, found {len(lines)}'
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        if state['jobs'][name]['returncode']:
            raise RuntimeError(f'{name} failed; see {logs / (name + ".log")}')
    state['current'] = None
    state['completed'] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
