# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Preserve complete 32-token responses for paired error analysis."""
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import torch
from tqdm import tqdm
from transformers import LogitsProcessorList

from status_io import write_status


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_root(root):
    return Path(os.environ.get('SVLM_DATA_ROOT', Path(root) / 'datasets')).expanduser().resolve()


def image_paths(root):
    paths = []
    for split, expected in [('val', 440), ('test', 294)]:
        selected = sorted((data_root(root) / 'imgs' / split).glob('*.jpg'),
                          key=lambda p: int(p.stem.split('_')[-1]))
        if len(selected) != expected:
            raise ValueError(f'{split}: expected {expected} images, found {len(selected)}')
        paths.extend(selected)
    assert len({p.name for p in paths}) == 734
    return paths


def collect(root, model_name, mode, alpha, tokenizer, eos_id, tags, build_inputs, generate, make_processors,
            output, source_file, limit=0, resume=False, split='all', returns_prompt=True):
    output = Path(output) / model_name / mode
    output.mkdir(parents=True, exist_ok=True)
    paths = image_paths(root)
    if split != 'all':
        paths = [p for p in paths if p.parent.name == split]
    if limit:
        paths = paths[:limit]
    data = data_root(root)
    sources = [Path(source_file), Path(__file__), Path(__file__).with_name('status_io.py')]
    signature = {'model': model_name, 'mode': mode, 'alpha': alpha if mode == 'dc' else None,
                 'max_new_tokens': 32, 'stop': 'original model EOS or 32-token cap; no comma stop',
                 'returned_sequence_includes_prompt': returns_prompt,
                 'prompt': 'unchanged default', 'python': os.path.realpath(os.sys.executable),
                 'sources': {p.name: digest(p) for p in sources},
                 'noise_sha256': digest(data / 'noise/reference_noise.jpg'),
                 'candidates_sha256': digest(data / 'tag_candidates/refined_tags.txt'),
                 'ground_truth_sha256': {s: digest(data / f'tag_gts/human_gt_{s}.jsonl') for s in ['val', 'test']},
                 'images': {p.name: digest(p) for p in paths}}
    metadata = output / 'metadata.json'
    records_path = output / 'records.jsonl'
    records = []
    if records_path.exists():
        if not resume:
            raise FileExistsError(f'{records_path}: use --resume or a new directory')
        if json.loads(metadata.read_text(encoding='utf-8')) != signature:
            raise ValueError('Resume configuration differs from original run')
        records = [json.loads(line) for line in records_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    done = {r['image'] for r in records}
    if len(done) != len(records) or not done <= {p.name for p in paths}:
        raise ValueError('Duplicate or unexpected image keys')
    write_status(metadata, signature, required=True)
    initial = len(records)
    started = time.time()

    def status(stage, **extra):
        elapsed = time.time() - started
        finished = len(records) - initial
        state = {'stage': stage, 'model': model_name, 'mode': mode, 'completed': len(records),
                 'total': len(paths), 'seconds_this_run': elapsed, 'new_records': finished,
                 'eta_seconds': (len(paths) - len(records)) * elapsed / finished if finished else None, **extra}
        if not write_status(output / 'status.json', state, required=stage != 'running'):
            warnings.warn('Progress file busy; raw outputs are saved and progress will retry.')

    status('running')
    try:
        with records_path.open('a', encoding='utf-8') as stream:
            remaining = [p for p in paths if p.name not in done]
            for path in tqdm(remaining, desc=f'{model_name} {mode}', initial=initial, total=len(paths)):
                inputs = build_inputs(str(path))
                # Kanana의 시각 입력 경로는 새 토큰만 반환한다.
                prefix = inputs['input_ids'].shape[1] if returns_prompt else 0
                processors = LogitsProcessorList(make_processors(inputs))
                # 콤마 종료나 첫 토큰 관측기를 끼우지 않고 기존 생성 전체를 보존한다.
                with torch.no_grad():
                    ids = generate(inputs, processors, None, 32)[0, prefix:].tolist()
                if not 1 <= len(ids) <= 32:
                    raise ValueError(f'Invalid generation length: {len(ids)}')
                raw = tokenizer.decode(ids, skip_special_tokens=True)
                row = {'image': path.name, 'split': path.parent.name, 'model': model_name, 'mode': mode,
                       'generated_ids': ids, 'raw_text': raw, 'new_tokens': len(ids),
                       'ended_with_eos': ids[-1] == eos_id,
                       'hit_token_limit': len(ids) == 32 and ids[-1] != eos_id}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                records.append(row)
                status('running', current_image=path.name)
        assert {r['image'] for r in records} == {p.name for p in paths}
        # 완료 표시는 메모리의 개수가 아니라 디스크에 남은 내용까지 확인한 뒤 쓴다.
        stored = [json.loads(line) for line in records_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if stored != records:
            raise IOError('Saved responses differ from generated records; resume after inspecting the file')
        status('complete')
    except Exception as exc:
        status('failed', error=str(exc))
        raise
