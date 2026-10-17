# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Capture unchanged first-token prefill attention on the common test cohort."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from tqdm import tqdm

from attention_probe import LastQueryProbe, map_regions
from status_io import write_status
import tag_generation_common

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen3.5_4b', 'kanana'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first_tags', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve() / args.model
    out.mkdir(parents=True, exist_ok=True)
    state = {'stage': 'loading', 'model': args.model, 'completed': 0, 'total': args.limit or 294,
             'started': time.time(), 'pid': os.getpid()}
    def status(**values):
        state.update(values)
        write_status(out / 'status.json', state, required=state['stage'] != 'running')
    status()
    try:
        # 기존 입력 구성과 모델 로딩을 사용하고 생성 루프만 건너뛴다.
        family = 'kanana' if args.model == 'kanana' else 'qwen'
        source = ROOT / 'studies' / f'{family}_tag_generation.py'
        prior_argv, prior_collect = sys.argv, tag_generation_common.collect
        sys.argv = [str(source), '--output_dir', str(out / 'loader')]
        if family == 'qwen': sys.argv += ['--model_size', '4b']
        tag_generation_common.collect = lambda *a, **kw: None
        try:
            original = runpy.run_path(str(source))
        finally:
            sys.argv, tag_generation_common.collect = prior_argv, prior_collect
        model, tokenizer = original['model'], original['tokenizer']
        data = original['DATA_ROOT']
        paths = sorted((data / 'imgs/test').glob('*.jpg'), key=lambda p: int(p.stem.split('_')[-1]))
        assert len(paths) == 294
        if args.limit: paths = paths[:args.limit]
        reference_path = args.first_tags / args.model / 'plain/records.jsonl'
        references = {r['image']: r for r in map(json.loads, reference_path.read_text(encoding='utf-8').splitlines())
                      if r['split'] == 'test'}
        probe = LastQueryProbe(model, family)
        sources = [source, Path(__file__), Path(__file__).with_name('attention_probe.py'),
                   Path(inspect.getfile(type(model))), Path(inspect.getfile(type(probe.modules[0][1])))]
        signature = {'model': args.model, 'cohort': [p.name for p in paths], 'leading_k': 10,
                     'sources': {str(p): sha(p) for p in sources},
                     'reference_sha256': sha(reference_path),
                     'candidates_sha256': sha(data / 'tag_candidates/refined_tags.txt'),
                     'query': 'last prompt position, before first token generation',
                     'regions': ['visual embedding slots', 'candidate block inside tags', 'all remaining prompt tokens'],
                     'attention_layers': [(n, m.layer_idx) for n, m in probe.modules],
                     'linear_layers': list(getattr(getattr(model.config, 'text_config', None), 'layer_types', [])),
                     'interpretation': 'pre-gating softmax weights; observational; logit correction does not alter prefill attention'}
        signature = json.loads(json.dumps(signature))
        meta = out / 'metadata.json'
        published_records = out / 'records.jsonl'
        journal = Path(tempfile.gettempdir()) / 'svlm_attention_journal' / hashlib.sha256(str(out).encode()).hexdigest()[:20]
        journal.mkdir(parents=True, exist_ok=True)
        records_path = journal / 'records.jsonl'
        if meta.exists() and args.resume:
            prior = json.loads(meta.read_text(encoding='utf-8'))
            if prior != signature:
                raise ValueError('Resume source/configuration differs')
        write_status(meta, signature)
        if not records_path.exists() and published_records.exists() and args.resume:
            shutil.copyfile(published_records, records_path)
        if (records_path.exists() or published_records.exists()) and not args.resume: raise FileExistsError(records_path)
        records = [json.loads(line) for line in records_path.read_text(encoding='utf-8').splitlines()] if records_path.exists() else []
        done = {r['image'] for r in records}
        assert len(done) == len(records)
        for r in records:
            assert sha(out / r['array']) == r['array_sha256']
        initial = len(records)
        started = time.time()
        with records_path.open('a', encoding='utf-8') as stream:
            for path in tqdm([p for p in paths if p.name not in done], initial=initial,
                             total=len(paths), desc=f'{args.model} attention', ascii=True):
                inputs = original['build_inputs'](str(path))
                ids = inputs['input_ids'][0].tolist()
                if inputs['attention_mask'].min().item() != 1:
                    raise ValueError('Unexpected padding; this protocol requires one unpadded image')
                visual = np.array(ids) == (-1 if family == 'kanana' else model.config.image_token_id)
                regions, tag_index, markers = map_regions(tokenizer, ids, visual, original['tags'])
                if family == 'kanana':
                    with torch.inference_mode():
                        forward_inputs = model.prepare_mm_inputs(**{k: inputs[k] for k in
                            ['input_ids', 'pixel_values', 'image_metas', 'attention_mask']})
                    forward = lambda: model.language_model(**forward_inputs, use_cache=False, return_dict=True)
                else:
                    forward = lambda: model(**inputs, use_cache=False, return_dict=True)
                with torch.inference_mode():
                    baseline = forward().logits[:, -1].clone() if len(records) == initial else None
                    probe.rows = {}
                    probe.attach()
                    try: logits = forward().logits[:, -1].clone()
                    finally: probe.detach()
                if baseline is not None:
                    if not torch.equal(baseline, logits):
                        raise ValueError(f'Probe changed logits: {(baseline-logits).abs().max().item()}')
                    probe.validation['hook_on_off_logits_equal'] = True
                    write_status(out / 'validation.json', probe.validation)
                predicted = logits.argmax(-1).item()
                if predicted != references[path.name]['first_token_id']:
                    raise ValueError(f'First-token reference mismatch on {path.name}: {predicted}')
                weights, layer_ids = probe.array()
                assert weights.shape[-1] == len(ids)
                target = out / f'{path.stem}.npz'
                with target.open('wb') as file:
                    np.savez_compressed(file, weights=weights, layer_ids=layer_ids, input_ids=np.array(ids),
                                        regions=regions, tag_index=tag_index)
                    file.flush(); os.fsync(file.fileno())
                mean = weights.mean(axis=(0, 1))
                row = {'image': path.name, 'model': args.model, 'split': 'test',
                       'image_sha256': sha(path), 'first_token_id': predicted, 'matches_saved_first_token': True,
                       'sequence_length': len(ids), 'layer_ids': layer_ids.tolist(), 'heads': weights.shape[1],
                       'region_token_counts': [int((regions == k).sum()) for k in range(3)],
                       'region_mass': [float(mean[regions == k].sum()) for k in range(3)],
                       'region_density': [float(mean[regions == k].mean()) for k in range(3)],
                       'tag_mass': [float(mean[tag_index == k].sum()) for k in range(len(original['tags']))],
                       'tag_token_counts': [int((tag_index == k).sum()) for k in range(len(original['tags']))],
                       'array': target.name, 'array_sha256': sha(target), 'markers': markers}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
                records.append(row)
                elapsed = time.time() - started
                status(stage='running', completed=len(records), total=len(paths), current_image=path.name,
                       eta_seconds=(len(paths)-len(records))*elapsed/(len(records)-initial))
        stored = [json.loads(line) for line in records_path.read_text(encoding='utf-8').splitlines()]
        if stored != records or {r['image'] for r in records} != {p.name for p in paths}:
            raise ValueError('On-disk cohort verification failed')
        staging = out / f'.records.{os.getpid()}.tmp'
        shutil.copyfile(records_path, staging)
        if sha(staging) != sha(records_path): raise IOError('Journal copy verification failed')
        staging.replace(published_records)
        if sha(published_records) != sha(records_path): raise IOError('Published journal verification failed')
        status(stage='complete', completed=len(records), total=len(paths), eta_seconds=0)
    except Exception as exc:
        status(stage='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
