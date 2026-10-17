# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Teacher-force every candidate as a first response to the fixed noise image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

import torch
from scipy.stats import rankdata
from tqdm import tqdm

import tag_generation_common
from status_io import write_status
from tag_perplexity import TagScorer, append_text, summarize_log_probs, target_log_probs

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen3.5_4b', 'kanana'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()/args.model
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory; preserve previous attempts')
    out.mkdir(parents=True, exist_ok=True)
    state = {'stage': 'loading', 'model': args.model, 'pid': os.getpid(), 'completed': 0,
             'total': 124, 'started': time.time()}
    def status(**extra):
        state.update(extra)
        write_status(out/'status.json', state)
    status()
    try:
        family = 'kanana' if args.model == 'kanana' else 'qwen'
        source = ROOT/'studies'/f'{family}_tag_generation.py'
        source_files = [Path(__file__), source, Path(tag_generation_common.__file__),
                        Path(__file__).with_name('tag_perplexity.py'), Path(__file__).with_name('status_io.py'),
                        ROOT/'models/revisions.json']
        if family == 'kanana':
            source_files += list((ROOT/'models/kanana').glob('*.py'))
        source_hashes = {str(p.resolve()): sha(p) for p in source_files}
        previous_argv, previous_collect = sys.argv, tag_generation_common.collect
        sys.argv = [str(source), '--output_dir', str(out/'loader')]
        if family == 'qwen':
            sys.argv += ['--model_size', '4b']
        tag_generation_common.collect = lambda *a, **kw: None
        try:
            adapter = runpy.run_path(str(source))
        finally:
            sys.argv, tag_generation_common.collect = previous_argv, previous_collect
        assert not adapter['constrained_decoding_mode'] and not adapter['debiasing_mode']
        model, tok, candidates = adapter['model'], adapter['tokenizer'], adapter['tags']
        assert len(candidates) == len(set(candidates)) == 124
        token_ids = [tok.encode(tag, add_special_tokens=False) for tag in candidates]
        for tag, ids in zip(candidates, token_ids):
            assert ids and not set(ids)&set(tok.all_special_ids)
            assert tok.decode(ids, skip_special_tokens=False) == tag
        noise = adapter['DATA_ROOT']/'noise/reference_noise.jpg'
        inputs = adapter['build_inputs'](str(noise))
        assert inputs['input_ids'].shape[0] == 1
        assert inputs['attention_mask'].eq(1).all()
        audits, rows = [], []
        audit_indices = sorted({0, 123, min(range(124), key=lambda i: len(token_ids[i])),
                                max(range(124), key=lambda i: len(token_ids[i]))})
        status(stage='running')
        with torch.inference_mode():
            scorer = TagScorer(model, inputs, family)
            # Kanana의 시각 임베딩 재사용이 원래 wrapper와 같은지 별도로 확인한다.
            wrapper_audit = None
            if family == 'kanana':
                original = append_text(inputs, token_ids[0])
                wrapper_logits = model(**original).logits[:, -(len(token_ids[0])+1):]
                expected = target_log_probs(wrapper_logits, token_ids[0])
                del wrapper_logits
                reused = scorer.score(token_ids[0])
                diff = max(abs(a-b) for a, b in zip(expected, reused))
                wrapper_audit = {'max_abs_logprob_difference': diff, 'passed': diff <= .15,
                                 'absolute_tolerance_nats': .15}
                assert wrapper_audit['passed'], wrapper_audit
            for i, (tag, ids) in enumerate(tqdm(list(zip(candidates, token_ids)),
                    desc=f'{args.model} noise tag PPL', ascii=True)):
                log_probs = scorer.score(ids)
                row = {'candidate_position': i+1, 'tag': tag, 'token_ids': ids,
                       'token_log_probs': log_probs, **summarize_log_probs(log_probs)}
                if i in audit_indices:
                    audits.append({'tag': tag, **scorer.audit(ids, log_probs)})
                rows.append(row)
                status(completed=i+1, latest_tag=tag, elapsed_seconds=time.time()-state['started'])
        # 열린 동기화 파일에 계속 덧붙이지 않고, 완성된 원장을 검증한 뒤 교체한다.
        serialized = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode('utf-8')
        pending = out/f'.tags.{os.getpid()}.tmp'
        pending.write_bytes(serialized)
        assert pending.read_bytes() == serialized
        pending.replace(out/'tags.jsonl')
        assert [json.loads(line) for line in (out/'tags.jsonl').read_text(encoding='utf-8').splitlines()] == rows
        ranks = rankdata([r['mean_nll'] for r in rows], method='average').tolist()
        for row, rank in zip(rows, ranks):
            row['ppl_rank'] = rank
        assert source_hashes == {str(p.resolve()): sha(p) for p in source_files}
        record = {
            'model': args.model, 'image_condition': 'fixed shared noise', 'candidates': candidates,
            'scoring': 'Raw full-vocabulary log_softmax; mean target-token NLL; exp(mean NLL)',
            'target': 'Canonical first-tag body tokens only; no added space, comma, EOS or prompt loss',
            'scope': 'Tag-prefix conditional perplexity, not probability of terminating at exactly one tag',
            'trie_mask': False, 'debiasing': False, 'temperature': 1., 'use_cache': False,
            'all_model_layers_used': True, 'prefix_token_count': inputs['input_ids'].shape[1],
            'prefix_ids': inputs['input_ids'][0].tolist(),
            'prompt_sha256': hashlib.sha256((adapter['system_prompt']+adapter['user_prompt_text']).encode()).hexdigest(),
            'noise_path': str(noise), 'noise_sha256': sha(noise),
            'candidates_path': str(adapter['DATA_ROOT']/'tag_candidates/refined_tags.txt'),
            'candidates_sha256': sha(adapter['DATA_ROOT']/'tag_candidates/refined_tags.txt'),
            'model_name': adapter['MODEL_NAME'], 'model_revision': adapter.get('MODEL_REVISION'),
            'torch_version': torch.__version__, 'transformers_version': __import__('transformers').__version__,
            'dtype': str(next(model.parameters()).dtype), 'python': sys.executable,
            'source_sha256': source_hashes, 'tags_jsonl_sha256': sha(out/'tags.jsonl'),
            'validation': {'round_trip_all_tags': True, 'causal_prefix_audits': audits,
                           'kanana_wrapper_audit': wrapper_audit}, 'rows': rows,
            'elapsed_seconds': time.time()-state['started'],
        }
        write_status(out/'capture.json', record)
        status(stage='complete', finished=time.time())
    except Exception as exc:
        status(stage='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
