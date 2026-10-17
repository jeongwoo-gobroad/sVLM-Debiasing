# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Evaluate plain responses individually, retaining completed items in capped output."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from plain_error_rules import classify, censor_tail
from status_io import write_status


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path(os.environ.get('SVLM_DATA_ROOT', 'datasets')))
    parser.add_argument('--overrides', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = set((args.data / 'tag_candidates/refined_tags.txt').read_text(encoding='utf-8').splitlines())
    overrides = {r['key']: r for r in map(json.loads, args.overrides.read_text(encoding='utf-8').splitlines())} if args.overrides else {}
    groups, keys, sources, pending = {}, {}, {}, []
    with (args.output / 'pair_reviews.jsonl').open('w', encoding='utf-8') as ledger:
        for model in ['qwen3.5_4b', 'kanana']:
            source = args.input / model / 'plain/records.jsonl'
            sources[str(source)] = sha(source)
            with source.open(encoding='utf-8') as stream:
                for line in tqdm(stream, desc=f'{model} plain errors'):
                    row = json.loads(line)
                    assert row['model'] == model and row['mode'] == 'plain'
                    key = f"{model}/plain/{row['image']}"
                    group = (model, row['split'])
                    seen = keys.setdefault(group, set())
                    if row['image'] in seen: raise ValueError('Duplicate image')
                    seen.add(row['image'])
                    image = args.data / 'imgs' / row['split'] / row['image']
                    with Image.open(image) as picture: picture.verify()
                    evaluated, censored, reason = censor_tail(row, candidates)
                    result = classify(evaluated, candidates)
                    raw_hash = hashlib.sha256(row['raw_text'].encode()).hexdigest()
                    method = 'streamed_rules'
                    if key in overrides:
                        prior = overrides[key]
                        if prior['raw_sha256'] != raw_hash: raise ValueError('Stale manual review')
                        if 'evaluated_sha256' in prior:
                            if prior['evaluated_sha256'] != hashlib.sha256(evaluated.encode()).hexdigest():
                                raise ValueError('Manual review predates the tail exclusion policy')
                            tags = set(prior['decision']['generated_tags'])
                            result.update(prompt_noncompliance=prior['decision']['prompt_noncompliance'],
                                generated_tags=sorted(tags), outside_tags=sorted(tags-candidates),
                                out_of_candidate=bool(tags-candidates), needs_review=False)
                            method = 'individual_review_of_evaluated_text'
                        elif not prior['decision']['prompt_noncompliance']:
                            # 검토된 명사 목록의 접미부만 제거한 경우 그 판정은 여전히 유효하다.
                            result['prompt_noncompliance'] = False
                            result['needs_review'] = False
                            method = 'prior_individual_noun_list_review_retained_prefix'
                    record = {**row, 'key': key, 'image_path': str(image.resolve()),
                              'raw_sha256': raw_hash, 'image_sha256': sha(image),
                              'evaluated_text': evaluated, 'excluded_tail': censored,
                              'exclusion_reason': reason, 'review_method': method, **result}
                    ledger.write(json.dumps(record, ensure_ascii=False) + '\n')
                    if result['needs_review']: pending.append(record)
                    counts = groups.setdefault(group, Counter())
                    counts['n'] += 1
                    for flag in ['prompt_noncompliance', 'out_of_candidate', 'empty_evaluated_response']:
                        counts[flag] += int(result[flag])
                    counts['cap_responses'] += int(row['hit_token_limit'])
                    counts['excluded_tails'] += bool(censored)
                    counts['ambiguous_capped_tails'] += reason == 'cap_undelimited_tail_uncertain'
                    counts['overlap'] += result['prompt_noncompliance'] and result['out_of_candidate']
    expected = {p.name for p in (args.data / 'imgs/test').glob('*.jpg')}
    assert len(expected) == 294
    assert all(keys[(m, 'test')] == expected for m in ['qwen3.5_4b', 'kanana'])
    (args.output / 'review_queue.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in pending), encoding='utf-8')
    rows = [{'model': m, 'split': s, **c} for (m, s), c in groups.items()]
    with (args.output / 'counts.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_status(args.output / 'manifest.json', {'counts': rows, 'pending_reviews': len(pending),
        'paired_test294_complete': True, 'sources': sources,
        'code': {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('plain_error_rules.py')]},
        'policy': 'Plain only. Independent overlapping response flags. Keep completed prefix and exact candidate tail. Censor undelimited noncandidate tail only on known token-cap termination; uncertain completion retained as separate exclusion count.'})
    if pending: raise ValueError(f'{len(pending)} individual reviews required')


if __name__ == '__main__': main()
