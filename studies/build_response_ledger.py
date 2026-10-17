# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Build paired complete-tag error records from verified 32-token responses."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from tqdm import tqdm

from analysis_common import MODELS, MODES, read_rows, sha
from plain_error_rules import censor_tail


def constrained_tags(row, candidates):
    raw = row['raw_text']
    if row.get('ended_with_eos') is not None:
        evaluated, excluded, reason = censor_tail(row, candidates)
    else:
        if row.get('provenance', {}).get('kind') != 'validated_normalized_log':
            raise ValueError('Unknown stop flags require validated log provenance')
        segments = re.split(r'[,\n;]', raw)
        tail = segments[-1]
        clean = tail.strip()
        excluded = ''
        reason = 'legacy_complete_tail_stop_unknown'
        if clean and clean not in candidates:
            # 후보 접두부인지 기록하되, 누락된 EOS/토큰 수는 추정하지 않는다.
            reason = 'legacy_candidate_prefix' if any(t.startswith(clean) for t in candidates) else (
                'legacy_incomplete_character' if '\ufffd' in clean else 'legacy_uncertain_tail')
            excluded = tail
        evaluated = raw[:-len(excluded)] if excluded else raw
    parts = [s.strip() for s in re.split(r'[,\n;]', evaluated) if s.strip()]
    if any(s not in candidates for s in parts):
        raise ValueError('Noncandidate constrained item before the excluded tail; review required')
    return set(parts), excluded, reason, evaluated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'plain_reviews', 'data', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output folder to preserve prior records')
    candidates_path = args.data/'tag_candidates/refined_tags.txt'
    gt_path = args.data/'tag_gts/human_gt_test.jsonl'
    candidates = candidates_path.read_text(encoding='utf-8').splitlines()
    candidate_set = set(candidates)
    assert len(candidates) == len(candidate_set) == 124
    gt = {}
    for item in read_rows(gt_path):
        assert not gt.keys() & item.keys()
        gt.update({name: set(tags) for name, tags in item.items()})
    expected = {f'test_{i}.jpg' for i in range(1, 295)}
    assert set(gt) == expected and all(tags <= candidate_set for tags in gt.values())
    reviewed_rows = [r for r in read_rows(args.plain_reviews) if r['split'] == 'test']
    reviews = {r['key']: r for r in reviewed_rows}
    assert len(reviews) == len(reviewed_rows) == 588
    assert set(reviews) == {f'{m}/plain/{n}' for m in MODELS for n in expected}
    sources = {str(p.resolve()): sha(p) for p in [candidates_path, gt_path, args.plain_reviews]}
    records = []
    for model in MODELS:
        for mode in MODES:
            source = args.input/model/mode/'records.jsonl'
            if not source.exists():
                source = args.input/'reuse'/model/mode/'records.jsonl'
            sources[str(source.resolve())] = sha(source)
            saved = [r for r in read_rows(source) if r['split'] == 'test']
            assert len(saved) == 294 and {r['image'] for r in saved} == expected
            for row in tqdm(saved, desc=f'{model} {mode} complete tags', ascii=True):
                assert row['model'] == model and row['mode'] == mode
                key = f"{model}/{mode}/{row['image']}"
                raw_hash = hashlib.sha256(row['raw_text'].encode()).hexdigest()
                if mode == 'plain':
                    reviewed = reviews[key]
                    assert reviewed['raw_sha256'] == raw_hash and not reviewed['needs_review']
                    evaluated, excluded, reason = censor_tail(row, candidate_set)
                    assert evaluated == reviewed['evaluated_text'] and excluded == reviewed['excluded_tail']
                    tags = set(reviewed['generated_tags'])
                else:
                    tags, excluded, reason, evaluated = constrained_tags(row, candidate_set)
                truth = gt[row['image']]
                wrong = tags-truth
                records.append({'key': key, 'model': model, 'mode': mode, 'image': row['image'],
                    'raw_sha256': raw_hash, 'evaluated_text': evaluated,
                    'excluded_tail': excluded, 'exclusion_reason': reason,
                    'stop_metadata_available': row.get('ended_with_eos') is not None,
                    'tags': sorted(tags), 'ground_truth': sorted(truth),
                    'wrong_tags': sorted(wrong), 'wrong_candidate_tags': sorted(wrong & candidate_set),
                    'generated_n': len(tags), 'wrong_n': len(wrong),
                    'wrong_candidate_n': len(wrong & candidate_set)})
    assert len(records) == len({r['key'] for r in records}) == 1764
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = args.output/'per_image.jsonl'
    ledger.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    sources[str(ledger.resolve())] = sha(ledger)
    report = {'records': len(records), 'paired_images': 294, 'sources': sources,
        'code_sha256': {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('plain_error_rules.py')]},
        'policy': 'Completed unique tags per image; reviewed plain outputs; only incomplete final fragment excluded.'}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()


