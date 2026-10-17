# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Validate fixed-noise PPL captures and paired candidate-error matrices."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from analysis_common import MODELS, MODES, read_rows, sha


def read_capture(folder, candidates, noise):
    path = folder/'capture.json'
    capture = json.loads(path.read_text(encoding='utf-8'))
    status = json.loads((folder/'status.json').read_text(encoding='utf-8'))
    assert status['stage'] == 'complete' and status['completed'] == 124
    assert capture['candidates'] == candidates and capture['noise_sha256'] == sha(noise)
    assert capture['image_condition'] == 'fixed shared noise'
    assert not capture['trie_mask'] and not capture['debiasing'] and not capture['use_cache']
    assert capture['all_model_layers_used'] and capture['temperature'] == 1.
    snapshots = capture.get('source_snapshot_paths', {})
    assert all(sha(snapshots.get(p, p)) == h for p, h in capture['source_sha256'].items())
    assert sha(folder/'tags.jsonl') == capture['tags_jsonl_sha256']
    saved = list(read_rows(folder/'tags.jsonl'))
    assert len(saved) == len(capture['rows']) == 124
    for i, (row, raw) in enumerate(zip(capture['rows'], saved)):
        assert row['tag'] == candidates[i] and row['candidate_position'] == i+1
        assert {k: v for k, v in row.items() if k != 'ppl_rank'} == raw
        values = row['token_log_probs']
        assert len(values) == len(row['token_ids']) == row['token_count'] > 0
        assert all(math.isfinite(v) and v <= 0 for v in values)
        mean = -math.fsum(values)/len(values)
        assert math.isclose(row['sum_nll'], -math.fsum(values), abs_tol=1e-12)
        assert math.isclose(row['mean_nll'], mean, abs_tol=1e-12)
        assert math.isclose(row['perplexity'], math.exp(mean), rel_tol=1e-12)
    np.testing.assert_array_equal(rankdata([r['mean_nll'] for r in saved]),
                                  [r['ppl_rank'] for r in capture['rows']])
    audits = capture['validation']['causal_prefix_audits']
    assert len(audits) >= 2 and all(a['passed'] for a in audits)
    for audit in audits:
        source = next(r for r in saved if r['tag'] == audit['tag'])
        assert source['token_ids'] == audit['token_ids']
        assert source['token_log_probs'] == audit['teacher_forced_log_probs']
        error = max(abs(a-b) for a, b in zip(source['token_log_probs'], audit['without_target_log_probs']))
        assert math.isclose(error, audit['max_abs_logprob_difference'], abs_tol=1e-12)
        assert error <= audit['absolute_tolerance_nats'] == .15
    assert capture['validation']['round_trip_all_tags']
    if capture['model'] == 'kanana':
        assert capture['validation']['kanana_wrapper_audit']['passed']
    return capture


def load_errors(folder, data, candidates):
    prior = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    assert all(sha(p) == h for p, h in prior['sources'].items())
    original_ledgers = [p for p in prior['sources'] if Path(p).name == 'per_image.jsonl']
    assert len(original_ledgers) == 1 and sha(original_ledgers[0]) == sha(folder/'per_image.jsonl')
    gt = {}
    for row in read_rows(data/'tag_gts/human_gt_test.jsonl'):
        assert not gt.keys() & row.keys()
        gt.update({k: set(v) for k, v in row.items()})
    names = [f'test_{i}.jpg' for i in range(1, 295)]
    assert set(gt) == set(names)
    records = list(read_rows(folder/'per_image.jsonl'))
    keyed = {r['key']: r for r in records}
    assert len(records) == len(keyed) == 1764
    assert set(keyed) == {f'{m}/{c}/{i}' for m in MODELS for c in MODES for i in names}
    matrices, outside = {}, {}
    for model in MODELS:
        for mode in ['c', 'dc']:
            matrix = np.zeros((294, 124), dtype=np.uint8)
            excluded = 0
            for i, name in enumerate(names):
                row = keyed[f'{model}/{mode}/{name}']
                assert row['model'] == model and row['mode'] == mode and row['image'] == name
                assert set(row['ground_truth']) == gt[name]
                wrong = set(row['tags'])-gt[name]
                assert wrong == set(row['wrong_tags'])
                assert wrong & set(candidates) == set(row['wrong_candidate_tags'])
                matrix[i] = [tag in wrong for tag in candidates]
                excluded += len(wrong-set(candidates))
            matrices[model, mode], outside[model, mode] = matrix, excluded
    return matrices, outside, names, prior


def add_plain_errors(folder, data, candidates, matrices, outside, names):
    gt = {}
    for item in read_rows(data/'tag_gts/human_gt_test.jsonl'):
        assert not gt.keys() & item.keys()
        gt.update({k: set(v) for k, v in item.items()})
    assert set(gt) == set(names)
    ledger = {r['key']: r for r in read_rows(folder/'per_image.jsonl')}
    for model in MODELS:
        matrix = np.zeros((len(names), len(candidates)), dtype=np.uint8)
        excluded = 0
        for i, name in enumerate(names):
            row = ledger[f'{model}/plain/{name}']
            assert row['model'] == model and row['mode'] == 'plain' and row['image'] == name
            assert set(row['ground_truth']) == gt[name]
            wrong = set(row['tags'])-gt[name]
            assert wrong == set(row['wrong_tags'])
            assert wrong & set(candidates) == set(row['wrong_candidate_tags'])
            matrix[i] = [tag in wrong for tag in candidates]
            excluded += len(wrong-set(candidates))
        matrices[model, 'plain'], outside[model, 'plain'] = matrix, excluded
