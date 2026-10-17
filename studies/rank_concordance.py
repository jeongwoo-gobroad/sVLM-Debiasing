# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""All-pair order concordance with paired whole-image randomization."""
import numpy as np
from tqdm import trange


def reference_pairs(mean_nll):
    values = np.asarray(mean_nll, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite one-dimensional reference with at least two candidates')
    left, right = np.triu_indices(len(values), 1)
    usable = values[left] != values[right]
    left, right = left[usable], right[usable]
    if not len(left):
        raise ValueError('All reference scores are tied: no comparable candidate pairs')
    preferred = values[left] < values[right]
    return np.where(preferred, left, right), np.where(preferred, right, left)


def pair_counts(counts, pairs):
    values = np.atleast_2d(np.asarray(counts))
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected finite nonnegative candidate counts')
    if not np.equal(values, np.floor(values)).all():
        raise ValueError('Candidate counts must be integers')
    preferred, other = pairs
    a, b = values[:, preferred], values[:, other]
    return np.column_stack(((a > b).sum(1), (a < b).sum(1), (a == b).sum(1),
                            ((a == 0) & (b == 0)).sum(1)))


def doubled_scores(counts, pairs):
    values = pair_counts(counts, pairs)
    return 2*values[:, 0]+values[:, 2]


def paired_concordance_change(mean_nll, before, after, permutations=99999, seed=42,
                              batch_size=128, progress=False):
    before, after = np.asarray(before), np.asarray(after)
    if before.ndim != 2 or before.shape != after.shape or len(before) < 2:
        raise ValueError('Expected paired image-by-candidate matrices')
    if not np.isin(before, [0, 1]).all() or not np.isin(after, [0, 1]).all():
        raise ValueError('Each candidate must contribute at most once per image')
    if before.shape[1] != len(mean_nll):
        raise ValueError('Reference and response candidates must align')
    if permutations < 1 or batch_size < 1:
        raise ValueError('Permutation budget and batch size must be positive')
    pairs = reference_pairs(mean_nll)
    n, m = len(before), len(pairs[0])
    before, after = before.astype(float), after.astype(float)
    a, b = before.sum(0), after.sum(0)
    score_before, score_after = doubled_scores([a, b], pairs)
    observed = int(score_after-score_before)
    difference = after-before
    exact = n <= 20 and 2**n <= permutations
    total = 2**n if exact else permutations
    rng = np.random.default_rng(seed)
    null = np.empty(total, dtype=np.int64)
    for start in trange(0, total, batch_size, desc='Concordance swaps', ascii=True,
                        disable=not progress, leave=False):
        size = min(batch_size, total-start)
        if exact:
            mask = ((np.arange(start, start+size, dtype=np.uint64)[:, None]
                     >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
        else:
            mask = (rng.random((size, n)) < .5).astype(float)
        # 한 이미지의 오답 집합을 통째로 바꿔 태그 간 의존성을 보존한다.
        shift = mask @ difference
        perm_before, perm_after = (a+shift).astype(np.int64), (b-shift).astype(np.int64)
        null[start:start+size] = doubled_scores(perm_after, pairs)-doubled_scores(perm_before, pairs)
    extreme = int(np.count_nonzero(np.abs(null) >= abs(observed)))
    addition = 0 if exact else 1
    return {
        'concordance_before': float(score_before/(2*m)),
        'concordance_after': float(score_after/(2*m)),
        'delta_concordance': observed/(2*m), 'delta_percentage_points': 100*observed/(2*m),
        'observed_delta_score_units': observed, 'score_denominator': 2*m,
        'p_permutation': (extreme+addition)/(total+addition),
        'extreme_permutations': extreme, 'permutations_used': total, 'exact': exact,
        'paired_images': n, 'comparable_pairs': m, 'seed': seed,
        'nonidentical_image_pairs': int(np.any(difference != 0, axis=1).sum()),
    }, null
