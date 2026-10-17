# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
from itertools import combinations, product
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'studies'))
from rank_concordance import doubled_scores, pair_counts, paired_concordance_change, reference_pairs


def independent_score(reference, counts):
    result = 0
    for i, j in combinations(range(len(reference)), 2):
        if reference[i] == reference[j]:
            continue
        if reference[i] > reference[j]:
            i, j = j, i
        result += 2 if counts[i] > counts[j] else int(counts[i] == counts[j])
    return result


class ConcordanceTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([1., 2., 2., 4.])
        self.before = np.array([[1,1,0,0], [1,0,0,0], [1,0,1,0], [0,0,0,1]])
        self.after = np.array([[0,1,0,1], [1,0,0,1], [0,1,1,1], [0,0,0,1]])

    def test_ties_zero_counts_and_reference_orientation(self):
        pairs = reference_pairs(self.reference)
        self.assertEqual(len(pairs[0]), 5)
        counts = np.array([[3,2,1,0], [0,0,0,0], [0,1,2,3], [2,2,0,0]])
        expected = [independent_score(self.reference, x) for x in counts]
        np.testing.assert_array_equal(doubled_scores(counts, pairs), expected)
        np.testing.assert_array_equal(pair_counts(counts, pairs)[1], [0,0,5,5])
        self.assertEqual(expected[:3], [10,5,0])

    def test_exact_randomization_matches_independent_image_swaps(self):
        result, null = paired_concordance_change(self.reference, self.before, self.after, 16)
        expected = []
        for mask in product([False, True], repeat=4):
            a, b = self.before.copy(), self.after.copy()
            a[list(mask)], b[list(mask)] = self.after[list(mask)], self.before[list(mask)]
            expected.append(independent_score(self.reference, b.sum(0))
                            -independent_score(self.reference, a.sum(0)))
        np.testing.assert_array_equal(np.sort(null), sorted(expected))
        obs = independent_score(self.reference, self.after.sum(0))-independent_score(self.reference, self.before.sum(0))
        self.assertEqual(result['observed_delta_score_units'], obs)
        self.assertEqual(result['p_permutation'], np.mean(np.abs(expected) >= abs(obs)))
        self.assertTrue(result['exact'])

    def test_monte_carlo_matches_masks_and_plus_one(self):
        result, null = paired_concordance_change(self.reference, self.before, self.after, 11)
        expected = []
        for mask in np.random.default_rng(42).random((11,4)) < .5:
            a, b = self.before.copy(), self.after.copy()
            a[mask], b[mask] = self.after[mask], self.before[mask]
            expected.append(independent_score(self.reference, b.sum(0))-independent_score(self.reference, a.sum(0)))
        np.testing.assert_array_equal(null, expected)
        self.assertEqual(result['p_permutation'], (1+sum(abs(null) >= abs(result['observed_delta_score_units'])))/12)
        self.assertFalse(result['exact'])

    def test_identical_conditions_have_zero_change_and_p_one(self):
        result, null = paired_concordance_change(self.reference, self.before, self.before, 16)
        self.assertEqual(result['delta_concordance'], 0)
        self.assertEqual(result['p_permutation'], 1)
        np.testing.assert_array_equal(null, 0)

    def test_condition_reversal_and_batch_size_invariance(self):
        a, x = paired_concordance_change(self.reference, self.before, self.after, 11, batch_size=3)
        b, y = paired_concordance_change(self.reference, self.after, self.before, 11, batch_size=8)
        self.assertEqual(a['delta_concordance'], -b['delta_concordance'])
        self.assertEqual(a['p_permutation'], b['p_permutation'])
        np.testing.assert_array_equal(x, -y)

    def test_candidate_reordering_does_not_change_result(self):
        order = [2,0,3,1]
        a, x = paired_concordance_change(self.reference, self.before, self.after, 16)
        b, y = paired_concordance_change(self.reference[order], self.before[:,order], self.after[:,order], 16)
        self.assertEqual(a, b)
        np.testing.assert_array_equal(x, y)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            reference_pairs([1,1,1])
        with self.assertRaises(ValueError):
            paired_concordance_change(self.reference, self.before*2, self.after)
        with self.assertRaises(ValueError):
            paired_concordance_change(self.reference, self.before[:2], self.after)
        with self.assertRaises(ValueError):
            paired_concordance_change([1,2], self.before, self.after)


if __name__ == '__main__':
    unittest.main()
