# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'studies'))
from tag_perplexity import append_text, summarize_log_probs, target_log_probs


class PerplexityTests(unittest.TestCase):
    def test_shift_uses_previous_position_and_ignores_final_logit(self):
        logits = torch.tensor([[[0., 2., 1.], [4., 0., 1.], [-99., 99., 0.]]])
        result = target_log_probs(logits, [1, 0])
        expected = [math.log(math.exp(2)/(1+math.exp(2)+math.e)),
                    math.log(math.exp(4)/(math.exp(4)+1+math.e))]
        for a, b in zip(result, expected):
            self.assertAlmostEqual(a, b, places=6)

    def test_length_normalization_is_on_log_likelihood_not_ppl(self):
        first = summarize_log_probs([-2., -4.])
        twice = summarize_log_probs([-2., -4., -2., -4.])
        self.assertEqual(first['perplexity'], math.exp(3))
        self.assertEqual(first['perplexity'], twice['perplexity'])
        self.assertEqual(twice['sum_nll'], 12)

    def test_prefix_and_visual_positions_are_preserved(self):
        data = {'input_ids': torch.tensor([[11, 12]]), 'attention_mask': torch.tensor([[1, 1]]),
                'mm_token_type_ids': torch.tensor([[1, 0]]), 'pixel_values': torch.tensor([3.])}
        joined = append_text(data, [7, 8])
        self.assertEqual(joined['input_ids'].tolist(), [[11, 12, 7, 8]])
        self.assertEqual(joined['mm_token_type_ids'].tolist(), [[1, 0, 0, 0]])
        self.assertEqual(data['input_ids'].tolist(), [[11, 12]])
        self.assertIs(joined['pixel_values'], data['pixel_values'])

    def test_empty_extension_for_first_token_audit(self):
        data = {'input_ids': torch.tensor([[1, 2]])}
        self.assertEqual(append_text(data, [])['input_ids'].tolist(), [[1, 2]])

    def test_invalid_scores_and_alignment_rejected(self):
        for scores in ([], [float('nan')], [.2]):
            with self.assertRaises(ValueError):
                summarize_log_probs(scores)
        with self.assertRaises(ValueError):
            target_log_probs(torch.zeros(1, 2, 4), [1, 2])


if __name__ == '__main__':
    unittest.main()
