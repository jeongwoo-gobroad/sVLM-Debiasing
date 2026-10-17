# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studies'))
from plain_error_rules import censor_tail, classify
from analyze_attention import paired_change, holm


class AttentionStudyTests(unittest.TestCase):
    def test_eos_oov_is_not_censored(self):
        row = dict(raw_text='사과, 자동차', ended_with_eos=True, hit_token_limit=False, new_tokens=9)
        text, tail, _ = censor_tail(row, {'사과'})
        self.assertEqual(tail, '')
        self.assertTrue(classify(text, {'사과'})['out_of_candidate'])

    def test_cap_keeps_completed_prefix_and_does_not_complete_tail(self):
        row = dict(raw_text='사과, 자동차, 주방', ended_with_eos=False, hit_token_limit=True, new_tokens=32)
        text, tail, reason = censor_tail(row, {'사과', '주방용품'})
        self.assertEqual(tail, ' 주방')
        self.assertEqual(reason, 'cap_candidate_prefix')
        self.assertEqual(classify(text, {'사과', '주방용품'})['outside_tags'], ['자동차'])

    def test_exact_candidate_at_cap_is_kept(self):
        row = dict(raw_text='사과', ended_with_eos=False, hit_token_limit=True, new_tokens=32)
        self.assertEqual(censor_tail(row, {'사과'})[:2], ('사과', ''))

    def test_unknown_stop_is_rejected(self):
        row = dict(raw_text='주방', ended_with_eos=False, hit_token_limit=None, new_tokens=None)
        with self.assertRaises(ValueError): censor_tail(row, {'주방용품'})

    def test_paired_test_and_holm(self):
        result = paired_change([True]*8+[False]*2, [False]*10)
        self.assertEqual(result['reduced'], 8)
        self.assertEqual(result['increased'], 0)
        self.assertAlmostEqual(result['p_exact_mcnemar_two_sided'], 2/256)
        self.assertEqual(holm([.01, .04]), [.02, .04])


if __name__ == '__main__': unittest.main()
