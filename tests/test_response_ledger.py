# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Check complete-tag tail handling and multiplicity correction."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies'))
from build_response_ledger import constrained_tags
from analysis_common import holm


class ResponseLedgerTests(unittest.TestCase):
    def legacy(self, text):
        return {'raw_text': text, 'provenance': {'kind': 'validated_normalized_log'}}

    def test_completed_tag_retained_and_partial_tail_excluded(self):
        tags, tail, reason, _ = constrained_tags(self.legacy('apple, apple, ban'), {'apple','banana'})
        self.assertEqual(tags, {'apple'})
        self.assertEqual(tail, ' ban')
        self.assertEqual(reason, 'legacy_candidate_prefix')
        tags, tail, _, _ = constrained_tags(self.legacy('apple, banana'), {'apple','banana'})
        self.assertEqual(tags, {'apple','banana'})
        self.assertEqual(tail, '')

    def test_interior_outside_item_requires_review(self):
        with self.assertRaises(ValueError):
            constrained_tags(self.legacy('other, apple'), {'apple'})

    def test_unknown_provenance_not_silently_censored(self):
        with self.assertRaises(ValueError):
            constrained_tags({'raw_text':'apple, ban'}, {'apple','banana'})


    def test_holm_monotonic_stepdown(self):
        self.assertEqual(holm([.5,.01]),[.5,.02])
        self.assertEqual(holm([.04,.03]),[.06,.06])


if __name__ == '__main__': unittest.main()
