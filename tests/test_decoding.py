# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Exercise the copied decoder without downloading or loading a language model."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [ROOT / folder / name for folder in ['seomun_test', 'seomun_train']
           for name in ['qwen_n.py', 'qwen_n_v3.py', 'kanana_n.py']]


def load_classes(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    namespace = {'torch': torch, 'LogitsProcessor': object, 'ratio_threshold': 0.4,
                 'tokenizer': SimpleNamespace(encode=lambda *a, **k: [4])}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class DecoderTests(unittest.TestCase):
    def test_eos_is_available_before_first_tag(self):
        for path in SCRIPTS:
            for debias in [False, True]:
                with self.subTest(script=path, debias=debias):
                    ns = load_classes(path)
                    normal, spaced = ns['TokenTrie'](), ns['TokenTrie']()
                    normal.insert([1, 2], 'tag')
                    spaced.insert([3], 'tag')
                    ref = torch.arange(6, dtype=torch.float32)
                    decoder = ns['TrieLogitsProcessor'](0, 5, {4}, normal, spaced, ref, ref, debias)
                    result = decoder(torch.empty((1, 0), dtype=torch.long), ref.unsqueeze(0))
                    self.assertTrue(torch.isfinite(result[0, [1, 5]]).all())
                    self.assertTrue(torch.isneginf(result[0, [0, 2, 3, 4]]).all())

    def test_mask_normalization_and_eos_preservation(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                ns = load_classes(path)
                normal, spaced = ns['TokenTrie'](), ns['TokenTrie']()
                normal.insert([1], 'one')
                normal.insert([2, 3], 'two')
                spaced.insert([2], 'one')
                ref = torch.tensor([1., 4., 3., 2., 0., 5.])
                decoder = ns['TrieLogitsProcessor'](0, 5, {4}, normal, spaced, ref, ref, True)
                scores = torch.tensor([[9., 2., 6., 8., 4., 5.]])
                result = decoder._debias([1, 2, 5], scores, True)
                expected = torch.tensor([0., 1., .75]) - .4 * torch.tensor([.5, 0., 1.])
                expected[-1] = .75
                torch.testing.assert_close(result[0, [1, 2, 5]], expected)
                self.assertTrue(torch.isneginf(result[0, [0, 3, 4]]).all())
                # The original implementation also corrects at a completed leaf.
                decoder(torch.empty((1, 0), dtype=torch.long), scores)
                leaf = decoder(torch.tensor([[1]]), scores)
                self.assertTrue(torch.isfinite(leaf[0, [4, 5]]).all())
                self.assertTrue(torch.isneginf(leaf[0, [0, 1, 2, 3]]).all())

    def test_prompt_choices_are_consistent(self):
        for path in SCRIPTS + [ROOT / 'seomun_test/qwen_vcd.py']:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'system_prompt' for t in n.targets))
            values = {}
            for variant in ['default', 'a', 'b', 'c']:
                ns = {'args': SimpleNamespace(prompt_variant=variant)}
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), 'exec'), ns)
                values[variant] = ns['system_prompt']
            self.assertEqual(values['a'], '')
            self.assertEqual(values['default'], '정해진 단어로만 답하십시오.\n\n')
            self.assertIn('유저의 지시에 따르십시오.', values['b'])
            self.assertTrue(values['c'].endswith(values['default']))


if __name__ == '__main__':
    unittest.main()
