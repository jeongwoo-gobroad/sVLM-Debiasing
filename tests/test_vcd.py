# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Check visual conditioning and the contrastive equation without loading weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[1] / 'seomun_test/qwen_vcd.py'


def definitions(model=None):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = {'torch': torch, 'F': F, 'LogitsProcessor': object, 'model': model}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class VCDTests(unittest.TestCase):
    def test_official_noise_schedule_and_repeatability(self):
        noise = definitions()['add_diffusion_noise']
        pixels = torch.linspace(-1, 1, 24).reshape(2, 12)
        for step in [0, 500, 999]:
            # Same CPU float32 random stream as upstream randn_like.
            torch.manual_seed(42)
            epsilon = torch.randn_like(pixels)
            betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (.005 - .00001) + .00001
            alpha_bar = torch.cumprod(1 - betas, dim=0)[step]
            expected = alpha_bar.sqrt() * pixels + (1 - alpha_bar).sqrt() * epsilon
            torch.testing.assert_close(noise(pixels, step, 42), expected, rtol=0, atol=0)
            torch.testing.assert_close(noise(pixels, step, 42), noise(pixels, step, 42), rtol=0, atol=0)

    def test_visual_tensors_and_generated_prefix_are_forwarded(self):
        calls = []
        distorted = torch.tensor([2., 1., 0., -1.])

        def model(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(logits=distorted.reshape(1, 1, -1))

        pixels = torch.zeros((2, 12))
        grid = torch.tensor([[1, 2, 2]])
        inputs = {'input_ids': torch.tensor([[7, 8]]), 'attention_mask': torch.ones((1, 2), dtype=torch.long),
                  'pixel_values': pixels, 'image_grid_thw': grid, 'mm_token_type_ids': torch.tensor([[0, 1]])}
        processor = definitions(model)['VCDLogitsProcessor'](inputs, 2, 1., .1)
        scores = torch.tensor([[3., 2., 0., -3.]])
        actual = processor(torch.tensor([[7, 8, 9]]), scores)
        self.assertIs(calls[0]['pixel_values'], pixels)
        self.assertIs(calls[0]['image_grid_thw'], grid)
        self.assertFalse(calls[0]['use_cache'])
        self.assertEqual(calls[0]['input_ids'].tolist(), [[7, 8, 9]])
        self.assertEqual(calls[0]['mm_token_type_ids'].tolist(), [[0, 1, 0]])
        expected = 2 * scores - distorted
        expected[scores < scores.max() + torch.log(torch.tensor(.1))] = -torch.inf
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
