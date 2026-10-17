# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Evaluate saved predictions with the original experiment metrics."""
import argparse
import json
from pathlib import Path
import os

from metrics import load_answer_jsonl, load_candidate_txt, evaluate_models

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SVLM_DATA_ROOT", ROOT / "datasets")).expanduser().resolve()


def evaluate(gt, baseline, corrected):
    answers = load_answer_jsonl(gt)
    for path in [baseline, corrected]:
        predictions = load_candidate_txt(path)
        if predictions.keys() != answers.keys():
            raise ValueError(f'{path}: expected exactly {len(answers)} images, got {len(predictions)}')
    result = evaluate_models(gt, baseline, corrected, 0.05)
    # NumPy scalars need conversion before writing JSON.
    return json.loads(json.dumps(result, default=lambda x: x.item()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=Path, default=DATA_ROOT / 'tag_gts/human_gt_test.jsonl')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--corrected', type=Path)
    parser.add_argument('--reference', action='store_true', help='Evaluate all supplied reference comparisons')
    parser.add_argument('--reference_root', type=Path, default=Path(os.environ.get('SVLM_REFERENCE_ROOT', ROOT / 'reference_results')))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/evaluation.json')
    args = parser.parse_args()
    results = {}
    if args.reference:
        base = args.reference_root / 'test'
        for name in ['qwen3.5_2b', 'qwen3.5_4b', 'qwen3_2b', 'qwen3_4b', 'kanana']:
            results[name] = evaluate(args.gt, base / name / f'{name}_test_plain.txt', base / name / f'{name}_test_dc.txt')
        for name in ['qwen3.5_2b', 'qwen3.5_4b', 'qwen3_2b', 'qwen3_4b', 'kanana']:
            results[name + '_trie'] = evaluate(args.gt, base / name / f'{name}_test_plain.txt', base / name / f'{name}_test_c.txt')
        for variant in ['a', 'b', 'c']:
            prefix = base / 'qwen3.5_4b' / f'qwen3.5_4b_test_prompt[{variant}]'
            results['prompt_' + variant] = evaluate(args.gt, Path(str(prefix) + '_plain.txt'), Path(str(prefix) + '_dc.txt'))
        vanilla = base / 'qwen3.5_4b/qwen3.5_4b_test_plain.txt'
        for name, path in [('kelip', base / 'kelip/kelip_test.txt'), ('vcd', base / 'qwen3.5_4b/qwen4b_vcd.txt'), ('vcd_trie', base / 'qwen3.5_4b/qwen4b_vcd_trie.txt')]:
            results[name] = evaluate(args.gt, vanilla, path)
        control = base / 'qwen3.5_4b/qwen4b_vcd_trie.txt'
        for name, baseline in [('trie', base / 'qwen3.5_4b/qwen3.5_4b_test_c.txt'), ('vcd', base / 'qwen3.5_4b/qwen4b_vcd.txt')]:
            results['vcd_trie_vs_' + name] = evaluate(args.gt, baseline, control)
    else:
        if not args.baseline or not args.corrected:
            parser.error('Supply --reference or both --baseline and --corrected')
        results['comparison'] = evaluate(args.gt, args.baseline, args.corrected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, result in results.items():
        jd, f1 = result['jaccard_distance'], result['f1_samples']
        print(f"{name}: dJ {jd['mean_a']:.4f} -> {jd['mean_b']:.4f}; F1 {f1['mean_a']:.4f} -> {f1['mean_b']:.4f}")


if __name__ == '__main__':
    main()
