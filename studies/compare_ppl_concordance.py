# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Direct plain/c/dc changes in all-pair concordance with fixed noise PPL."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm

from analysis_common import MODELS, MODES, COMPARISONS, holm, sha, write_csv
from ppl_inputs import add_plain_errors, load_errors, read_capture
from rank_concordance import paired_concordance_change, pair_counts, reference_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['captures', 'input', 'data', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--previous', type=Path, help='Optional earlier matrix output to cross-check')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--permutations', type=int, default=99999)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output folder to preserve previous results')
    candidates_path = args.data/'tag_candidates/refined_tags.txt'
    candidates = candidates_path.read_text(encoding='utf-8').splitlines()
    assert len(candidates) == len(set(candidates)) == 124
    noise = args.data/'noise/reference_noise.jpg'
    captures = {m: read_capture(args.captures/m, candidates, noise) for m in MODELS}
    assert all(c['model'] == m and c['candidates_sha256'] == sha(candidates_path) for m, c in captures.items())
    matrices, outside, names, _ = load_errors(args.input, args.data, candidates)
    add_plain_errors(args.input, args.data, candidates, matrices, outside, names)
    if args.previous:
        previous_report = json.loads((args.previous/'report.json').read_text(encoding='utf-8'))
        assert all(sha(p) == h for p, h in previous_report['sources'].items())
        with np.load(args.previous/'paired_error_indicators.npz', allow_pickle=False) as previous:
            np.testing.assert_array_equal(previous['candidates'], candidates)
            np.testing.assert_array_equal(previous['image_names'], names)
            for (model, mode), matrix in matrices.items():
                np.testing.assert_array_equal(matrix, previous[model+'__'+mode])
    summaries, changes, per_tag, per_pair, nulls = [], [], [], [], {}
    references = {}
    for model in tqdm(MODELS, desc='PPL order concordance', ascii=True):
        reference = np.array([r['mean_nll'] for r in captures[model]['rows']])
        references[model] = reference
        pairs = reference_pairs(reference)
        m = len(pairs[0])
        for mode in MODES:
            counts = matrices[model, mode].sum(0).astype(np.int64)
            concordant, discordant, tied, zero_zero = map(int, pair_counts(counts, pairs)[0])
            summaries.append({'model': model, 'mode': mode,
                'concordance': (concordant+.5*tied)/m,
                'concordance_percent': 100*(concordant+.5*tied)/m,
                'concordant_pairs': concordant, 'discordant_pairs': discordant,
                'frequency_tied_pairs': tied, 'zero_zero_pairs': zero_zero,
                'comparable_pairs': m, 'excluded_reference_tied_pairs': 124*123//2-m,
                'candidate_wrong_occurrences': int(counts.sum()),
                'outside_candidate_occurrences': outside[model, mode],
                'zero_frequency_candidates': int((counts == 0).sum()), 'paired_images': len(names)})
            for i, row in enumerate(captures[model]['rows']):
                per_tag.append({'model': model, 'mode': mode, 'tag': row['tag'],
                    'mean_nll': row['mean_nll'], 'perplexity': row['perplexity'], 'ppl_rank': row['ppl_rank'],
                    'wrong_image_count': int(counts[i]), 'wrong_rank_average': float(rankdata(-counts)[i])})
            for i, j in zip(*pairs):
                per_pair.append({'model': model, 'mode': mode, 'preferred_tag': candidates[i],
                    'other_tag': candidates[j], 'preferred_nll': float(reference[i]), 'other_nll': float(reference[j]),
                    'preferred_wrong_count': int(counts[i]), 'other_wrong_count': int(counts[j]),
                    'concordance_score': 1. if counts[i] > counts[j] else .5 if counts[i] == counts[j] else 0.,
                    'both_zero': bool(counts[i] == counts[j] == 0)})
        for before, after in COMPARISONS:
            result, samples = paired_concordance_change(reference, matrices[model,before], matrices[model,after],
                                                       args.permutations, args.seed, progress=True)
            changes.append({'model': model, 'before': before, 'after': after, **result})
            nulls[model+'__'+before+'__'+after] = samples
    for row, value in zip(changes, holm([r['p_permutation'] for r in changes])):
        row['p_holm_six'] = float(value)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('condition_summary', summaries), ('paired_concordance_changes', changes),
                       ('per_tag', per_tag), ('per_pair', per_pair)]:
        write_csv(args.output/(name+'.csv'), rows)
    np.savez_compressed(args.output/'permutation_draws.npz', **nulls)
    np.savez_compressed(args.output/'paired_error_indicators.npz', image_names=np.array(names),
        candidates=np.array(candidates), **{m+'__'+c: a for (m,c),a in matrices.items()},
        **{m+'__mean_nll': r for m,r in references.items()})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    labels = ['Vanilla', 'Trie', 'Trie+correction']
    for ax, model in zip(axes, MODELS):
        values = [next(r['concordance_percent'] for r in summaries if r['model'] == model and r['mode'] == c) for c in MODES]
        bars = ax.bar(labels, values, color=['#88739b','#397ba0','#419572'], width=.6)
        ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=4, fontsize=10)
        ax.set_title('Qwen 3.5 4B' if model == MODELS[0] else 'Kanana 3B')
        ax.axhline(50, color='gray', linewidth=.8, linestyle=':')
        ax.set_ylim(0, 100)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('All-pair order concordance (%)')
    fig.text(.5, .01, 'Frequency ties receive half credit; dotted line = 50%, not an accuracy threshold.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1])
    for extension in ['png','pdf']:
        fig.savefig(args.output/('concordance_summary.'+extension), dpi=220)
    plt.close(fig)
    sources = [args.input/'report.json', args.input/'per_image.jsonl', candidates_path,
        args.data/'tag_gts/human_gt_test.jsonl', noise]
    if args.previous:
        sources += [args.previous/'report.json', args.previous/'paired_error_indicators.npz']
    for model in MODELS:
        sources.extend(args.captures/model/p for p in ['capture.json','tags.jsonl','status.json'])
    if args.protocol:
        sources.append(args.protocol)
    code = [Path(__file__).with_name(p) for p in ['compare_ppl_concordance.py','rank_concordance.py',
        'analysis_common.py', 'ppl_inputs.py']]
    report = {'analysis': 'Direct changes in all-pair concordance with fixed noise PPL preference',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'new_inference': False,
        'paired_images': 294, 'candidates': 124, 'reused_candidate_scores': 248, 'reused_response_records': 1764,
        'metric': '(concordant pairs + 0.5 * frequency-tied pairs) / reference-comparable pairs',
        'reference': 'Lower saved mean NLL is preferred; exact reference ties excluded; all other unordered pairs',
        'delta': 'C_after-C_before; negative means less aligned; percentage points =100*delta',
        'frequency': 'Images containing a unique complete candidate absent from GT; upstream incomplete-tail exclusion retained',
        'test': 'Two-sided paired whole-image condition swaps; recompute pooled counts; integer score comparison; Monte Carlo plus-one',
        'multiplicity': 'Holm across two models times three condition comparisons; six tests',
        'permutations': args.permutations, 'seed': args.seed,
        'confidence_intervals': 'Not calculated; permutation null quantiles are not effect confidence intervals',
        'conditions': summaries, 'changes': changes,
        'validation': {'all_six_matrices_equal_previous': True if args.previous else None,
                       'capture_and_original_sources_verified': True},
        'limitations': ['Exploratory follow-up after prior analyses; Holm6 does not cover research-history selection',
            'Conditional on a single fixed noise reference and prompt; not causal evidence or accuracy',
            'The null requires within-image exchangeability of complete error vectors, not merely equal concordance',
            'Images assumed independent; unknown source clustering not modeled',
            'Complete outside-candidate errors excluded and counted separately',
            'Ties, total error counts, and different GT prevalence affect concordance; all-zero errors score0.5',
            'No direct between-model effect comparison performed'],
        'sources': {str(p.resolve()): sha(p) for p in sources},
        'code_sha256': {str(p.resolve()): sha(p) for p in code},
        'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    report['artifact_sha256'] = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved six paired concordance comparisons to {args.output}')


if __name__ == '__main__':
    main()
