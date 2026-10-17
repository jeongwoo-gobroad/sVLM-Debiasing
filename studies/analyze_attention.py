# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Verify paired attention arrays and describe leading-tag outcomes without causal claims."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon
from tqdm import tqdm


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def holm(values):
    order = np.argsort(values)
    result = np.empty(len(values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(values)-rank)*values[i]))
        result[i] = running
    return result.tolist()


def paired_change(before, after):
    before, after = np.asarray(before, bool), np.asarray(after, bool)
    reduced, increased = int((before & ~after).sum()), int((~before & after).sum())
    p = binomtest(reduced, reduced+increased, .5).pvalue if reduced+increased else 1.
    delta = after.astype(float) - before.astype(float)
    random = np.random.default_rng(20261017)
    boot = delta[random.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
    return {'reduced': reduced, 'increased': increased, 'p_exact_mcnemar_two_sided': p,
            'change_percentage_points': float(delta.mean()*100),
            'paired_bootstrap95_pp': (np.quantile(boot, [.025, .975])*100).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--first_tags', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Separate analysis folder; defaults to INPUT/analysis')
    parser.add_argument('--figure_height', type=float, default=.85, help='Figure canvas height in inches (at least .8)')
    args = parser.parse_args()
    if args.figure_height < .8:
        parser.error('--figure_height must be at least .8 to keep the axis labels readable')
    out = args.output or args.input / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    candidates = (args.data / 'tag_candidates/refined_tags.txt').read_text(encoding='utf-8').splitlines()
    leading = set(candidates[:10])
    gt = {}
    for item in read_jsonl(args.data / 'tag_gts/human_gt_test.jsonl'): gt.update(item)
    names = sorted(gt, key=lambda s: int(Path(s).stem.split('_')[-1]))
    assert len(names) == 294
    maps, summaries, comparison, first_rows, sources = {}, [], {}, [], {}
    for model in ['qwen3.5_4b', 'kanana']:
        folder = args.input / model
        status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
        assert status['stage'] == 'complete' and status['completed'] == 294
        validation = json.loads((folder / 'validation.json').read_text(encoding='utf-8'))
        assert validation['hook_on_off_logits_equal'] and validation['official_eager_row_max_abs_error'] <= 1e-5
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        records = read_jsonl(folder / 'records.jsonl')
        assert len(records) == 294 and {r['image'] for r in records} == set(names)
        assert all(sha(p) == h for p, h in metadata['sources'].items())
        sources[str(folder / 'records.jsonl')] = sha(folder / 'records.jsonl')
        expected_layers = [x[1] for x in metadata['attention_layers']]
        normalized, per_image, leading_boundaries = [], [], []
        for row in tqdm(records, desc=f'{model} verify attention'):
            assert row['matches_saved_first_token']
            array_file = folder / row['array']
            assert sha(array_file) == row['array_sha256']
            array = np.load(array_file)
            weights, region, tag = array['weights'], array['regions'], array['tag_index']
            assert array['layer_ids'].tolist() == expected_layers
            assert weights.shape[0] == len(expected_layers) and weights.shape[-1] == len(region)
            assert np.isfinite(weights).all() and (weights >= 0).all()
            assert np.allclose(weights.sum(-1), 1, atol=1e-5)
            assert set(region.tolist()) == {0, 1, 2} and not np.any((tag >= 0) & (region != 1))
            mean = weights.mean(axis=(0, 1))
            lead, remaining = (tag >= 0) & (tag < 10), tag >= 10
            tag_positions = np.flatnonzero(region == 1)
            leading_boundaries.append(float(np.searchsorted(tag_positions, np.flatnonzero(lead)[-1], side='right') / len(tag_positions)))
            a, b = float(mean[lead].mean()), float(mean[remaining].mean())
            per_image.append({'image': row['image'], 'leading_density': a, 'remaining_density': b,
                              'leading_density_ratio': a/b, 'leading_mass': float(mean[lead].sum()),
                              **{f'{label}_mass': row['region_mass'][j] for j, label in enumerate(['visual','tag','prompt'])},
                              **{f'{label}_density': row['region_density'][j] for j, label in enumerate(['visual','tag','prompt'])}})
            per_layer = weights.mean(axis=1)
            bins = []
            for j in range(3):
                raw = per_layer[:, region == j]
                n = raw.shape[-1]
                resampled = []
                for values in raw:
                    cumulative = np.r_[0., np.cumsum(values, dtype=np.float64)]
                    mass = np.diff(np.interp(np.linspace(0, n, 65), np.arange(n+1), cumulative))
                    density = mass / (n/64)
                    assert np.isclose(density.sum()*(n/64), values.sum(), atol=1e-6)
                    resampled.append(density)
                bins.append(np.stack(resampled))
            normalized.append(np.stack(bins))
        write_csv(out / f'{model}_per_image.csv', per_image)
        maps[model] = (np.mean(normalized, axis=0), expected_layers, float(np.mean(leading_boundaries)))
        lead_values = np.array([r['leading_density'] for r in per_image])
        rest_values = np.array([r['remaining_density'] for r in per_image])
        p_attention = float(wilcoxon(lead_values, rest_values, alternative='two-sided').pvalue)
        row_summary = {'model': model, 'n': 294, 'layers': len(expected_layers),
                       'leading_density_mean': float(lead_values.mean()), 'remaining_density_mean': float(rest_values.mean()),
                       'ratio_of_mean_densities': float(lead_values.mean()/rest_values.mean()),
                       'leading_greater_count': int((lead_values > rest_values).sum()), 'p_leading_vs_rest': p_attention}
        for field in ['visual_mass','tag_mass','prompt_mass','visual_density','tag_density','prompt_density']:
            row_summary[field] = float(np.mean([r[field] for r in per_image]))
        summaries.append(row_summary)
        flags = {}
        for mode in ['plain', 'c', 'dc']:
            path = args.first_tags / model / mode / 'records.jsonl'
            source_rows = [r for r in read_jsonl(path) if r['split'] == 'test']
            assert len(source_rows) == 294 and {r['image'] for r in source_rows} == set(names)
            by_name = {r['image']: r for r in source_rows}
            flags[mode] = [by_name[name]['first_tag'] in leading and by_name[name]['first_tag'] not in gt[name] for name in names]
            valid = sum(r['is_candidate'] for r in source_rows)
            wrong = sum(flags[mode])
            first_rows.append({'model': model, 'mode': mode, 'n': 294, 'valid_candidate_first_tags': valid,
                               'wrong_leading': wrong, 'wrong_leading_pct_all': wrong/294*100,
                               'wrong_leading_pct_valid': wrong/valid*100 if valid else None})
            sources[str(path)] = sha(path)
        comparison[model] = {'c_vs_dc': paired_change(flags['c'], flags['dc']),
                             'plain_vs_dc': paired_change(flags['plain'], flags['dc'])}
        by_name = {r['image']: r for r in per_image}
        rho, p = spearmanr([by_name[n]['leading_density_ratio'] for n in names], flags['c'])
        comparison[model]['exploratory_attention_vs_wrong_leading_c'] = {
            'spearman_rho': float(rho) if np.isfinite(rho) else None,
            'p_unadjusted': float(p) if np.isfinite(p) else None}
    for row, p in zip(summaries, holm([r['p_leading_vs_rest'] for r in summaries])):
        row['p_leading_vs_rest_holm_two_models'] = p
    adjusted = holm([comparison[m]['c_vs_dc']['p_exact_mcnemar_two_sided'] for m in comparison])
    for model, p in zip(comparison, adjusted): comparison[model]['c_vs_dc']['p_holm_two_models'] = p
    write_csv(out / 'attention_summary.csv', summaries)
    write_csv(out / 'leading_first_tags.csv', first_rows)
    vmax = max(float(v[0].max()) for v in maps.values())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6})
    for model, (grid, layers, boundary) in maps.items():
        fig, axes = plt.subplots(1, 3, figsize=(3.35, args.figure_height), sharey=True)
        for j, (axis, title) in enumerate(zip(axes, ['Visual', 'Candidate tags', 'Other prompt'])):
            picture = axis.imshow(grid[j], aspect='auto', origin='upper', cmap='magma',
                                  norm=LogNorm(vmin=1e-6, vmax=vmax), extent=(0, 1, len(layers)-.5, -.5))
            axis.set_title(title, fontsize=6)
            axis.set_xticks([0, .5, 1]); axis.tick_params(labelsize=5, length=2)
            axis.get_xticklabels()[0].set_ha('left')
            axis.get_xticklabels()[-1].set_ha('right')
            if j == 1: axis.axvline(boundary, color='cyan', linestyle=':', linewidth=.7)
        # 층을 줄이지 않고 축 눈금과 반복 라벨만 간소화한다.
        ticks = np.unique(np.linspace(0, len(layers)-1, min(4, len(layers))).astype(int))
        axes[0].set_yticks(ticks, [str(layers[i]+1) for i in ticks]); axes[0].set_ylabel('Layer', fontsize=6)
        fig.subplots_adjust(left=.10, right=.81, bottom=.32, top=.80, wspace=.13)
        fig.text(.52, .035, 'Relative position  |  Color: mean softmax probability (log scale)', ha='center', fontsize=5)
        color_axis = fig.add_axes([.845, .32, .018, .48])
        colorbar = fig.colorbar(picture, cax=color_axis, orientation='vertical')
        colorbar.ax.tick_params(labelsize=5, length=2, pad=1)
        fig.savefig(out / f'{model}_attention.png', dpi=400, bbox_inches='tight', pad_inches=.03)
        fig.savefig(out / f'{model}_attention.pdf', bbox_inches='tight', pad_inches=.03)
        plt.close(fig)
    report = {'n_per_model': 294, 'leading_k': 10, 'leading_tags': candidates[:10],
              'attention': summaries, 'first_tag_counts': first_rows, 'paired_tests': comparison,
              'sources': sources, 'analysis_sha256': sha(__file__),
              'figure_layout': {'canvas_width_inches': 3.35, 'canvas_height_inches': args.figure_height,
                                'layers_unchanged': True, 'shared_vertical_colorbar': True},
              'figure': 'All heads averaged within each layer, then all294 images equally averaged. Mass-preserving average token probability in64 relative-position bins per region. Cyan line ends fixed first10 candidate tags. Shared log color scale; no chosen examples/layers/heads.',
              'limitations': 'Prefill attention identical across modes; correction operates on logits. Qwen full softmax layers only, pre-output-gate. Fixed candidate order cannot isolate causal position effects from tag identity or semantics. Evidence is observational.'}
    (out / 'verification.json').write_text(json.dumps({'passed': True, 'attention_records': 588,
        'saved_first_tag_records_used': 1764, 'all_arrays_checked': True}, indent=2), encoding='utf-8')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
