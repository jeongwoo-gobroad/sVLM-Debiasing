# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Paired test-set ablation comparisons; the pooled study is not used here."""
import csv
import argparse
import json
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SVLM_DATA_ROOT", ROOT / "datasets")).expanduser().resolve()
sys.path.insert(0,str(ROOT/'evaluation'))
from evaluate import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference_root',type=Path,default=Path(os.environ.get('SVLM_REFERENCE_ROOT',ROOT/'reference_results')))
    parser.add_argument('--output',type=Path,default=ROOT/'results/table5')
    args=parser.parse_args()
    gt=DATA_ROOT / 'tag_gts/human_gt_test.jsonl'
    rows=[];results={}
    for model in ['qwen3.5_4b','kanana']:
        folder=args.reference_root/'test'/model
        plain=folder/f'{model}_test_plain.txt'
        for mode in ['plain','c','dc']:
            metrics=evaluate(gt,plain,folder/f'{model}_test_{mode}.txt') if mode!='plain' else None
            if mode=='plain':
                baseline=evaluate(gt,plain,folder/f'{model}_test_c.txt')
            jd=metrics['jaccard_distance'] if metrics else baseline['jaccard_distance']
            f1=metrics['f1_samples'] if metrics else baseline['f1_samples']
            rows.append({'model':model,'mode':mode,'n':294,
                         'dJ':jd['mean_b'] if metrics else jd['mean_a'],
                         'F1':f1['mean_b'] if metrics else f1['mean_a'],
                         'p_dJ_vs_plain':jd['p_value'] if metrics else '',
                         'p_F1_vs_plain':f1['p_value'] if metrics else '',
                         'improved_dJ_p005':bool(jd['mean_b']<jd['mean_a'] and jd['p_value']<.05) if metrics else '',
                         'improved_F1_p005':bool(f1['mean_b']>f1['mean_a'] and f1['p_value']<.05) if metrics else ''})
            if metrics:results[model+'_'+mode+'_vs_plain']=metrics
        results[model+'_dc_vs_c']=evaluate(gt,folder/f'{model}_test_c.txt',folder/f'{model}_test_dc.txt')
    dest=args.output
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'table5_significance.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    with (dest/'table5_significance.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for row in rows:print(row)


if __name__=='__main__':main()
