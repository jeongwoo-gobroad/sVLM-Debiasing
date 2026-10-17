# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Index the first completed tag from saved full responses; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    args = parser.parse_args()
    candidates = set(args.candidates.read_text(encoding='utf-8').splitlines())
    for model in ['qwen3.5_4b', 'kanana']:
        for mode in ['plain', 'c', 'dc']:
            source = args.input / model / mode / 'records.jsonl'
            target = args.output / model / mode
            target.mkdir(parents=True, exist_ok=True)
            rows = []
            with source.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['split'] != 'test': continue
                    if not row.get('generated_ids'): raise ValueError('Raw generated IDs are required')
                    first = row['raw_text'].split(',', 1)[0].strip()
                    complete = ',' in row['raw_text'] or row['ended_with_eos'] or first in candidates
                    rows.append({'image': row['image'], 'split': 'test', 'model': model, 'mode': mode,
                                 'first_token_id': row['generated_ids'][0], 'first_tag': first,
                                 'is_candidate': complete and first in candidates,
                                 'first_tag_complete': complete})
            assert len(rows) == 294 and len({r['image'] for r in rows}) == 294
            (target / 'records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
            (target / 'source.json').write_text(json.dumps({'path': str(source.resolve()),
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
