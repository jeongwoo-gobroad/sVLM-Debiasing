# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Read and review exactly one retained image-response pair."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--key')
    parser.add_argument('--tags_json')
    parser.add_argument('--prose', choices=['yes', 'no'])
    parser.add_argument('--reason')
    args = parser.parse_args()
    path = args.analysis / ('pair_reviews.jsonl' if args.key else 'review_queue.jsonl')
    with path.open(encoding='utf-8') as stream:
        row = next((r for r in map(json.loads, stream) if not args.key or r['key'] == args.key), None)
    if row is None:
        print('No matching response.'); return
    if args.tags_json is None:
        print(json.dumps({k: row[k] for k in ['key', 'image_path', 'raw_text', 'evaluated_text', 'excluded_tail', 'ambiguous_segments']}, ensure_ascii=False))
        return
    if args.prose is None or not args.reason: parser.error('A prose decision and reason are required')
    tags = json.loads(args.tags_json)
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags): parser.error('tags_json must be a string list')
    decision = {'key': row['key'], 'raw_sha256': row['raw_sha256'],
                'evaluated_sha256': hashlib.sha256(row['evaluated_text'].encode()).hexdigest(),
                'decision': {'prompt_noncompliance': args.prose == 'yes', 'generated_tags': tags},
                'reason': args.reason}
    with (args.analysis / 'overrides.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(decision, ensure_ascii=False)+'\n')


if __name__ == '__main__': main()
