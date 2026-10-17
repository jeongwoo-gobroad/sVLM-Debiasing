# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Check dataset keys, candidate membership, image decoding and byte hashes."""
import hashlib
import json
from pathlib import Path
import os
from statistics import mean, pstdev

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SVLM_DATA_ROOT", ROOT / "datasets")).expanduser().resolve()


def main():
    candidates = [line.strip() for line in (DATA_ROOT / 'tag_candidates/refined_tags.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(candidates) == len(set(candidates)) == 124, 'Expected 124 unique candidate labels'
    split_hashes = {}
    for split, expected in [('val', 440), ('test', 294)]:
        gt = {}
        for line in (DATA_ROOT / f'tag_gts/human_gt_{split}.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                assert not (gt.keys() & record.keys()), 'Duplicate GT keys'
                gt.update(record)
        images = {p.name: p for p in (DATA_ROOT / f'imgs/{split}').glob('*.jpg')}
        assert len(gt) == expected and gt.keys() == images.keys(), 'Split coverage mismatch'
        hashes = set()
        for name, tags in tqdm(gt.items(), desc=f'Validate {split}'):
            assert set(tags) <= set(candidates), f'{name}: unknown GT labels'
            with Image.open(images[name]) as image:
                image.verify()
            hashes.add(hashlib.sha256(images[name].read_bytes()).hexdigest())
        assert len(hashes) == expected, 'Duplicate image bytes within split'
        split_hashes[split] = hashes
        sizes = [len(set(tags)) for tags in gt.values()]
        print(f'{split}: {len(gt)} images; labels {mean(sizes):.4f} +/- {pstdev(sizes):.4f}')
    assert not split_hashes['val'] & split_hashes['test'], 'Identical image bytes cross splits'
    manifest_path = DATA_ROOT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    for record in tqdm(manifest, desc='Verify copied files'):
        path = DATA_ROOT / Path(record['destination']).relative_to('datasets')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], str(path)
    print(f'Validated {len(candidates)} candidates and {len(manifest)} copied files')


if __name__ == '__main__':
    main()
