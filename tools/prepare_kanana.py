# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Supply weights without replacing the experiment's local Kanana implementation."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    manifest = json.loads((ROOT / 'models/kanana/weights.json').read_text(encoding='utf-8'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--local_source', type=Path, help='Original .kanana_model folder')
    group.add_argument('--revision', default=manifest['revision'], help='Hugging Face revision; weights must match the recorded experiment')
    args = parser.parse_args()
    destination = ROOT / 'models/kanana'
    index = json.loads((destination / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    names = sorted(set(index['weight_map'].values()))
    for name in tqdm(names, desc='Prepare Kanana weights'):
        target = destination / name
        if target.exists():
            raise FileExistsError(f'{target} already exists; existing weights were preserved')
        if args.local_source:
            source = args.local_source / name
        else:
            from huggingface_hub import hf_hub_download
            source = Path(hf_hub_download('kakaocorp/kanana-1.5-v-3b-instruct', name, revision=args.revision))
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != manifest['weights'][name]['sha256']:
            raise ValueError(f'{name}: weights do not match the recorded experiment')
        shutil.copy2(source, target)


if __name__ == '__main__':
    main()
