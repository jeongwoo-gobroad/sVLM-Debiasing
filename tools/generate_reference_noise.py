# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Recreate the supplied reference asset explicitly, never during inference."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rng = np.random.default_rng(seed=42)
    image = Image.fromarray(rng.integers(0, 256, (540, 540, 3), dtype=np.uint8))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format='JPEG', quality=75)
    actual = hashlib.sha256(args.output.read_bytes()).hexdigest()
    expected = 'f5e0722fda2a4a6f307b5df242658d818845958140e002d307f907975624ef62'
    if actual != expected:
        raise ValueError('Encoded JPEG differs from the supplied asset; use its original bytes')
    print(actual)


if __name__ == '__main__':
    main()
