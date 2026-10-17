# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Shared helpers for verified response and preference analysis."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = ['qwen3.5_4b', 'kanana']
MODES = ['plain', 'c', 'dc']
COMPARISONS = [('plain', 'c'), ('plain', 'dc'), ('c', 'dc')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def holm(values):
    order = np.argsort(values)
    answer = np.zeros(len(values))
    maximum = 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, min(1.0, (len(values)-rank)*values[index]))
        answer[index] = maximum
    return answer.tolist()
