# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

import argparse
import subprocess
import shutil
import re
import os
import json
from datetime import datetime
from tqdm import tqdm

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hierarchical grid optimizer for --feot_thres")
parser.add_argument("--lo",          type=float, default=0.0,    help="Initial lower bound of n")
parser.add_argument("--hi",          type=float, default=1.0,    help="Initial upper bound of n")
parser.add_argument("--precision",   type=int,   default=2,       help="Max decimal digits (default 6)")
parser.add_argument("--target_script", type=str, default=str(Path(__file__).with_name("qwen_n.py")), help="Path to script")
parser.add_argument("--margin",      type=int,   default=1,       help="Extra grid points to keep around best (default 1)")
parser.add_argument("--mode",        dest="mode", type=str)
parser.add_argument("--model_size",        dest="model_size", type=str, choices=["2b", "4b"], required=True)
parser.add_argument('--prompt_variant', choices=['default', 'a', 'b', 'c'], default='default',
                    help='default or paper Prompt 1/2/3 (a/b/c)')
parser.add_argument('--limit', type=int, default=0, help='Smoke test image count; 0 means the full split')
parser.add_argument('--output_dir', type=Path, default=REPO_ROOT / 'results' / 'val')
args = parser.parse_args()
if args.limit != 0:
    parser.error('Validation searches require the complete split; --limit must be 0')
if not (0 <= args.lo <= args.hi <= 1) or args.precision < 1 or args.margin < 1:
    parser.error('Use 0 <= lo <= hi <= 1, precision >= 1, margin >= 1')
args.output_dir.mkdir(parents=True, exist_ok=True)


LO_INIT      = args.lo
HI_INIT      = args.hi
PRECISION    = args.precision
SCRIPT_NAME  = args.target_script
MARGIN       = args.margin        # valley 범위를 best 좌우 MARGIN 칸만큼 확장
KEY          = 1

model_size = args.model_size

RESULT_LOG   = str(args.output_dir / f"qwen3.5_{model_size}/abs2_thres_search_log_{args.mode}.jsonl")
BEST_FILE    = str(args.output_dir / f"qwen3.5_{model_size}/abs2_thres_best_result_{args.mode}.json")
OUT_FILE     = str(args.output_dir / f"qwen3.5_{model_size}/qwen3.5_{model_size}_val_dc.txt")

if args.prompt_variant != 'default':
    suffix = f'_prompt[{args.prompt_variant}]'
    OUT_FILE = OUT_FILE.replace('_val_dc.txt', f'_val{suffix}_dc.txt')
    RESULT_LOG = RESULT_LOG.replace('.jsonl', suffix + '.jsonl')
    BEST_FILE = BEST_FILE.replace('.json', suffix + '.json')
for output in [OUT_FILE, RESULT_LOG, BEST_FILE]:
    Path(output).parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────
cache: dict[float, float] = {}   # n → jaccard (전체 실행에서 공유)

def rnd(v: float, d: int) -> float:
    return round(v, d)

def parse_jaccard(filepath: str) -> tuple[float | None, float | None]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            m = re.search(r"Average_Jaccard_Distance:\s*([\d.]+)", first)
            jd = float(m.group(1)) if m else None

            tag_counts = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # "val_1.jpg: ['태그1', '태그2', ...]" 형식
                tag_match = re.search(r":\s*(\[.*\])\s*$", line)
                if tag_match:
                    try:
                        tags = json.loads(tag_match.group(1).replace("'", '"'))
                        tag_counts.append(len(tags))
                    except json.JSONDecodeError:
                        pass

            avg_tags = round(sum(tag_counts) / len(tag_counts), 4) if tag_counts else None
            return jd, avg_tags
    except FileNotFoundError:
        pass
    return None, None


def log_step(n: float, jd: float, avg_tags: float | None):
    with open(RESULT_LOG, "a", encoding="utf-8") as f:
        entry = {
            "n": n,
            "jaccard": jd,
            "avg_tags_per_image": avg_tags,
            "timestamp": datetime.now().isoformat(),
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def run_qwen(n: float, digit: int) -> float | None:
    key = rnd(n, digit + 2)           # 부동소수점 키 안정화
    if key in cache:
        print(f"  [CACHE] n={key:.{digit+2}f}  →  J={cache[key]:.6f}")
        return cache[key]

    n_str = f"{key:.{max(digit, 6)}f}"
    cmd = [
        sys.executable,
        SCRIPT_NAME,
        "--prompt_variant", args.prompt_variant,
        "--output_dir", str(args.output_dir),
        "--constrained",
        "--debiasing",
        f"--ratio_thres={n_str}",
        f"--model_size={model_size}"
    ]
    print(f"  [RUN]   n={n_str}  →  {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, timeout=7200, check=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Validation inference timed out; no stale result was read")
    except Exception as e:
        raise RuntimeError("Validation inference failed; no stale result was read") from e

    # qwen.py 저장 파일: stat_save_file_name + "_dcf.txt"
    out_file = OUT_FILE
    with open(out_file, encoding='utf-8') as stream:
        count = sum(bool(re.match(r'^val_\d+\.jpg:', line)) for line in stream)
    if count != 440:
        raise ValueError(f'Expected 440 validation predictions, got {count}')
    jd, avg_tags = parse_jaccard(out_file)
    if jd is None:
        print(f" [WARN] 결과 파싱 실패 ({out_file})")
        return None

    archive = Path(OUT_FILE).parent / 'alpha_predictions' / args.prompt_variant
    archive.mkdir(parents=True, exist_ok=True)
    saved = archive / f'alpha_{n_str}.txt'
    shutil.copyfile(out_file, saved)
    metadata = Path(out_file + '.metadata.json')
    if metadata.exists():
        shutil.copyfile(metadata, str(saved) + '.metadata.json')
    cache[key] = jd
    log_step(key, jd, avg_tags)
    print(f" [RESULT] n={n_str} → J={jd:.6f}  avg_tags={avg_tags}", flush=True)
    return jd

# ──────────────────────────────────────────────
# 핵심: 계층적 그리드 탐색
# ──────────────────────────────────────────────
def find_valley(points: list[float], values: list[float], margin: int) -> tuple[float, float]:
    """
    points/values 배열에서 local minimum 을 찾고,
    그 좌우 margin 칸을 포함한 [lo_new, hi_new] 구간을 반환.
    모든 점이 단조이면(valley 없음) 최솟값 끝 방향으로 구간 설정.
    """
    best_idx = int(min(range(len(values)), key=lambda i: values[i]))

    lo_idx = max(0, best_idx - margin)
    hi_idx = min(len(points) - 1, best_idx + margin)

    return points[lo_idx], points[hi_idx]


def hierarchical_search(lo: float, hi: float, precision: int) -> tuple[float, float]:
    """
    digit=1 (0.1 단위) 부터 digit=precision 까지 반복.
    각 단계에서 현재 [lo, hi] 를 step 간격으로 scan하고
    valley (local minima 구간) 를 찾아 다음 단계 구간으로 넘긴다.
    반환: (best_n, best_jaccard)
    """
    best_n   = None
    best_jd  = float("inf")

    for digit in tqdm(range(KEY, precision + 1), unit="job", desc="Computing..."):
        step = round(10 ** (-digit), digit)

        # 현재 구간을 step 간격으로 격자 생성
        grid = []
        v = rnd(lo, digit)
        while v <= rnd(hi, digit) + step * 0.5:   # 0.5*step: 부동소수점 끝점 포함 보정
            grid.append(rnd(v, digit))
            v = rnd(v + step, digit)

        # 중복 제거 & 정렬
        grid = sorted(set(grid))
        # 범위 바깥 점 제거
        grid = [g for g in grid if rnd(lo, digit) - step*0.01 <= g <= rnd(hi, digit) + step*0.01]

        if not grid:
            print(f"  [SKIP] digit={digit}: 격자 비어있음 (lo={lo}, hi={hi}, step={step})")
            continue

        print(f"\n{'='*60}")
        print(f"[DIGIT {digit}/{precision}]  step={step}  range=[{lo}, {hi}]  points={len(grid)}")

        values = []
        for pt in grid:
            jd = run_qwen(pt, digit)
            if jd is None:
                jd = float("inf")   # 실패한 점은 최악값으로 처리
            values.append(jd)

            # 전역 최솟값 추적
            if jd < best_jd:
                best_jd = jd
                best_n  = pt

        # ── 결과 출력 ──
        print(f"\n  [SCAN RESULT] digit={digit}")
        for pt, jd in zip(grid, values):
            marker = " ◀ best" if pt == best_n else ""
            print(f"    n={pt:.{digit+2}f}  J={jd:.6f}{marker}")

        # ── valley 탐지 → 다음 구간 결정 ──
        if len(grid) >= 2:
            lo, hi = find_valley(grid, values, MARGIN)
        else:
            lo = hi = grid[0]

        print(f"\n  [NARROW] 다음 탐색 구간: [{lo}, {hi}]")

        # 수렴: 구간 너비가 step 이하면 더 이상 좁힐 수 없음
        if rnd(hi - lo, digit + 2) <= step * 1.5 and digit < precision:
            # 구간이 이미 최소 → lo±step, hi±step 으로 약간 넓혀서 다음 digit 에서 3점 이상 확보
            lo = max(LO_INIT, rnd(lo - step, digit))
            hi = min(HI_INIT, rnd(hi + step, digit))
            print(f"  [EXPAND] 구간이 너무 좁아 ±step 확장: [{lo}, {hi}]")

    return best_n, best_jd


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    print(f"[CONFIG] lo={LO_INIT}  hi={HI_INIT}  precision={PRECISION}")
    print(f"[CONFIG] script={SCRIPT_NAME}  margin={MARGIN}")

    best_n, best_jd = hierarchical_search(LO_INIT, HI_INIT, PRECISION)

    # ── 최종 결과 저장 ──
    final = {
        "best_n":            best_n,
        "best_jaccard":      best_jd,
        "total_evaluations": len(cache),
        "all_results":       [{"n": k, "jaccard": v} for k, v in sorted(cache.items())],
        "timestamp":         datetime.now().isoformat(),
    }
    with open(BEST_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[DONE] 최적 n          = {best_n:.{PRECISION}f}")
    print(f"       최소 Jaccard    = {best_jd:.6f}")
    print(f"       총 실행 횟수    = {len(cache)}")
    print(f"       결과 저장       → {BEST_FILE}")
    print(f"       전체 로그       → {RESULT_LOG}")


if __name__ == "__main__":
    main()