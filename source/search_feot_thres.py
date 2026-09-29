import argparse
import subprocess
import re
import os
import json
from datetime import datetime

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hierarchical grid optimizer for --feot_thres")
parser.add_argument("--lo",          type=float, default=0.0,    help="Initial lower bound of n")
parser.add_argument("--hi",          type=float, default=2.0,    help="Initial upper bound of n")
parser.add_argument("--precision",   type=int,   default=6,       help="Max decimal digits (default 6)")
parser.add_argument("--qwen_script", type=str,   default="qwen.py", help="Path to qwen.py")
parser.add_argument("--margin",      type=int,   default=1,       help="Extra grid points to keep around best (default 1)")
args = parser.parse_args()

LO_INIT      = args.lo
HI_INIT      = args.hi
PRECISION    = args.precision
QWEN_SCRIPT  = args.qwen_script
MARGIN       = args.margin        # valley 범위를 best 좌우 MARGIN 칸만큼 확장

RESULT_LOG   = "thres_search_xlog_n6.jsonl"
BEST_FILE    = "thres_best_xresult_n6.json"

# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────
cache: dict[float, float] = {}   # n → jaccard (전체 실행에서 공유)

def rnd(v: float, d: int) -> float:
    return round(v, d)

def parse_jaccard(filepath: str) -> float | None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        m = re.search(r"Average_Jaccard_Distance:\s*([\d.]+)", first)
        if m:
            return float(m.group(1))
    except FileNotFoundError:
        pass
    return None

def log_step(n: float, jd: float):
    with open(RESULT_LOG, "a", encoding="utf-8") as f:
        entry = {"n": n, "jaccard": jd, "timestamp": datetime.now().isoformat()}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def run_qwen(n: float, digit: int) -> float | None:
    key = rnd(n, digit + 2)           # 부동소수점 키 안정화
    if key in cache:
        print(f"  [CACHE] n={key:.{digit+2}f}  →  J={cache[key]:.6f}")
        return cache[key]

    n_str = f"{key:.{max(digit, 6)}f}"
    cmd = [
        "../.venv/Scripts/python.exe",
        QWEN_SCRIPT,
        "--constrained",
        # "--force_eot",
        "--debiasing",
        # f"--feot_thres={n_str}",
        f"--ratio_thres={n_str}"
    ]
    print(f"  [RUN]   n={n_str}  →  {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, timeout=7200, check=False)
    except subprocess.TimeoutExpired:
        print("  [ERROR] Timeout.")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    # qwen.py 저장 파일: stat_save_file_name + "_dcf.txt"
    out_file = "qwen_4_gpt5-4_v4_dc.txt"
    jd = parse_jaccard(out_file)
    if jd is None:
        print(f"  [WARN]  결과 파싱 실패 ({out_file})")
        return None

    cache[key] = jd
    log_step(key, jd)
    print(f"  [RESULT] n={n_str}  →  J={jd:.6f}", flush=True)
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

    for digit in range(1, precision + 1):
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
            lo = rnd(lo - step, digit)
            hi = rnd(hi + step, digit)
            print(f"  [EXPAND] 구간이 너무 좁아 ±step 확장: [{lo}, {hi}]")

    return best_n, best_jd


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    print(f"[CONFIG] lo={LO_INIT}  hi={HI_INIT}  precision={PRECISION}")
    print(f"[CONFIG] qwen_script={QWEN_SCRIPT}  margin={MARGIN}")

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