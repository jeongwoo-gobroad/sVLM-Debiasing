import os
import re
import json
import argparse
import math
from datetime import datetime

import torch
import KELIP.kelip as kelip
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pics_folder",      type=str,   default="D:\\training_data\\seomun\\reordered_anon\\val")
parser.add_argument("--tags_file",        type=str,   default="D:\\training_data\\seomun\\refined_tags.txt")
parser.add_argument("--ground_truth_file",type=str,   default="../seomun_gt_gen/gpt-5.4_gen_val_v5.jsonl")
parser.add_argument("--out_file",         type=str,   default="./kelip_threshold_train_result.txt")
parser.add_argument("--log_file",         type=str,   default="./kelip_threshold_search_log.jsonl")
parser.add_argument("--best_file",        type=str,   default="./kelip_threshold_best_result.json")
parser.add_argument("--lo",               type=float, default=-1.0)
parser.add_argument("--hi",               type=float, default=1.0)
parser.add_argument("--precision",        type=int,   default=2)
parser.add_argument("--margin",           type=int,   default=1)
parser.add_argument("--text_template",    type=str,   default="이 사진에는 {label}이(가) 있습니다.")
parser.add_argument("--model_arch",       type=str,   default="ViT-B/32")
args = parser.parse_args()

LO_INIT   = args.lo
HI_INIT   = args.hi
PRECISION = args.precision
MARGIN    = args.margin
KEY       = 1 

with open(args.tags_file, "r", encoding="utf-8") as f:
    tags = [line.strip() for line in f if line.strip()]

ground_truth = {}
with open(args.ground_truth_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ground_truth.update(json.loads(line.strip()))

image_filenames = []
for fname in os.listdir(args.pics_folder):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')) and fname in ground_truth:
        image_filenames.append(fname)

if not image_filenames:
    exit()

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess_img, tokenizer = kelip.build_model(args.model_arch)
model = model.to(device)
model.eval()

text_sentences = [args.text_template.replace("{label}", tag) for tag in tags]
text_tokens = tokenizer.encode(text_sentences).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # L2 정규화

text_features = text_features.cpu()  # (num_tags, D)

print("이미지 임베딩 추출 중...")
image_embeddings = {}  # fname → (num_tags,) cosine similarity tensor

with torch.no_grad():
    for fname in tqdm(image_filenames, desc="Extracting Image Features"):
        path = os.path.join(args.pics_folder, fname)
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess_img(img).unsqueeze(0).to(device)
            feat = model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 정규화 (kelipscore.py 패턴)
            # 코사인 유사도: (1, D) @ (D, num_tags) → (1, num_tags)
            sim = torch.matmul(feat.cpu(), text_features.T).squeeze(0)  # (num_tags,)
            image_embeddings[fname] = sim
        except Exception as e:
            print(f"[WARN] {fname} 임베딩 실패: {e}")

valid_filenames = [f for f in image_filenames if f in image_embeddings]
if not valid_filenames:
    print("임베딩된 이미지가 없습니다.")
    exit()

def evaluate_threshold(threshold: float) -> tuple[float, dict]:
    total_jaccard_distance = 0.0
    all_predictions = {}

    for fname in valid_filenames:
        sim = image_embeddings[fname]  # (num_tags,)
        mask = sim >= threshold
        pred_tags = set(tag for tag, selected in zip(tags, mask.tolist()) if selected)

        gt_tags = set(ground_truth.get(fname, []))

        if not pred_tags and not gt_tags:
            j_dist = 0.0
        elif not pred_tags or not gt_tags:
            j_dist = 1.0
        else:
            intersection = len(pred_tags.intersection(gt_tags))
            union = len(pred_tags.union(gt_tags))
            j_dist = 1.0 - (intersection / union)

        total_jaccard_distance += j_dist
        all_predictions[fname] = list(pred_tags)

    avg_jd = total_jaccard_distance / len(valid_filenames)
    return avg_jd, all_predictions

def save_result(avg_jd: float, all_predictions: dict, threshold: float):
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write(f"Average_Jaccard_Distance: {avg_jd:.4f}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write("-" * 50 + "\n")
        for fname, predicted_tags in all_predictions.items():
            f.write(f"{fname}: {predicted_tags}\n")

cache: dict[float, float] = {}

def rnd(v: float, d: int) -> float:
    return round(v, d)

def log_step(n: float, jd: float):
    with open(args.log_file, "a", encoding="utf-8") as f:
        entry = {
            "threshold": n,
            "jaccard": jd,
            "timestamp": datetime.now().isoformat(),
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def run_evaluation(n: float, digit: int) -> float | None:
    key = rnd(n, digit + 2)
    if key in cache:
        print(f"  [CACHE] threshold={key:.{digit+2}f} → J={cache[key]:.6f}")
        return cache[key]

    jd, _ = evaluate_threshold(key)

    cache[key] = jd
    log_step(key, jd)
    print(f"  [RESULT] threshold={key:.{digit+2}f} → J={jd:.6f}", flush=True)
    return jd

def find_valley(points: list[float], values: list[float], margin: int) -> tuple[float, float]:
    best_idx = int(min(range(len(values)), key=lambda i: values[i]))
    lo_idx = max(0, best_idx - margin)
    hi_idx = min(len(points) - 1, best_idx + margin)
    return points[lo_idx], points[hi_idx]

def hierarchical_search(lo: float, hi: float, precision: int) -> tuple[float, float]:
    best_n  = None
    best_jd = float("inf")

    for digit in tqdm(range(KEY, precision + 1), unit="step", desc="Grid Search"):
        step = round(10 ** (-digit), digit)

        grid = []
        v = rnd(lo, digit)
        while v <= rnd(hi, digit) + step * 0.5:
            grid.append(rnd(v, digit))
            v = rnd(v + step, digit)

        grid = sorted(set(grid))
        grid = [g for g in grid if rnd(lo, digit) - step * 0.01 <= g <= rnd(hi, digit) + step * 0.01]

        if not grid:
            print(f"  [SKIP] digit={digit}: 격자 비어있음 (lo={lo}, hi={hi}, step={step})")
            continue

        print(f"\n{'='*60}")
        print(f"[DIGIT {digit}/{precision}] step={step} range=[{lo}, {hi}] points={len(grid)}")

        values = []
        for pt in grid:
            jd = run_evaluation(pt, digit)
            if jd is None:
                jd = float("inf")
            values.append(jd)

            if jd < best_jd:
                best_jd = jd
                best_n  = pt

        print(f"\n  [SCAN RESULT] digit={digit}")
        for pt, jd in zip(grid, values):
            marker = " ◀ best" if pt == best_n else ""
            print(f"    threshold={pt:.{digit+2}f}  J={jd:.6f}{marker}")

        if len(grid) >= 2:
            lo, hi = find_valley(grid, values, MARGIN)
        else:
            lo = hi = grid[0]

        print(f"\n  [NARROW] 다음 탐색 구간: [{lo}, {hi}]")

        if rnd(hi - lo, digit + 2) <= step * 1.5 and digit < precision:
            lo = rnd(lo - step, digit)
            hi = rnd(hi + step, digit)
            print(f"  [EXPAND] 구간이 너무 좁아 ±step 확장: [{lo}, {hi}]")

    return best_n, best_jd

def main():
    print(f"[CONFIG] lo={LO_INIT}  hi={HI_INIT}  precision={PRECISION}  margin={MARGIN}")
    print(f"[CONFIG] tags={len(tags)}  images={len(valid_filenames)}")
    print(f"[CONFIG] text_template: {args.text_template}")

    best_n, best_jd = hierarchical_search(LO_INIT, HI_INIT, PRECISION)

    final_jd, final_predictions = evaluate_threshold(best_n)
    save_result(final_jd, final_predictions, best_n)

    final = {
        "best_threshold": best_n,
        "best_jaccard":   best_jd,
        "total_evaluations": len(cache),
        "all_results": [{"threshold": k, "jaccard": v} for k, v in sorted(cache.items())],
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.best_file, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[DONE] 최적 threshold = {best_n:.{PRECISION}f}")
    print(f"       최소 Jaccard   = {best_jd:.6f}")
    print(f"       총 평가 횟수   = {len(cache)}")
    print(f"       결과 저장      → {args.out_file}")
    print(f"       최적값 저장    → {args.best_file}")
    print(f"       전체 로그      → {args.log_file}")

if __name__ == "__main__":
    main()
