# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
import json
import ast
import numpy as np
from scipy import stats


def calculate_jaccard_distance(set_true, set_pred):
    intersection = len(set_true.intersection(set_pred))
    union = len(set_true.union(set_pred))
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)


def calculate_f1_score(set_true, set_pred):
    tp = len(set_true.intersection(set_pred))
    fp = len(set_pred - set_true)
    fn = len(set_true - set_pred)

    if tp == 0:
        if fp == 0 and fn == 0:
            return 1.0
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def calculate_rank_biserial(w_plus, n):
    """
    Kerby(2014)의 matched-pairs rank-biserial correlation 공식.
    W+는 반드시 alternative='greater'로 호출하여 얻은 양의 순위합이어야 한다.
    전체 순위합 R = n*(n+1)/2, W- = R - W+ 이므로:
        r_rb = (W+ - W-) / R = (2*W+ - R) / R
    값 범위: [-1, +1]
    +1 → A가 모든 쌍에서 우세, -1 → B가 모든 쌍에서 우세
    """
    total_rank_sum = n * (n + 1) / 2
    if total_rank_sum == 0:
        return 0.0
    w_minus = total_rank_sum - w_plus
    return (w_plus - w_minus) / total_rank_sum


def interpret_rank_biserial(r_rb):
    """
    Cohen(1988) 기준을 비모수 효과 크기에 적용한 해석.
    절댓값 기준: 0.1 미만=매우 작음, 0.1~0.3=작음, 0.3~0.5=중간, 0.5 이상=큼
    """
    abs_r = abs(r_rb)
    if abs_r >= 0.5:
        return "큼 (Large effect)"
    elif abs_r >= 0.3:
        return "중간 (Medium effect)"
    elif abs_r >= 0.1:
        return "작음 (Small effect)"
    else:
        return "매우 작음 (Negligible effect)"


def load_answer_jsonl(filepath):
    answers = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for filename, tags in data.items():
                answers[filename] = set(tags)
    return answers


def load_candidate_txt(filepath):
    candidates = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:]
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if ':' in line:
                filename, tags_str = line.split(':', 1)
                filename = filename.strip()
                tags_str = tags_str.strip()

                try:
                    tags_list = ast.literal_eval(tags_str)
                    candidates[filename] = set(tags_list)
                except (ValueError, SyntaxError):
                    pass
    return candidates


def wilcoxon_with_rbc(x, y, n, p_value_threshold):
    """
    p-value는 양측 검정으로, W+는 단측(greater)으로 각각 추출한다.
    - alternative='two-sided': min(W+, W-)를 반환 → p-value에만 사용
    - alternative='greater' : W+를 반환 → rank-biserial 계산에 사용
    두 호출의 zero_method를 동일하게 맞춰야 n_r(유효 샘플 수)이 일치한다.
    """
    _, p_value = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
    w_plus, _ = stats.wilcoxon(x, y, zero_method='wilcox', alternative='greater')

    # zero_method='wilcox'로 0인 차이를 제거하므로, 실제 유효 샘플 수를 다시 계산
    d = np.array(x) - np.array(y)
    n_r = int(np.sum(d != 0))  # 차이가 0이 아닌 쌍의 수

    r_rb = calculate_rank_biserial(w_plus, n_r)
    return w_plus, p_value, r_rb


def evaluate_models(answer_path, cand_a_path, cand_b_path, p_value_threshold):
    answers = load_answer_jsonl(answer_path)
    cand_a = load_candidate_txt(cand_a_path)
    cand_b = load_candidate_txt(cand_b_path)

    dist_a = []
    dist_b = []
    f1_a = []
    f1_b = []

    tp_a_total = fp_a_total = fn_a_total = 0
    tp_b_total = fp_b_total = fn_b_total = 0

    for filename, true_tags in answers.items():
        if filename in cand_a and filename in cand_b:
            pred_a = cand_a[filename]
            pred_b = cand_b[filename]

            d_a = calculate_jaccard_distance(true_tags, pred_a)
            d_b = calculate_jaccard_distance(true_tags, pred_b)

            f1_sample_a = calculate_f1_score(true_tags, pred_a)
            f1_sample_b = calculate_f1_score(true_tags, pred_b)

            dist_a.append(d_a)
            dist_b.append(d_b)
            f1_a.append(f1_sample_a)
            f1_b.append(f1_sample_b)

            tp_a_total += len(true_tags & pred_a)
            fp_a_total += len(pred_a - true_tags)
            fn_a_total += len(true_tags - pred_a)

            tp_b_total += len(true_tags & pred_b)
            fp_b_total += len(pred_b - true_tags)
            fn_b_total += len(true_tags - pred_b)

    if not dist_a or not dist_b:
        raise ValueError("비교할 수 있는 유효한 데이터 쌍이 없습니다.")

    n_samples = len(dist_a)

    jaccard_w_plus, jaccard_p_value, jaccard_r_rb = wilcoxon_with_rbc(
        dist_a, dist_b, n_samples, p_value_threshold
    )
    f1_w_plus, f1_p_value, f1_r_rb = wilcoxon_with_rbc(
        f1_a, f1_b, n_samples, p_value_threshold
    )

    mean_jaccard_a = np.mean(dist_a)
    mean_jaccard_b = np.mean(dist_b)

    mean_f1_a = np.mean(f1_a)
    mean_f1_b = np.mean(f1_b)

    micro_f1_a = (
        0.0 if (2 * tp_a_total + fp_a_total + fn_a_total) == 0
        else (2 * tp_a_total) / (2 * tp_a_total + fp_a_total + fn_a_total)
    )
    micro_f1_b = (
        0.0 if (2 * tp_b_total + fp_b_total + fn_b_total) == 0
        else (2 * tp_b_total) / (2 * tp_b_total + fp_b_total + fn_b_total)
    )

    return {
        "jaccard_distance": {
            "mean_a": mean_jaccard_a,
            "mean_b": mean_jaccard_b,
            "w_plus": jaccard_w_plus,
            "p_value": jaccard_p_value,
            "rank_biserial": jaccard_r_rb,
            "rbc_interpretation": interpret_rank_biserial(jaccard_r_rb),
            "significant": jaccard_p_value < p_value_threshold,
        },
        "f1_samples": {
            "mean_a": mean_f1_a,
            "mean_b": mean_f1_b,
            "w_plus": f1_w_plus,
            "p_value": f1_p_value,
            "rank_biserial": f1_r_rb,
            "rbc_interpretation": interpret_rank_biserial(f1_r_rb),
            "significant": f1_p_value < p_value_threshold,
        },
        "f1_micro": {
            "a": micro_f1_a,
            "b": micro_f1_b,
        },
    }


