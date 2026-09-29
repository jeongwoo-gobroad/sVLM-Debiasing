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

    jaccard_statistic, jaccard_p_value = stats.wilcoxon(dist_a, dist_b)
    f1_statistic, f1_p_value = stats.wilcoxon(f1_a, f1_b)

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
            "statistic": jaccard_statistic,
            "p_value": jaccard_p_value,
            "significant": jaccard_p_value < p_value_threshold,
        },
        "f1_samples": {
            "mean_a": mean_f1_a,
            "mean_b": mean_f1_b,
            "statistic": f1_statistic,
            "p_value": f1_p_value,
            "significant": f1_p_value < p_value_threshold,
        },
        "f1_micro": {
            "a": micro_f1_a,
            "b": micro_f1_b,
        },
    }


if __name__ == "__main__":
    answer_file = "../gpt_gt_gen/gpt5_4_gt_new_v2.jsonl"
    candidate_a_file = "../gen-class-others/qwen_4_gpt5-4_v4_plain.txt"
    candidate_b_file = "../gen-class-others/qwen_4_gpt5-4_v4_dc.txt"
    p_value_threshold = 0.05

    try:
        results = evaluate_models(
            answer_file,
            candidate_a_file,
            candidate_b_file,
            p_value_threshold
        )

        print(f"후보군 A 평균 자카드 거리: {results['jaccard_distance']['mean_a']:.4f}")
        print(f"후보군 B 평균 자카드 거리: {results['jaccard_distance']['mean_b']:.4f}")
        print(f"Wilcoxon W-statistic (Jaccard): {results['jaccard_distance']['statistic']}")
        print(f"p-value (Jaccard): {results['jaccard_distance']['p_value']:.4e}, 유의미한 차이 여부: {results['jaccard_distance']['significant']}")

        print(f"후보군 A 평균 F1-score(samples): {results['f1_samples']['mean_a']:.4f}")
        print(f"후보군 B 평균 F1-score(samples): {results['f1_samples']['mean_b']:.4f}")
        print(f"Wilcoxon W-statistic (F1): {results['f1_samples']['statistic']}")
        print(f"p-value (F1): {results['f1_samples']['p_value']:.4e}, 유의미한 차이 여부: {results['f1_samples']['significant']}")

        print(f"후보군 A F1-score(micro): {results['f1_micro']['a']:.4f}")
        print(f"후보군 B F1-score(micro): {results['f1_micro']['b']:.4f}")

    except FileNotFoundError as e:
        print(e)