import os
import json
import argparse

import torch
import KELIP.kelip as kelip
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pics_folder",      type=str,   default="D:\\training_data\\seomun\\reordered_anon\\test")
parser.add_argument("--tags_file",        type=str,   default="D:\\training_data\\seomun\\refined_tags.txt")
parser.add_argument("--ground_truth_file",type=str,   default="../seomun_gt_gen/gpt-5.4_gen_test_v7.jsonl")
parser.add_argument("--threshold",        type=float, required=True)
parser.add_argument("--out_file",         type=str, default="./kelip_test.txt")
parser.add_argument("--text_template",    type=str, default="이 사진에는 {label}이(가) 있습니다.")
parser.add_argument("--model_arch",       type=str, default="ViT-B/32")
args = parser.parse_args()

with open(args.tags_file, "r", encoding="utf-8") as f:
    tags = [line.strip() for line in f if line.strip()]

image_filenames = sorted([
    fname for fname in os.listdir(args.pics_folder)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))
])

if not image_filenames:
    exit()

ground_truth = {}
with open(args.ground_truth_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ground_truth.update(json.loads(line.strip()))

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess_img, tokenizer = kelip.build_model(args.model_arch)
model = model.to(device)
model.eval()

text_sentences = [args.text_template.replace("{label}", tag) for tag in tags]
text_tokens = tokenizer.encode(text_sentences).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

text_features = text_features.cpu()  

total_jaccard_distance = 0.0
all_predictions = {}
valid_filenames = []

with torch.no_grad():
    for fname in tqdm(image_filenames, desc="Inferring"):
        path = os.path.join(args.pics_folder, fname)
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess_img(img).unsqueeze(0).to(device)
            feat = model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = torch.matmul(feat.cpu(), text_features.T).squeeze(0)  # (num_tags,)

            pred_tags = set(tag for tag, selected in zip(tags, (sim >= args.threshold).tolist()) if selected)
            gt_tags   = set(ground_truth.get(fname, []))

            if not pred_tags and not gt_tags:
                j_dist = 0.0
            elif not pred_tags or not gt_tags:
                j_dist = 1.0
            else:
                intersection = len(pred_tags.intersection(gt_tags))
                union        = len(pred_tags.union(gt_tags))
                j_dist       = 1.0 - (intersection / union)

            total_jaccard_distance += j_dist
            all_predictions[fname]  = list(pred_tags)
            valid_filenames.append(fname)

        except Exception as e:
            print(f"[WARN] {fname} 처리 실패: {e}")

if not valid_filenames:
    print("처리된 이미지가 없습니다.")
    exit()

average_jaccard_distance = total_jaccard_distance / len(valid_filenames)

os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

with open(args.out_file, "w", encoding="utf-8") as f:
    f.write(f"Average_Jaccard_Distance: {average_jaccard_distance:.4f}\n")
    f.write("-" * 50 + "\n")
    for fname, predicted_tags in all_predictions.items():
        f.write(f"{fname}: {predicted_tags}\n")