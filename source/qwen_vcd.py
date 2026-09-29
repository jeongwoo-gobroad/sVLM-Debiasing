import os
import transformers.dynamic_module_utils
transformers.dynamic_module_utils.resolve_trust_remote_code = lambda *args, **kwargs: True

import json
import re
import torch
import torch.nn.functional as F
import numpy as np

import argparse

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# VCD 하이퍼파라미터 (논문 기본값)
ALPHA = 1.0   # contrastive 강도 (Eq.3)
BETA  = 0.1   # adaptive plausibility constraint (Eq.4)
GAMMA = 0.1   # Gaussian forward diffusion noise 비율 (Eq.2)
T     = 500   # diffusion 스텝 수

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=ALPHA)
parser.add_argument("--beta",  type=float, default=BETA)
parser.add_argument("--gamma", type=float, default=GAMMA)
parser.add_argument("--T",     type=int,   default=T)
args = parser.parse_args()

ALPHA    = args.alpha
BETA     = args.beta
GAMMA    = args.gamma
T        = args.T

stat_save_file_name  = "./qwen4b_test_vcd"
pics_folder          = "D:\\training_data\\seomun\\reordered\\test"
tags_file            = "D:\\training_data\\seomun\\result\\refined_tags.txt"
ground_truth_file    = "../seomun_gt_gen/gpt-5.4_gen_test_v4.jsonl"
MODEL_NAME           = "Qwen/Qwen3.5-4B"
MAX_IMG_SIDE         = 540

final_stat_save_file = f"{stat_save_file_name}.txt"

# ==========================================
# 1. 데이터 로드
# ==========================================
with open(tags_file, "r", encoding="utf-8") as f:
    tags = [line.strip() for line in f if line.strip()]

with open(tags_file, "r", encoding="utf-8") as f:
    tags_raw_1 = [" " + line.strip() for line in f if line.strip()]

tags_raw = []
tags_raw.extend(tags)
tags_raw.extend(tags_raw_1)

ground_truth = {}
with open(ground_truth_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ground_truth.update(json.loads(line.strip()))

image_filenames = []
for fname in os.listdir(pics_folder):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')) and fname in ground_truth:
        image_filenames.append(fname)

if not image_filenames:
    exit()

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

eos_id = tokenizer.eos_token_id
if eos_id is None:
    eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

tags_str         = ", ".join(tags)
system_prompt    = "유저의 지시에 따라 정해진 단어로만 답하십시오."
user_prompt_text = f"<tags>\n{tags_str}\n</tags>\n중에 이 사진에서 찾아 볼 수 있는 것을 , 로 구분하여 나열하십시오: "

def make_distorted_image(pil_img: Image.Image, gamma: float, t: int) -> Image.Image:
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    rng = np.random.default_rng()
    for _ in range(t):
        arr = np.sqrt(1.0 - gamma) * arr + np.sqrt(gamma) * rng.standard_normal(arr.shape)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

class VCDLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        dist_inputs: dict,   # v' 에 대한 processor 출력 (input_ids, attention_mask 등)
        dist_prompt_length: int,
        alpha: float,
        beta: float,
    ):
        self.dist_inputs        = dist_inputs
        self.dist_prompt_length = dist_prompt_length
        self.alpha              = alpha
        self.beta               = beta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 현재 생성된 토큰 시퀀스 (prompt 이후 부분)
        generated_ids = input_ids[0, self.dist_prompt_length:].tolist()

        # print(len(generated_ids)) # 디버그용
        dist_input_ids = self.dist_inputs["input_ids"]
        if generated_ids:
            gen_tensor     = torch.tensor([generated_ids], dtype=torch.long, device=dist_input_ids.device)
            dist_input_ids = torch.cat([dist_input_ids, gen_tensor], dim=1)

        dist_attention_mask = torch.ones_like(dist_input_ids)

        with torch.no_grad():
            dist_out    = model(input_ids=dist_input_ids, attention_mask=dist_attention_mask)
            null_logits = dist_out.logits[0, -1, :].float()

        # Eq.3: logit_vcd = (1 + α) * logit_v - α * logit_v'
        vcd_logits = (1.0 + self.alpha) * scores.float() - self.alpha * null_logits

        # Eq.4: Adaptive Plausibility Constraint
        probs_orig = F.softmax(scores.float(), dim=-1)
        max_prob   = probs_orig.max(dim=-1, keepdim=True).values
        mask       = probs_orig < self.beta * max_prob
        vcd_logits[mask] = -float("inf")

        return vcd_logits.to(scores.dtype)

# ==========================================
# 6. 추론 루프
# ==========================================
total_jaccard_distance = 0.0
all_predictions        = {}
valid_filenames        = []

for fname in tqdm(image_filenames, desc="eval"):
    path = os.path.join(pics_folder, fname)

    try:
        orig_img = Image.open(path).convert("RGB")

        w, h = orig_img.size
        if max(w, h) > MAX_IMG_SIDE:
            scale    = MAX_IMG_SIDE / max(w, h)
            orig_img = orig_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": path.replace("\\", "/")}},
                {"type": "text",      "text": user_prompt_text},
            ]}
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cuda")

        prompt_length         = inputs["input_ids"].shape[1]
        logits_processor_list = LogitsProcessorList()

        dist_img  = make_distorted_image(orig_img, GAMMA, T)
        dist_path = "temp_distorted.jpg"
        dist_img.save(dist_path)

        dist_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": dist_path.replace("\\", "/")}},
                {"type": "text",      "text": user_prompt_text},
            ]}
        ]

        dist_inputs = processor.apply_chat_template(
            dist_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cuda")

        if os.path.exists(dist_path):
            os.remove(dist_path)

        vcd_processor = VCDLogitsProcessor(
            dist_inputs=dist_inputs,
            dist_prompt_length=prompt_length,
            alpha=ALPHA,
            beta=BETA,
        )
        logits_processor_list.append(vcd_processor)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                logits_processor=logits_processor_list,
                eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
            )

        new_ids     = gen_ids[0, inputs["input_ids"].shape[1]:].tolist()
        text_output = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(text_output)
        text_output = re.sub(r'[-*]', ',', text_output)

        parsed_tags = [t.strip() for t in text_output.replace(", ", ",").replace("\n", ",").split(',')]
        pred_tags   = set([t.strip() for t in parsed_tags if t.strip()])

        all_predictions[fname] = list(pred_tags)
        valid_filenames.append(fname)

        gt_tags = set(ground_truth.get(fname, []))

        if not pred_tags and not gt_tags:
            j_dist = 0.0
        elif not pred_tags or not gt_tags:
            j_dist = 1.0
        else:
            intersection = len(pred_tags.intersection(gt_tags))
            union        = len(pred_tags.union(gt_tags))
            j_dist       = 1.0 - (intersection / union)

        total_jaccard_distance += j_dist

    except Exception as e:
        print(e)
        continue

if not valid_filenames:
    exit()

average_jaccard_distance = total_jaccard_distance / len(valid_filenames)

os.makedirs(os.path.dirname(os.path.abspath(final_stat_save_file)), exist_ok=True)

with open(final_stat_save_file, "w", encoding="utf-8") as f:
    f.write(f"Average_Jaccard_Distance: {average_jaccard_distance:.4f}\n")
    f.write("-" * 50 + "\n")
    for fname, predicted_tags in all_predictions.items():
        f.write(f"{fname}: {predicted_tags}\n")