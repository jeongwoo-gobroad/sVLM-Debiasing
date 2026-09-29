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
from transformers import AutoModelForVision2Seq, AutoProcessor, LogitsProcessor, LogitsProcessorList

# ==========================================
# 설정 변수 및 실험 모드 ON/OFF
# ==========================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--debiasing", dest="deb", action="store_true"
)
parser.add_argument(
    "--constrained", dest="con", action="store_true"
)
parser.add_argument(
    "--force_eot", dest="eot", action="store_true"
)

args = parser.parse_args()

debiasing_mode            = args.deb
constrained_decoding_mode = args.con
force_eot                 = args.eot

print(f"Options: {debiasing_mode}, {constrained_decoding_mode}, {force_eot}")

stat_save_file_name       = "kanana_tag_gpt"

pics_folder       = "D:\\training_data\\seomun\\result"
tags_file         = "D:\\training_data\\seomun\\result\\refined_tags.txt"
# ground_truth_file = "D:\\training_data\\seomun\\result\\refined_meta_tags.jsonl"
ground_truth_file = "../gpt_gt_gen/gpt4_1_mini_gt.jsonl"
MODEL_NAME        = "kakaocorp/kanana-1.5-v-3b-instruct"
MAX_IMG_SIDE      = 540

ratio_threshold   = 2.5
minimum           = 0.15

# 저장 파일명 구성
mode_str = ""
if debiasing_mode: mode_str += "d"
if constrained_decoding_mode: mode_str += "c"
if force_eot: mode_str += "f"
if not mode_str: mode_str = "plain"

final_stat_save_file = f"{stat_save_file_name}_{mode_str}.txt"

# ==========================================
# 1. 데이터 로드 (태그 및 정답 데이터)
# ==========================================
with open(tags_file, "r", encoding="utf-8") as f:
    tags = [line.strip() for line in f if line.strip()]

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

# ==========================================
# 2. 모델 및 토크나이저 초기화
# ==========================================
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = processor.tokenizer

eos_id = tokenizer.eos_token_id
if eos_id is None:
    eos_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]

# ==========================================
# 3. 구분자 토큰 및 프롬프트 설정
# ==========================================
comma_token_ids = set()
for sep in [",", ", ", " ,"]:
    comma_token_ids.update(tokenizer.encode(sep, add_special_tokens=False))

tags_str = ", ".join(tags)
system_prompt = "사진에 대한 설명을 하지 마십시오. 불필요한 말과 문장을 생성하지 마십시오. 유저의 지시에 철저히 따르시오."
user_prompt_text = f"<tags>\n{tags_str}\n</tags>\n중에 이 사진에서 찾아 볼 수 있는 것을 , 로 구분하여 나열하십시오: "

# ==========================================
# 4. Null Image 확률 사전 계산 (Debiasing)
# ==========================================
null_probs_cached = None
if debiasing_mode:
    def make_null_image(w: int, h: int) -> Image.Image:
        rng = np.random.default_rng(seed=42)
        return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))

    null_img = make_null_image(MAX_IMG_SIDE, MAX_IMG_SIDE)
    null_sample = {
        "image": [null_img],
        "conv": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": user_prompt_text},
        ]
    }

    null_inputs = processor.batch_encode_collate(
        [null_sample], padding_side="left", add_generation_prompt=True, max_length=8192
    )
    null_inputs = {
        k: (v.to(model.device).to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v.to(model.device) if isinstance(v, torch.Tensor) else v)
        for k, v in null_inputs.items()
    }

    with torch.no_grad():
        null_outputs = model(**null_inputs)
        null_logits = null_outputs.logits[0, -1, :]
        null_probs_cached = F.softmax(null_logits.float(), dim=-1)

class DebiasingLogitsProcessor(LogitsProcessor):
    def __init__(self, null_probs: torch.Tensor, prompt_length: int, comma_ids: set):
        self.null_probs = null_probs
        self.prompt_length = prompt_length
        self.comma_ids = comma_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_len = input_ids.shape[1]
        
        is_first_token = seq_len == 0
        is_after_comma = (seq_len > 0) and (input_ids[0, -1].item() in self.comma_ids)
        
        if is_first_token or is_after_comma:
            probs = F.softmax(scores.float(), dim=-1)

            # debiased_probs = probs - self.null_probs.to(scores.device)

            debiased_probs = probs / (self.null_probs.to(scores.device) + 1e-5)

            mask = (debiased_probs >= ratio_threshold).float()
            debiased_probs = debiased_probs * mask

            debiased_probs = torch.clamp(debiased_probs, min=1e-10) 
            scores = torch.log(debiased_probs).to(scores.dtype)
            
        return scores

# ==========================================
# 5. Trie & Constrained Decoding 로직
# ==========================================
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_leaf = False
        self.objects = []

class TokenTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_ids: list[int], object_name: str):
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_leaf = True
        node.objects.append(object_name)

my_trie = TokenTrie()
for tag in tags:
    tids = tokenizer.encode(tag, add_special_tokens=False)
    my_trie.insert(tids, tag)

sep_token_ids = set()
for sep in [",", ", ", " ,", "\n", " \n", " ", "  ", "-", " -", "*", " *"]:
    sep_token_ids.update(tokenizer.encode(sep, add_special_tokens=False))

print(sep_token_ids)

class TrieLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, prompt_length, sep_ids, force_eot, eos_id):
        self.trie = trie
        self.prompt_length = prompt_length
        self.sep_ids = sep_ids
        self.force_eot = force_eot
        self.eos_id = eos_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        assistant_ids = input_ids
        
        node = self.trie.root
        is_free_mode = False
        completed_tags = []

        for tid in assistant_ids:
            if is_free_mode:
                if tid in self.sep_ids:
                    is_free_mode = False
                    node = self.trie.root
            else:
                if tid in node.children:
                    node = node.children[tid]
                else:
                    if node.is_leaf:
                        completed_tags.extend(node.objects)
                        is_free_mode = True
                        if tid in self.sep_ids:
                            is_free_mode = False
                            node = self.trie.root
                    else:
                        is_free_mode = True

        vocab_size = scores.shape[-1]
        # if len(assistant_ids) > 0:
        #     print(f"Current assistant token: {tokenizer.decode(assistant_ids[-1])}")

        if is_free_mode:
            probs = F.softmax(scores.float(), dim=-1)
            if torch.max(probs).item() < minimum:
                allowed_tokens = {self.eos_id}
            else:  
                allowed_tokens = set(range(vocab_size))
        else:
            if node.is_leaf:
                already_exists = any(obj in completed_tags for obj in node.objects)
                if self.force_eot and already_exists:
                    allowed_tokens = {self.eos_id}
                else:
                    probs = F.softmax(scores.float(), dim=-1)
                    if torch.max(probs).item() < minimum:
                        allowed_tokens = {self.eos_id}
                    else:  
                        allowed_tokens = set(range(vocab_size))
            else:
                allowed_tokens = set(node.children.keys())
                
                if not allowed_tokens:
                    allowed_tokens = {self.eos_id}

        allowed_list = [tid for tid in allowed_tokens if tid < vocab_size]

        # 마스킹 적용 (허용되지 않은 토큰의 확률을 -무한대로 설정)
        mask = torch.full_like(scores, -float('inf'))
        mask[0, allowed_list] = 0
        
        return scores + mask

# ==========================================
# 6. 추론 및 Jaccard Distance 계산
# ==========================================
total_jaccard_distance = 0.0
all_predictions = {}
valid_filenames = []

for fname in tqdm(image_filenames, desc="Evaluating Images"):
    path = os.path.join(pics_folder, fname)
    
    try:
        pil_img = Image.open(path).convert("RGB")
        w, h = pil_img.size
        if max(w, h) > MAX_IMG_SIDE:
            scale = MAX_IMG_SIDE / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        
        sample = {
            "image": [pil_img],
            "conv": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "<image>"},
                {"role": "user", "content": user_prompt_text},
            ]
        }
        
        inputs = processor.batch_encode_collate(
            [sample], padding_side="left", add_generation_prompt=True, max_length=8192
        )
        
        inputs = {
            k: (v.to(model.device).to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v.to(model.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        prompt_length = inputs["input_ids"][0].shape[0]

        print(prompt_length)
        
        logits_processor_list = LogitsProcessorList()
        if debiasing_mode and null_probs_cached is not None:
            debiasing_processor = DebiasingLogitsProcessor(null_probs_cached, prompt_length, comma_token_ids)
            logits_processor_list.append(debiasing_processor)

        prefix_fn = None
        if constrained_decoding_mode:
            trie_processor = TrieLogitsProcessor(
                my_trie, 
                prompt_length, 
                sep_token_ids, 
                force_eot, 
                eos_id, 
            )
            logits_processor_list.append(trie_processor)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
                logits_processor=logits_processor_list,
            )

        text_output = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        text_output = re.sub(r'[-*]', ',', text_output)

        print(text_output)

        parsed_tags = [t.strip() for t in text_output.split(',')]
        pred_tags = set([t for t in parsed_tags if t in tags])

        all_predictions[fname] = list(pred_tags)
        valid_filenames.append(fname)
        
        gt_tags = set(ground_truth.get(fname, []))

        print(f"{fname}: GT | [{gt_tags}], PRED | [{pred_tags}]")
        
        if not pred_tags and not gt_tags:
            j_dist = 0.0
        elif not pred_tags or not gt_tags:
            j_dist = 1.0
        else:
            intersection = len(pred_tags.intersection(gt_tags))
            union = len(pred_tags.union(gt_tags))
            j_dist = 1.0 - (intersection / union)
            
        total_jaccard_distance += j_dist
        
    except Exception as e:
        print(e)
        continue

if not valid_filenames:
    exit()

average_jaccard_distance = total_jaccard_distance / len(valid_filenames)

with open(final_stat_save_file, "w", encoding="utf-8") as f:
    f.write(f"Average_Jaccard_Distance: {average_jaccard_distance:.4f}\n")
    f.write("-" * 50 + "\n")
    
    for fname, predicted_tags in all_predictions.items():
        f.write(f"{fname}: {predicted_tags}\n")