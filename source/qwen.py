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
parser.add_argument(
    "--no_prompt", dest="nop", action="store_true", default=False
)
parser.add_argument(
    "--feot_thres", dest="force_eot_threshold", type=float, default=0.2# 0.000058
)
parser.add_argument(
    "--ratio_thres", dest="ratio_threshold", type=float, default=1.000001
)
parser.add_argument(
    "--ratio_times", dest="ratio_times", type=float, default=0.0
)

args = parser.parse_args()

debiasing_mode            = args.deb
constrained_decoding_mode = args.con
force_eot                 = args.eot
no_tags_in_prompt         = args.nop

print(f"Options: {debiasing_mode}, {constrained_decoding_mode}, {force_eot}")

stat_save_file_name       = "qwen_4_gpt5-4_v4"

pics_folder       = "D:\\training_data\\seomun\\result"
tags_file         = "D:\\training_data\\seomun\\result\\refined_tags.txt"
# ground_truth_file = "D:\\training_data\\seomun\\result\\refined_meta_tags.jsonl"
ground_truth_file = "../gpt_gt_gen/gpt5_4_gt_new_v2.jsonl"
MODEL_NAME        = "Qwen/Qwen3.5-4B"
MAX_IMG_SIDE      = 540

ratio_threshold   = args.ratio_threshold
minimum           = args.force_eot_threshold
ratio_times       = args.ratio_times

# 저장 파일명 구성
mode_str = ""
if debiasing_mode: mode_str += "d"
if constrained_decoding_mode: mode_str += "c"
if force_eot: mode_str += "f"
if no_tags_in_prompt: mode_str += "nt"
if not mode_str: mode_str = "plain"

final_stat_save_file = f"{stat_save_file_name}_{mode_str}.txt"

# ==========================================
# 1. 데이터 로드 (태그 및 정답 데이터)
# ==========================================
with open(tags_file, "r", encoding="utf-8") as f:
    tags = [line.strip() for line in f if line.strip()]

with open(tags_file, "r", encoding="utf-8") as f:
    tags_raw_1 = [" " + line.strip() for line in f if line.strip()]
with open(tags_file, "r", encoding="utf-8") as f:
    tags_raw_2 = [", " + line.strip() for line in f if line.strip()]
with open(tags_file, "r", encoding="utf-8") as f:
    tags_raw_3 = [line.strip() + "," for line in f if line.strip()]
with open(tags_file, "r", encoding="utf-8") as f:
    tags_raw_4 = [line.strip() + ", " for line in f if line.strip()]

tags_raw = []
tags_raw.extend(tags)
tags_raw.extend(tags_raw_1)
tags_raw.extend(tags_raw_2)
tags_raw.extend(tags_raw_3)
tags_raw.extend(tags_raw_4)

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
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    # gguf_file="Qwen3.5-9B-UD-Q6_K_XL.gguf",
    dtype="auto", 
    device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

eos_id = tokenizer.eos_token_id
if eos_id is None:
    eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

# ==========================================
# 3. 구분자 토큰 및 프롬프트 설정
# ==========================================
comma_token_ids = set()
for sep in [",", ", ", " ,"]:
    comma_token_ids.update(tokenizer.encode(sep, add_special_tokens=False))

tags_str = ", ".join(tags)
system_prompt = "불필요한 말과 문장을 생성하지 마십시오. 유저의 지시에 철저히 따르시오."
user_prompt_text = f"<tags>\n{tags_str}\n</tags>\n중에 이 사진에서 찾아 볼 수 있는 것을 , 로 구분하여 나열하십시오: "
user_prompt_text_null = f"이 사진에서 찾아 볼 수 있는 것들을 , 로 구분하여 나열하십시오: "

# ==========================================
# 4. Null Image 확률 사전 계산 (Debiasing)
# ==========================================
null_probs_cached = None
if debiasing_mode:
    def make_null_image(w: int, h: int) -> Image.Image:
        rng = np.random.default_rng(seed=42)
        return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))

    null_img = make_null_image(MAX_IMG_SIDE, MAX_IMG_SIDE)
    null_path = "temp_null.jpg"
    null_img.save(null_path)
    
    messages = [
        # {"role": "system", "content": [
        #     {"type": "text", "text": system_prompt}
        # ]},
        {"role": "user", "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"{null_path.replace('\\', '/')}"
                }
            },
            {
                "type": "text",
                "text": user_prompt_text
            },
        ]}
    ]
    
    messages_2 = [
        # {"role": "system", "content": [
        #     {"type": "text", "text": system_prompt}
        # ]},
        {"role": "user", "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"{null_path.replace('\\', '/')}"
                }
            },
            {
                "type": "text",
                "text": user_prompt_text_null
            },
        ]}
    ]

    null_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False
    ).to("cuda")

    null_inputs_2 = processor.apply_chat_template(
        messages_2,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False
    ).to("cuda")

    with torch.no_grad():
        null_outputs = model(**null_inputs)
        null_logits = null_outputs.logits[0, -1, :]
        null_probs_cached = F.softmax(null_logits.float(), dim=-1)

        null_outputs_2 = model(**null_inputs_2)
        null_logits_2 = null_outputs_2.logits[0, -1, :]
        null_probs_cached_2 = F.softmax(null_logits_2.float(), dim=-1)
        
    if os.path.exists(null_path):
        os.remove(null_path)

# class DebiasingLogitsProcessor(LogitsProcessor):
#     def __init__(self, null_probs: torch.Tensor, null_probs_2: torch.Tensor, prompt_length: int, comma_ids: set):
#         self.null_probs = null_probs
#         self.null_probs_no_tags = null_probs_2
#         self.prompt_length = prompt_length
#         self.comma_ids = comma_ids

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         seq_len = input_ids.shape[1]
        
#         is_first_token = input_ids.shape[1] == self.prompt_length
#         is_after_comma = (seq_len > 0) and (input_ids[0, -1].item() in self.comma_ids)
        
#         if is_first_token or is_after_comma:
#             # print("ooo")
#             probs = F.softmax(scores.float(), dim=-1)

#             # debiased_probs = probs + F.softmax(self.null_probs.to(scores.device) / self.null_probs_no_tags.to(scores.device), dim=-1) * ratio_threshold
#             debiased_probs = probs - (self.null_probs.to(scores.device) * ratio_threshold)
#             # debiased_probs = probs / self.null_probs.to(scores.device)

#             # mask = (debiased_probs >= ratio_threshold).float()
#             # debiased_probs = debiased_probs * mask

#             debiased_probs = torch.clamp(debiased_probs, min=1e-15) 
#             scores = torch.log(debiased_probs).to(scores.dtype)
            
#         return scores
class DebiasingLogitsProcessor(LogitsProcessor):
    def __init__(self, null_probs: torch.Tensor, null_probs_2: torch.Tensor, prompt_length: int, comma_ids: set, eos_id: int):
        self.null_probs = null_probs
        self.null_probs_no_tags = null_probs_2
        self.prompt_length = prompt_length
        self.comma_ids = comma_ids
        self.eos_id = eos_id

        # no_tags_in_prompt 모드용: 태그 목록으로 인해 확률이 상승한 토큰 및 그 평균값 사전 계산
        # tag_bias[i] = null_probs[i] - null_probs_2[i] (태그 목록이 없을 때보다 높아진 만큼)
        # tag_bias = self.null_probs - self.null_probs_no_tags          # (vocab,)
        tag_bias = null_logits.float() >= null_logits_2.float()
        # boosted_mask = (tag_bias > 0)                                 # 태그 목록으로 인해 확률이 올라간 토큰
        boosted_mask = tag_bias
        boosted_vals = tag_bias[boosted_mask]
        self.boosted_mask = boosted_mask                              # (vocab,) BoolTensor
        # self.boosted_mean = boosted_vals.mean() if boosted_vals.numel() > 0 else torch.tensor(0.0)
        # mean_penalty[i] = boosted_mean  if  tag_bias[i] > 0  else  0
        self.tag_mean_penalty = torch.zeros_like(self.null_probs)
        self.tag_mean_penalty[boosted_mask] = 0.05 # self.boosted_mean

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_len = input_ids.shape[1]

        eos_score = scores[:, self.eos_id].clone()  # EOS 원본 보존

        is_first_token = input_ids.shape[1] == self.prompt_length
        is_after_comma = (seq_len > 0) and (input_ids[0, -1].item() in self.comma_ids)

        if is_first_token or is_after_comma:
            # print("ooo")
            probs = F.softmax(scores.float(), dim=-1)

            if no_tags_in_prompt:
                # 태그 목록 편향 교정:
                # 태그 목록이 있는 null과 없는 null을 비교해 확률이 상승한 토큰들의
                # 평균 상승량(boosted_mean)을 ratio_threshold 배 스케일해 차감
                # debiased_probs = probs - mean_penalty * ratio_threshold
                # debiased_probs = probs - (self.null_probs.to(scores.device) * ratio_threshold) + (self.tag_mean_penalty.to(scores.device) * 10)
                # debiased_probs = probs - (self.null_probs_no_tags.to(scores.device) * ratio_threshold)
                # print(scores.float().max())
                debiased_probs = scores.float() - null_logits_2.float() * ratio_threshold + self.tag_mean_penalty.to(scores.device) * ratio_times
            else:
                # debiased_probs = probs + F.softmax(self.null_probs.to(scores.device) / self.null_probs_no_tags.to(scores.device), dim=-1) * ratio_threshold
                # debiased_probs = probs - (self.null_probs.to(scores.device) * ratio_threshold)
                # debiased_probs = probs / self.null_probs.to(scores.device)
                # debiased_probs = probs - (self.tag_mean_penalty.to(scores.device) * ratio_threshold)
                # debiased_probs = scores.float() - null_logits.float() * ratio_threshold
                debiased_probs = scores.float() - null_logits.float() * ratio_threshold

            mask = (scores.float() >= null_logits_2.float()).float()
            debiased_probs = debiased_probs * mask

            scores[:, self.eos_id] = eos_score  # EOS는 debiasing 대상에서 제외

            debiased_probs = torch.clamp(debiased_probs, min=1e-15)
            scores = torch.log(debiased_probs).to(scores.dtype)
        # if no_tags_in_prompt:
        #         # 태그 목록 편향 교정:
        #         # 태그 목록이 있는 null과 없는 null을 비교해 확률이 상승한 토큰들의
        #         # 평균 상승량(boosted_mean)을 ratio_threshold 배 스케일해 차감
        #         # debiased_probs = probs - mean_penalty * ratio_threshold
        #         # debiased_probs = probs - (self.null_probs.to(scores.device) * ratio_threshold) + (self.tag_mean_penalty.to(scores.device) * 10)
        #         # debiased_probs = probs - (self.null_probs_no_tags.to(scores.device) * ratio_threshold)
        #         # print(scores.float().max())
        #         debiased_probs = scores.float() - null_logits_2.float() * ratio_threshold + self.tag_mean_penalty.to(scores.device) * ratio_times
        # else:
        #     # debiased_probs = probs + F.softmax(self.null_probs.to(scores.device) / self.null_probs_no_tags.to(scores.device), dim=-1) * ratio_threshold
        #     # debiased_probs = probs - (self.null_probs.to(scores.device) * ratio_threshold)
        #     # debiased_probs = probs / self.null_probs.to(scores.device)
        #     # debiased_probs = probs - (self.tag_mean_penalty.to(scores.device) * ratio_threshold)
        #     # debiased_probs = scores.float() - null_logits.float() * ratio_threshold
        #     debiased_probs = scores.float() - null_logits.float() * ratio_threshold

        # # mask = (scores.float() >= null_logits.float()).float()
        # # debiased_probs = debiased_probs * mask

        # debiased_probs = torch.clamp(debiased_probs, min=1e-15)
        # scores = torch.log(debiased_probs).to(scores.dtype)

        return scores
# ==========================================
# 5. Trie & Constrained Decoding 로직
# ==========================================
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_leaf = False
        self.objects = set()

class TokenTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_ids: list[int], object_name: str):
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_leaf = True # 부분적 일 수 있음에 주의!
        node.objects = {object_name} | node.objects

my_trie = TokenTrie()
for tag in tags_raw:
    tids = tokenizer.encode(tag, add_special_tokens=False)
    my_trie.insert(tids, tag.replace(",", "").strip())

sep_token_ids = set()
# for sep in [",", ", ", " ,", "\n", " \n", " ", "  ", "-", " -", "*", " *"]:
#     sep_token_ids.update(tokenizer.encode(sep, add_special_tokens=False))

for sep in [",", ", ", " "]:
    sep_token_ids.update(tokenizer.encode(sep, add_special_tokens=False))

print(sep_token_ids)

# class TrieLogitsProcessor(LogitsProcessor):
#     def __init__(self, trie, prompt_length, sep_ids, force_eot, eos_id):
#         self.trie = trie
#         self.prompt_length = prompt_length
#         self.sep_ids = sep_ids
#         self.force_eot = force_eot
#         self.eos_id = eos_id
#         self.cursor = None
#         self.completed_tags = []

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         vocab_size = scores.shape[-1]
#         allowed_tokens = set()

#         if self.cursor and input_ids[0, -1].item() in sep_token_ids: # 예를 들어 떡 / 떡볶이 인 경우에, 떡에서 끝난 경우
#             already_exists = any(obj in self.completed_tags for obj in self.cursor.objects)
#             if self.force_eot and already_exists:
#                 allowed_tokens = {self.eos_id}
#             else:
#                 probs = F.softmax(scores.float(), dim=-1)
#                 if torch.max(probs).item() < minimum:
#                     allowed_tokens = {self.eos_id}
#                 else:  
#                     self.completed_tags.extend(self.cursor.objects)
#             self.cursor = None
        
#         if not self.cursor:
#             self.cursor = self.trie.root
#         else:
#             self.cursor = self.cursor.children[input_ids[0, -1].item()]

#         if len(allowed_tokens) == 0:
#             if self.cursor.is_leaf:
#                 allowed_tokens = sep_token_ids | self.cursor.children.keys() # 예를 들어 떡 / 떡볶이 일 수 있으므로
#             else:
#                 allowed_tokens = self.cursor.children.keys()

#         allowed_list = [tid for tid in allowed_tokens if tid < vocab_size]

#         mask = torch.full_like(scores, -float('inf'))
#         mask[:, allowed_list] = 0
        
#         return scores + mask

# class TrieLogitsProcessor(LogitsProcessor):
#     def __init__(self, trie, prompt_length, sep_ids, force_eot, eos_id):
#         self.trie = trie
#         self.prompt_length = prompt_length
#         self.sep_ids = sep_ids
#         self.force_eot = force_eot
#         self.eos_id = eos_id
#         self.cursor = None
#         self.completed_tags = []

#     def _is_after_comma(self, input_ids: torch.LongTensor) -> bool:
#         """지금까지 생성된 토큰을 디코딩한 뒤 rstrip하여 마지막 문자가 ','인지 확인."""
#         generated_ids = input_ids[0, self.prompt_length:].tolist()
#         if len(generated_ids) == 0:
#             self.cursor = None
#             return False
#         text = tokenizer.decode(generated_ids, skip_special_tokens=False)
#         return text.rstrip()[-1:] == ","  # [-1:] 슬라이싱으로 빈 문자열 안전 처리

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         vocab_size = scores.shape[-1]
#         allowed_tokens = set()

#         is_after_comma = self._is_after_comma(input_ids)

#         if self.cursor and is_after_comma:  # 예를 들어 떡 / 떡볶이 인 경우에, 떡에서 끝난 경우
#             already_exists = any(obj in self.completed_tags for obj in self.cursor.objects)
#             if self.force_eot and already_exists:
#                 allowed_tokens = {self.eos_id}
#             else:
#                 probs = F.softmax(scores.float(), dim=-1)
#                 if torch.max(probs).item() < minimum:
#                     allowed_tokens = {self.eos_id}
#                 else:
#                     self.completed_tags.extend(list(self.cursor.objects))
#             self.cursor = None

#         if not self.cursor:
#             self.cursor = self.trie.root
#         elif self.cursor and not constrained_decoding_mode:
#             self.cursor = self.cursor.children[input_ids[0, -1].item()]

#         if len(allowed_tokens) == 0:
#             if not constrained_decoding_mode:
#                 allowed_tokens = set(range(vocab_size))
#             elif self.cursor.is_leaf:
#                 allowed_tokens = {self.eos_id} | sep_token_ids | self.cursor.children.keys()  # 예를 들어 떡 / 떡볶이 일 수 있으므로
#             else:
#                 allowed_tokens = self.cursor.children.keys()

#         allowed_list = [tid for tid in allowed_tokens if tid < vocab_size]

#         mask = torch.full_like(scores, -float('inf'))
#         mask[:, allowed_list] = 0

#         return scores + mask

class TrieLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, prompt_length, sep_ids, force_eot, eos_id):
        self.trie = trie
        self.prompt_length = prompt_length
        self.sep_ids = sep_ids
        self.force_eot = force_eot
        self.eos_id = eos_id
        self.cursor = None
        self.completed_tags = []

    def _is_after_comma(self, input_ids: torch.LongTensor) -> bool:
        generated_ids = input_ids[0, self.prompt_length:].tolist()
        if len(generated_ids) == 0:
            return False
        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        return text.rstrip()[-1:] == ","  # [-1:] 슬라이싱으로 빈 문자열 안전 처리

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        allowed_tokens = set()

        is_after_comma = self._is_after_comma(input_ids)

        # EOT Forcing 블록 — constrained_decoding_mode 와 무관하게 독립 동작
        if is_after_comma:
            # cursor가 있을 때만 completed_tags 갱신 가능 (constrained 모드)
            # cursor가 없을 때(unconstrained)도 force_eot의 확률 임계값 체크는 수행
            if self.cursor is not None and self.force_eot:
                already_exists = any(obj in self.completed_tags for obj in self.cursor.objects)
                if already_exists:
                    # 예를 들어 떡 / 떡볶이 인 경우에, 떡에서 끝난 경우
                    allowed_tokens = {self.eos_id}
                else:
                    probs = F.softmax(scores.float(), dim=-1)
                    if torch.max(probs).item() < minimum:
                        allowed_tokens = {self.eos_id}
                    else:
                        self.completed_tags.extend(list(self.cursor.objects))
            elif self.force_eot:
                # unconstrained 모드에서도 저확률 시 EoS 강제
                probs = F.softmax(scores.float(), dim=-1)
                if torch.max(probs).item() < minimum:
                    allowed_tokens = {self.eos_id}

            self.cursor = None

        # [B] Constrained Decoding 블록 — constrained_decoding_mode 일 때만 Trie 탐색
        if constrained_decoding_mode:
            if self.cursor is None:
                self.cursor = self.trie.root
            else:
                last_token = input_ids[0, -1].item()
                if last_token in self.cursor.children:
                    self.cursor = self.cursor.children[last_token]
                else:
                    self.cursor = self.trie.root

            if len(allowed_tokens) == 0:
                if self.cursor.is_leaf:
                    allowed_tokens = {self.eos_id} | sep_token_ids | self.cursor.children.keys()  # 예를 들어 떡 / 떡볶이 일 수 있으므로
                else:
                    allowed_tokens = self.cursor.children.keys()
        else:
            if len(allowed_tokens) == 0:
                allowed_tokens = set(range(vocab_size))

        allowed_list = [tid for tid in allowed_tokens if tid < vocab_size]

        mask = torch.full_like(scores, -float('inf'))
        mask[:, allowed_list] = 0

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
        messages = [
            # {"role": "system", "content": [
            #     {"type": "text", "text": system_prompt}
            # ]},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{path.replace('\\', '/')}"
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt_text if not no_tags_in_prompt else user_prompt_text_null
                },
            ]}
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
        ).to("cuda")

        prompt_length = inputs["input_ids"].shape[1]

        print(prompt_length)
        
        logits_processor_list = LogitsProcessorList()

        prefix_fn = None
        
        if debiasing_mode and null_probs_cached is not None:
            debiasing_processor = DebiasingLogitsProcessor(null_probs_cached, null_probs_cached_2, prompt_length, comma_token_ids, eos_id)
            logits_processor_list.append(debiasing_processor)

        if constrained_decoding_mode or force_eot:
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
                # temperature=0.0,
                logits_processor=logits_processor_list,
                eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
            )

        new_ids = gen_ids[0, inputs["input_ids"].shape[1]:].tolist()
        text_output = tokenizer.decode(new_ids, skip_special_tokens=True)
        text_output = re.sub(r'[-*]', ',', text_output)

        print(text_output)

        parsed_tags = [t.strip() for t in text_output.replace(", ", ",").replace("\n", ",").split(',')]
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