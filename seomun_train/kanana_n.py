# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SVLM_DATA_ROOT", REPO_ROOT / "datasets")).expanduser().resolve()

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
    "--ratio_thres", dest="ratio_threshold", type=float, default=0.4
)

parser.add_argument('--prompt_variant', choices=['default', 'a', 'b', 'c'], default='default',
                    help='default or paper Prompt 1/2/3 (a/b/c)')
parser.add_argument('--limit', type=int, default=0, help='Smoke test image count; 0 means the full split')
parser.add_argument('--output_dir', type=Path, default=REPO_ROOT / 'results' / 'val')
args = parser.parse_args()
if args.limit < 0:
    parser.error('--limit must be nonnegative')
if getattr(args, 'deb', False) and not args.con:
    parser.error('--debiasing requires --constrained')
args.output_dir.mkdir(parents=True, exist_ok=True)


debiasing_mode            = args.deb
constrained_decoding_mode = args.con

print(f"Options: {debiasing_mode}, {constrained_decoding_mode}")

stat_save_file_name = str(args.output_dir / "kanana/kanana_val")
stat_save_file_name += "" if args.prompt_variant == "default" else f"_prompt[{args.prompt_variant}]"

pics_folder = str(DATA_ROOT / "imgs/val")
tags_file = str(DATA_ROOT / "tag_candidates/refined_tags.txt")
ground_truth_file = str(DATA_ROOT / "tag_gts/human_gt_val.jsonl")
MODEL_NAME = os.environ.get("KANANA_MODEL_PATH", str(REPO_ROOT / "models/kanana"))
MAX_IMG_SIDE      = 720

ratio_threshold   = args.ratio_threshold

# 저장 파일명 구성
mode_str = ""
if debiasing_mode: mode_str += "d"
if constrained_decoding_mode: mode_str += "c"
if not mode_str: mode_str = "plain"

final_stat_save_file = f"{stat_save_file_name}_{mode_str}.txt"

# ==========================================
# 1. 데이터 로드 (태그 및 정답 데이터)
# ==========================================
Path(final_stat_save_file).parent.mkdir(parents=True, exist_ok=True)

with open(tags_file, "r", encoding="utf-8") as f:
    tags = [line.strip() for line in f if line.strip()]

with open(tags_file, "r", encoding="utf-8") as f:
    tags_space_start = [" " + line.strip() for line in f if line.strip()]

ground_truth = {}
with open(ground_truth_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ground_truth.update(json.loads(line.strip()))

image_filenames = []
for fname in os.listdir(pics_folder):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')) and fname in ground_truth:
        image_filenames.append(fname)

if set(image_filenames) != set(ground_truth):
    raise ValueError('Image filenames and ground-truth keys do not match')
if args.limit:
    image_filenames = image_filenames[:args.limit]
if not image_filenames:
    raise ValueError('No images found')

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
print(eos_id)
if eos_id is None:
    print("No EOS ID Found!")
    eos_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]

sep_token_ids = set()

for sep in [",", "\n"]:
    print(tokenizer.encode(sep, add_special_tokens=False))
    sep_token_ids.add(tokenizer.encode(sep, add_special_tokens=False)[0])

# ==========================================
# 3. 구분자 토큰 및 프롬프트 설정
# ==========================================
tags_str = "\n".join(tags)
system_prompt = {
    'default': "정해진 단어로만 답하십시오.\n\n",
    'a': "",
    'b': "'이 이미지에서 볼 수 있는 것들은'과 같은 불필요한 응답을 생성하지 말고, 유저의 지시에 따르십시오.",
    'c': "불필요한 말과 문장을 생성하지 마십시오. 정해진 단어로만 답하십시오.\n\n",
}[args.prompt_variant]
user_prompt_text = f"""<tags>\n{tags_str}\n</tags>\n\n위 <tags> 내 단어 중, 이 사진에서 볼 수 있는 핵심 객체들을 , 로 구분하여 나열하십시오."""

tags_space_str = "\n".join(tags_space_start)
user_prompt_text_space = f"<tags>\n{tags_space_str}\n</tags>\n\n위 <tags> 내 단어 중, 이 사진에서 볼 수 있는 핵심 객체들을 , 로 구분하여 나열하십시오."

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

space_trie = TokenTrie()
for tag in tags_space_start:
    tids = tokenizer.encode(tag, add_special_tokens=False)
    space_trie.insert(tids, tag.replace(",", "").strip())

stt_trie = TokenTrie()
for tag in tags:
    tids = tokenizer.encode(tag, add_special_tokens=False)
    stt_trie.insert(tids, tag.replace(",", "").strip())

class TrieLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_length, eos_id, sep_ids, normal_trie, space_trie, normal_logit, space_logit, debias_mode):
        self.prompt_length = prompt_length
        self.eos_id        = eos_id
        self.sep_ids       = sep_ids
        self.normal_trie   = normal_trie
        self.space_trie    = space_trie

        self.null_normal_logit  = normal_logit
        self.null_space_logit   = space_logit

        self.cursor         = None
        self.completed_tags = []

        self.is_debias_mode = debias_mode

        self.comma_id   = tokenizer.encode(",", add_special_tokens=False)[0]

    def _debias(self, allowed_tokens, scores, is_start):
        no_debias_ids    = [self.eos_id]
        no_debias_vals   = None
        candidate_tokens = torch.tensor(allowed_tokens, dtype=torch.long, device=scores.device)

        # print("debiasing")

        def minmaxnorm_candidates_prev(logits):
            nonlocal no_debias_ids
            nonlocal no_debias_vals
            nonlocal candidate_tokens
            vals = logits[-1, candidate_tokens]
            v_min, v_max = vals.min(), vals.max()
            normed = (vals - v_min) / (v_max - v_min + 1e-15)
            out = torch.full_like(logits[0], float('-inf'))
            out[candidate_tokens] = normed
            no_debias_vals = out[no_debias_ids]
            return out.unsqueeze(0)

        def minmaxnorm_candidates_post(logits):
            nonlocal candidate_tokens
            vals = logits[-1, candidate_tokens]
            v_min, v_max = vals.min(), vals.max()
            normed = (vals - v_min) / (v_max - v_min + 1e-15)
            out = torch.zeros_like(logits[0])
            out[candidate_tokens] = normed
            return out.unsqueeze(0)

        prev = minmaxnorm_candidates_prev(scores.float())
        post = minmaxnorm_candidates_post(self.null_normal_logit.float().to(scores.device).unsqueeze(0) if is_start else \
                                          self.null_space_logit.float().to(scores.device).unsqueeze(0)) 
        debiased_probs = prev - post * ratio_threshold
        debiased_probs[:, no_debias_ids] = no_debias_vals

        return debiased_probs
    
    def _no_debias(self, allowed_tokens, scores):
        mask = torch.full_like(scores, float('-inf'))
        mask[:, allowed_tokens] = 0

        return scores + mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        allowed_tokens = set()
        is_start       = False
        do_debias      = False
        
        if self.cursor is None: # 오직 생성 시작시에만 None 일 수 있음
            self.cursor = self.normal_trie.root
            is_start    = True
        else:
            last_token = input_ids[0, -1].item()
            if last_token in self.cursor.children:
                self.cursor = self.cursor.children[last_token]
                # print(tokenizer.decode(last_token))
            else: # 한 태그가 생성 종료되었다고 판단, 이제 공백 포함한 시작 토큰으로만 생성 가능
                already_exists = any(obj in self.completed_tags for obj in self.cursor.objects)
                if already_exists:
                    allowed_tokens = {self.eos_id}
                else:
                    self.completed_tags.extend(list(self.cursor.objects))
                # sep token 자유 생성의 처리 핵심
                self.cursor = self.space_trie.root
                do_debias = True

        if len(allowed_tokens) == 0:
            if self.cursor.is_leaf:
                allowed_tokens = {self.eos_id} | {self.comma_id}
                if len(self.cursor.children.keys()) != 0:  # 예를 들어 떡 / 떡볶이 일 수 있으므로
                    # print("wow@!")
                    allowed_tokens |= self.cursor.children.keys()
                is_start = True
            else:
                allowed_tokens |= self.cursor.children.keys()
        
        if do_debias or is_start:
            allowed_tokens |= {self.eos_id}

        allowed_tokens = list(allowed_tokens)

        scores = self._debias(allowed_tokens, scores, is_start) if (self.is_debias_mode and (is_start or do_debias)) else self._no_debias(allowed_tokens, scores)

        # print(tokenizer.decode(np.argmax(scores.to('cpu') + mask.to('cpu'), axis=-1), skip_special_tokens=True))

        return scores

# ==========================================
# 4. Null Image 확률 사전 계산 (Debiasing)
# ==========================================
if debiasing_mode or constrained_decoding_mode:
    null_img = Image.open(DATA_ROOT / "noise/reference_noise.jpg").convert("RGB")
    null_sample = {
        "image": [null_img],
        "conv": [
            {"role": "user", "content": system_prompt},
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

    null_sample_2 = {
        "image": [null_img],
        "conv": [
            {"role": "user", "content": system_prompt},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": user_prompt_text_space},
        ]
    }

    null_inputs_2 = processor.batch_encode_collate(
        [null_sample_2], padding_side="left", add_generation_prompt=True, max_length=8192
    )
    null_inputs_2 = {
        k: (v.to(model.device).to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v.to(model.device) if isinstance(v, torch.Tensor) else v)
        for k, v in null_inputs_2.items()
    }    

    with torch.no_grad():
        null_outputs = model(**null_inputs)
        null_logits = null_outputs.logits[0, -1, :]

        null_outputs_2 = model(**null_inputs_2)
        null_logits_2 = null_outputs_2.logits[0, -1, :]

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
                {"role": "user", "content": system_prompt},
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
        
        if constrained_decoding_mode:
            trie_processor = TrieLogitsProcessor(
                prompt_length, 
                eos_id,
                sep_token_ids,
                stt_trie,
                space_trie,
                null_logits,
                null_logits_2,
                debiasing_mode
            )
            logits_processor_list.append(trie_processor)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0,
                logits_processor=logits_processor_list,
            )

        text_output = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        text_output = re.sub(r'[-*]', ',', text_output)

        print(text_output)

        parsed_tags = [t.strip() for t in text_output.replace(",", " ").replace("\n", " ").split(' ')]
        pred_tags = set([t.strip() for t in parsed_tags if t.strip()])

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
        raise RuntimeError(f"Inference failed for {fname}") from e

if not valid_filenames:
    raise RuntimeError("No predictions were produced")

average_jaccard_distance = total_jaccard_distance / len(valid_filenames)

with open(final_stat_save_file, "w", encoding="utf-8") as f:
    f.write(f"Average_Jaccard_Distance: {average_jaccard_distance:.4f}\n")
    f.write("-" * 50 + "\n")
    
    for fname, predicted_tags in all_predictions.items():
        f.write(f"{fname}: {predicted_tags}\n")
run_metadata = {"arguments": vars(args), "model": MODEL_NAME, "ground_truth_sha256": __import__("hashlib").sha256(Path(ground_truth_file).read_bytes()).hexdigest(), "noise_sha256": __import__("hashlib").sha256((DATA_ROOT / "noise/reference_noise.jpg").read_bytes()).hexdigest(), "script_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(), "python": sys.executable, "samples": len(valid_filenames)}
Path(final_stat_save_file + ".metadata.json").write_text(json.dumps(run_metadata, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
