# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
# Derived from seomun_test/qwen_n.py; decoder and input construction preserved.
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
import math
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, AutoTokenizer, LogitsProcessor, LogitsProcessorList
parser = argparse.ArgumentParser()
parser.add_argument('--debiasing', dest='deb', action='store_true')
parser.add_argument('--constrained', dest='con', action='store_true')
parser.add_argument('--ratio_thres', dest='ratio_threshold', type=float, default=0.4)
parser.add_argument('--model_size', dest='msize', type=str, choices=['2b', '4b'], required=True)
parser.add_argument('--prompt_variant', choices=['default', 'a', 'b', 'c'], default='default', help='default or paper Prompt 1/2/3 (a/b/c)')
parser.add_argument('--limit', type=int, default=0, help='Smoke test image count; 0 means the full split')
parser.add_argument('--output_dir', type=Path, default=REPO_ROOT / 'results' / 'tag_outputs')
parser.add_argument('--revision', help='Override the recorded local-cache model revision')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--split', choices=['all', 'val', 'test'], default='all')
args = parser.parse_args()
if args.limit < 0:
    parser.error('--limit must be nonnegative')
if getattr(args, 'deb', False) and (not args.con):
    parser.error('--debiasing requires --constrained')
args.output_dir.mkdir(parents=True, exist_ok=True)
debiasing_mode = args.deb
constrained_decoding_mode = args.con
ratio_threshold = args.ratio_threshold
model_size = args.msize
pass
stat_save_file_name = str(args.output_dir / f'qwen3.5_{model_size}/qwen3.5_{model_size}_test')
stat_save_file_name += '' if args.prompt_variant == 'default' else f'_prompt[{args.prompt_variant}]'
pics_folder = str(DATA_ROOT / 'imgs/test')
tags_file = str(DATA_ROOT / 'tag_candidates/refined_tags.txt')
ground_truth_file = str(DATA_ROOT / 'tag_gts/human_gt_test.jsonl')
MODEL_NAME = f'Qwen/Qwen3.5-{str(model_size).upper()}'
MODEL_REVISION = args.revision or json.loads((REPO_ROOT / 'models/revisions.json').read_text(encoding='utf-8'))[MODEL_NAME]['revision']
MAX_IMG_SIDE = 720
mode_str = ''
if debiasing_mode:
    mode_str += 'd'
if constrained_decoding_mode:
    mode_str += 'c'
if not mode_str:
    mode_str = 'plain'
final_stat_save_file = f'{stat_save_file_name}_{mode_str}.txt'
Path(final_stat_save_file).parent.mkdir(parents=True, exist_ok=True)
with open(tags_file, 'r', encoding='utf-8') as f:
    tags = [line.strip() for line in f if line.strip()]
with open(tags_file, 'r', encoding='utf-8') as f:
    tags_space_start = [' ' + line.strip() for line in f if line.strip()]
model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, dtype='auto', device_map='auto')
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
eos_id = tokenizer.eos_token_id
pass
if eos_id is None:
    eos_id = tokenizer.encode('<|im_end|>', add_special_tokens=False)[0]
sep_token_ids = set()
for sep in [',', '\n']:
    pass
    sep_token_ids.add(tokenizer.encode(sep, add_special_tokens=False)[0])
tags_str = '\n'.join(tags)
system_prompt = {'default': '정해진 단어로만 답하십시오.\n\n', 'a': '', 'b': "'이 이미지에서 볼 수 있는 것들은'과 같은 불필요한 응답을 생성하지 말고, 유저의 지시에 따르십시오.", 'c': '불필요한 말과 문장을 생성하지 마십시오. 정해진 단어로만 답하십시오.\n\n'}[args.prompt_variant]
user_prompt_text = f'<tags>\n{tags_str}\n</tags>\n\n위 <tags> 내 단어 중, 이 사진에서 볼 수 있는 핵심 객체들을 , 로 구분하여 나열하십시오.'
tags_space_str = '\n'.join(tags_space_start)
user_prompt_text_space = f'<tags>\n{tags_space_str}\n</tags>\n\n위 <tags> 내 단어 중, 이 사진에서 볼 수 있는 핵심 객체들을 , 로 구분하여 나열하십시오.'

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
        node.is_leaf = True
        node.objects = {object_name} | node.objects
space_trie = TokenTrie()
for tag in tags_space_start:
    tids = tokenizer.encode(tag, add_special_tokens=False)
    space_trie.insert(tids, tag.replace(',', '').strip())
stt_trie = TokenTrie()
for tag in tags:
    tids = tokenizer.encode(tag, add_special_tokens=False)
    stt_trie.insert(tids, tag.replace(',', '').strip())

class TrieLogitsProcessor(LogitsProcessor):

    def __init__(self, prompt_length, eos_id, sep_ids, normal_trie, space_trie, normal_logit, space_logit, debias_mode):
        self.prompt_length = prompt_length
        self.eos_id = eos_id
        self.sep_ids = sep_ids
        self.normal_trie = normal_trie
        self.space_trie = space_trie
        self.null_normal_logit = normal_logit
        self.null_space_logit = space_logit
        self.cursor = None
        self.completed_tags = []
        self.is_debias_mode = debias_mode
        self.comma_id = tokenizer.encode(',', add_special_tokens=False)[0]

    def _debias(self, allowed_tokens, scores, is_start):
        no_debias_ids = [self.eos_id]
        no_debias_vals = None
        candidate_tokens = torch.tensor(allowed_tokens, dtype=torch.long, device=scores.device)

        def minmaxnorm_candidates_prev(logits):
            nonlocal no_debias_ids
            nonlocal no_debias_vals
            nonlocal candidate_tokens
            vals = logits[-1, candidate_tokens]
            v_min, v_max = (vals.min(), vals.max())
            normed = (vals - v_min) / (v_max - v_min + 1e-15)
            out = torch.full_like(logits[0], float('-inf'))
            out[candidate_tokens] = normed
            no_debias_vals = out[no_debias_ids]
            return out.unsqueeze(0)

        def minmaxnorm_candidates_post(logits):
            nonlocal candidate_tokens
            vals = logits[-1, candidate_tokens]
            v_min, v_max = (vals.min(), vals.max())
            normed = (vals - v_min) / (v_max - v_min + 1e-15)
            out = torch.zeros_like(logits[0])
            out[candidate_tokens] = normed
            return out.unsqueeze(0)
        prev = minmaxnorm_candidates_prev(scores.float())
        post = minmaxnorm_candidates_post(self.null_normal_logit.float().to(scores.device).unsqueeze(0) if is_start else self.null_space_logit.float().to(scores.device).unsqueeze(0))
        debiased_probs = prev - post * ratio_threshold
        debiased_probs[:, no_debias_ids] = no_debias_vals
        return debiased_probs

    def _no_debias(self, allowed_tokens, scores):
        mask = torch.full_like(scores, float('-inf'))
        mask[:, allowed_tokens] = 0
        return scores + mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        allowed_tokens = set()
        is_start = False
        do_debias = False
        if self.cursor is None:
            self.cursor = self.normal_trie.root
            is_start = True
        else:
            last_token = input_ids[0, -1].item()
            if last_token in self.cursor.children:
                self.cursor = self.cursor.children[last_token]
            else:
                already_exists = any((obj in self.completed_tags for obj in self.cursor.objects))
                if already_exists:
                    allowed_tokens = {self.eos_id}
                else:
                    self.completed_tags.extend(list(self.cursor.objects))
                self.cursor = self.space_trie.root
                do_debias = True
        if len(allowed_tokens) == 0:
            if self.cursor.is_leaf:
                allowed_tokens = {self.eos_id} | {self.comma_id}
                if len(self.cursor.children.keys()) != 0:
                    allowed_tokens |= self.cursor.children.keys()
                is_start = True
            else:
                allowed_tokens |= self.cursor.children.keys()
        if do_debias or is_start:
            allowed_tokens |= {self.eos_id}
        allowed_tokens = list(allowed_tokens)
        scores = self._debias(allowed_tokens, scores, is_start) if self.is_debias_mode and (is_start or do_debias) else self._no_debias(allowed_tokens, scores)
        return scores
if constrained_decoding_mode or debiasing_mode:
    null_img = Image.open(DATA_ROOT / 'noise/reference_noise.jpg').convert('RGB')
    null_path = str(DATA_ROOT / 'noise/reference_noise.jpg')
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': system_prompt}, {'type': 'image_url', 'image_url': {'url': f"{null_path.replace('\\', '/')}"}}, {'type': 'text', 'text': user_prompt_text}]}]
    messages_2 = [{'role': 'user', 'content': [{'type': 'text', 'text': system_prompt}, {'type': 'image_url', 'image_url': {'url': f"{null_path.replace('\\', '/')}"}}, {'type': 'text', 'text': user_prompt_text_space}]}]
    null_inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', enable_thinking=False).to('cuda')
    null_inputs_2 = processor.apply_chat_template(messages_2, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', enable_thinking=False).to('cuda')
    with torch.no_grad():
        null_outputs = model(**null_inputs)
        null_logits = null_outputs.logits[0, -1, :]
        null_outputs_2 = model(**null_inputs_2)
        null_logits_2 = null_outputs_2.logits[0, -1, :]

from tag_generation_common import collect

def build_inputs(path):
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': system_prompt}, {'type': 'image_url', 'image_url': {'url': f"{path.replace('\\', '/')}"}}, {'type': 'text', 'text': user_prompt_text}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', enable_thinking=False).to('cuda')
    return inputs

def make_processors(inputs):
    if not constrained_decoding_mode:
        return []
    return [TrieLogitsProcessor(inputs['input_ids'].shape[1], eos_id, sep_token_ids,
                               stt_trie, space_trie, null_logits, null_logits_2, debiasing_mode)]

def generate(inputs, processors, stopping, max_tokens):
    return model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, logits_processor=processors, eos_token_id=eos_id, stopping_criteria=stopping)

collect(REPO_ROOT, 'qwen3.5_4b', mode_str, ratio_threshold, tokenizer, eos_id, tags,
        build_inputs, generate, make_processors, args.output_dir, __file__,
        limit=args.limit, resume=args.resume, split=args.split)
