# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Trie-only processor copied from qwen_n.py for the VCD control."""
import torch
from transformers import LogitsProcessor

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

class TrieLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_length, eos_id, sep_ids, normal_trie, space_trie, tokenizer):
        self.prompt_length = prompt_length
        self.eos_id        = eos_id
        self.sep_ids       = sep_ids
        self.normal_trie   = normal_trie
        self.space_trie    = space_trie


        self.cursor         = None
        self.completed_tags = []


        self.comma_id   = tokenizer.encode(",", add_special_tokens=False)[0]

    
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

        scores = self._no_debias(allowed_tokens, scores)

        # print(tokenizer.decode(np.argmax(scores.to('cpu') + mask.to('cpu'), axis=-1), skip_special_tokens=True))

        return scores
