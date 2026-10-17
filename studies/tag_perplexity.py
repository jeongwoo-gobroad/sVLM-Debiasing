# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Length-normalized likelihood of a tag appended to a generation prefix."""
import math

import torch


def target_log_probs(logits, target_ids):
    """Logits are the last len(target)+1 positions of prefix+target."""
    ids = torch.as_tensor(target_ids, device=logits.device, dtype=torch.long)
    if logits.ndim != 3 or logits.shape[:2] != (1, len(ids)+1) or not len(ids):
        raise ValueError('Expected one causal logit before each target, plus the unused last logit')
    # 첫 태그 토큰은 그 직전인 프롬프트 마지막 위치에서 예측한다.
    scores = logits[0, :-1].float().log_softmax(-1)
    values = scores.gather(1, ids[:, None]).squeeze(1)
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite target likelihood')
    return values.cpu().double().tolist()


def summarize_log_probs(values):
    if not values or any(not math.isfinite(v) or v > 1e-6 for v in values):
        raise ValueError('Expected finite nonpositive token log probabilities')
    nll = -math.fsum(values)
    mean = nll/len(values)
    return {'token_count': len(values), 'sum_nll': nll, 'mean_nll': mean, 'perplexity': math.exp(mean)}


def append_text(inputs, target_ids):
    ids = torch.as_tensor([target_ids], device=inputs['input_ids'].device, dtype=torch.long)
    joined = dict(inputs)
    joined['input_ids'] = torch.cat([inputs['input_ids'], ids], dim=1)
    mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
    joined['attention_mask'] = torch.cat([mask, torch.ones_like(ids)], dim=1)
    for key in ('mm_token_type_ids', 'token_type_ids'):
        if key in inputs:
            joined[key] = torch.cat([inputs[key], torch.zeros_like(ids)], dim=1)
    if 'position_ids' in inputs:
        raise ValueError('Explicit processor positions require a model-specific extension')
    return joined


class TagScorer:
    def __init__(self, model, inputs, family):
        self.model, self.inputs, self.family = model, inputs, family
        self.prefix_length = inputs['input_ids'].shape[1]
        if family == 'kanana':
            self.prepared = model.prepare_mm_inputs(**{k: inputs[k] for k in
                ('input_ids', 'pixel_values', 'image_metas', 'attention_mask') if k in inputs})
            assert self.prepared['inputs_embeds'].shape[1] == self.prefix_length

    def forward(self, ids, keep):
        if self.family == 'qwen':
            return self.model(**append_text(self.inputs, ids), use_cache=False,
                              logits_to_keep=keep, return_dict=True).logits
        targets = torch.as_tensor([ids], dtype=torch.long, device=self.inputs['input_ids'].device)
        embeddings = self.model.embed_text_tokens(targets)
        prepared = {'inputs_embeds': torch.cat([self.prepared['inputs_embeds'], embeddings], dim=1),
                    'attention_mask': torch.cat([self.prepared['attention_mask'], torch.ones_like(targets)], dim=1)}
        return self.model.language_model(**prepared, use_cache=False, num_logits_to_keep=keep,
                                         return_dict=True).logits

    def score(self, ids):
        return target_log_probs(self.forward(ids, len(ids)+1), ids)

    def audit(self, ids, expected, tolerance=.15):
        # 정답 토큰을 아직 입력하지 않은 별도 forward와 대조해 미래 토큰 누출을 검사한다.
        independent = []
        for i, token in enumerate(ids):
            logits = self.forward(ids[:i], 1)[0, -1].float()
            independent.append(float(logits.log_softmax(-1)[token]))
        error = max(abs(a-b) for a, b in zip(independent, expected))
        record = {'token_ids': ids, 'without_target_log_probs': independent,
                  'teacher_forced_log_probs': expected, 'max_abs_logprob_difference': error,
                  'absolute_tolerance_nats': tolerance, 'passed': error <= tolerance}
        if not record['passed']:
            raise ValueError(f'Causal prefix audit failed: {record}')
        return record
