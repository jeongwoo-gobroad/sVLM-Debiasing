# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
"""Observe the last prompt query without replacing the model's attention backend."""
import importlib
import inspect
import re

import numpy as np
import torch


def map_regions(tokenizer, ids, visual, candidates):
    """Round-trip text runs before mapping character spans onto actual input tokens."""
    ids = list(ids)
    visual = np.asarray(visual, dtype=bool)
    regions = np.full(len(ids), 2, dtype=np.int8)
    regions[visual] = 0
    tag_index = np.full(len(ids), -1, dtype=np.int16)
    block = '\n'.join(candidates)
    matches, markers = 0, None
    start = 0
    while start < len(ids):
        if visual[start]:
            start += 1
            continue
        end = start + 1
        while end < len(ids) and not visual[end]:
            end += 1
        run = ids[start:end]
        if min(run) < 0:
            raise ValueError('Unknown non-visual negative token')
        decoded = tokenizer.decode(run, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        chunks = re.split(r'(\nAI: |\nHuman: )', decoded) if hasattr(tokenizer, 'encode_text_only') else [decoded]
        encoded_ids, offsets, char_offset = [], [], 0
        for chunk in chunks:
            if not chunk: continue
            encoded = tokenizer.backend_tokenizer.encode(chunk, add_special_tokens=False)
            encoded_ids.extend(encoded.ids)
            offsets.extend((a + char_offset, b + char_offset) for a, b in encoded.offsets)
            char_offset += len(chunk)
        if encoded_ids != run:
            raise ValueError('Text-token round trip differs; refuse approximate span mapping')
        needle = '<tags>\n' + block + '\n</tags>'
        if needle in decoded:
            if decoded.count(needle) != 1:
                raise ValueError('Ambiguous candidate block')
            matches += 1
            begin = decoded.index(needle) + len('<tags>\n')
            finish = begin + len(block)
            spans, offset = [], begin
            for tag in candidates:
                spans.append((offset, offset + len(tag)))
                offset += len(tag) + 1
            for local, (a, b) in enumerate(offsets):
                index = start + local
                if a < finish and b > begin:
                    regions[index] = 1
                owners = [j for j, (s, e) in enumerate(spans) if a < e and b > s]
                if len(owners) > 1:
                    raise ValueError('One token overlaps two candidate labels')
                if owners:
                    tag_index[index] = owners[0]
            markers = {'opening_char': begin - len('<tags>\n'), 'closing_char': finish + 1,
                       'text_run_start_token': start, 'decoded_text_run': decoded}
        start = end
    if matches != 1 or not all(np.any(tag_index == j) for j in range(len(candidates))):
        raise ValueError('Candidate block or candidate-token coverage is incomplete')
    if not all(np.any(regions == j) for j in range(3)):
        raise ValueError('A visual/tag/prompt region is empty')
    return regions, tag_index, markers


class LastQueryProbe:
    def __init__(self, model, family):
        self.family = family
        self.rows, self.handles, self.modules = {}, [], []
        self.validate_once = True
        self.validation = {}
        for name, module in model.named_modules():
            classname = type(module).__name__
            if (family == 'qwen' and classname == 'Qwen3_5Attention') or (
                    family == 'kanana' and name.startswith('language_model.') and
                    classname in {'LlamaAttention', 'LlamaSdpaAttention', 'LlamaFlashAttention2'}):
                self.modules.append((name, module))
        if not self.modules:
            raise ValueError('No supported language attention modules found')

    def attach(self):
        for name, module in self.modules:
            self.handles.append(module.register_forward_pre_hook(self.capture, with_kwargs=True))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def capture(self, module, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        if hidden.shape[0] != 1 or hidden.shape[1] < 2:
            raise ValueError('Probe expects one uncached multimodal prefill')
        if kwargs.get('past_key_values') is not None or kwargs.get('past_key_value') is not None:
            raise ValueError('Cached attention is outside this prefill-only protocol')
        implementation = importlib.import_module(type(module).__module__)
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        if self.family == 'qwen':
            query, _gate = torch.chunk(module.q_proj(hidden).view(
                *hidden.shape[:-1], -1, module.head_dim * 2), 2, dim=-1)
            query = module.q_norm(query.view(shape)).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
        else:
            if module.config.pretraining_tp != 1:
                raise ValueError('Unsupported tensor parallel projection')
            query = module.q_proj(hidden).view(shape).transpose(1, 2)
            key = module.k_proj(hidden).view(shape).transpose(1, 2)
        embeddings = kwargs.get('position_embeddings')
        if embeddings is None:
            raise ValueError('Actual rotary embeddings must be supplied by the backbone')
        query, key = implementation.apply_rotary_pos_emb(query, key, *embeddings)
        query = query[:, :, -1:, :]
        repeated = implementation.repeat_kv(key, module.num_key_value_groups)
        scale = getattr(module, 'scaling', module.head_dim ** -.5)
        scores = torch.matmul(query, repeated.transpose(-2, -1)) * scale
        mask = kwargs.get('attention_mask')
        if mask is not None:
            if mask.ndim != 4:
                raise ValueError('Only additive 4D attention masks are supported')
            mask = mask[:, :, -1:, :key.shape[-2]]
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        if not torch.isfinite(weights).all() or not torch.allclose(
                weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5):
            raise ValueError('Invalid normalized attention row')
        # 실제 백엔드는 그대로 둔다. 복제한 마지막 query 행만 관찰한다.
        self.rows[module.layer_idx] = weights[0, :, 0].cpu().numpy()
        if self.validate_once and not self.validation:
            if self.family == 'qwen':
                value = module.v_proj(hidden).view(shape).transpose(1, 2)
                _out, reference = implementation.eager_attention_forward(
                    module, query, key, value, mask, scaling=scale, dropout=0.0)
                delta = (reference.float() - weights.to(reference.dtype).float()).abs().max().item()
            else:
                # 동일한 실제 입력의 짧은 prefix로 공식 eager 구현과 수식/축을 검증한다.
                n = min(64, hidden.shape[1])
                probe_kwargs = {'hidden_states': hidden[:, :n],
                                'position_embeddings': tuple(x[:, :n] for x in embeddings),
                                'attention_mask': None, 'output_attentions': True, 'use_cache': False}
                _out, reference, _cache = implementation.LlamaAttention.forward(module, **probe_kwargs)
                q = module.q_proj(hidden[:, :n]).view(1, n, -1, module.head_dim).transpose(1, 2)
                k = module.k_proj(hidden[:, :n]).view(1, n, -1, module.head_dim).transpose(1, 2)
                q, k = implementation.apply_rotary_pos_emb(q, k, *probe_kwargs['position_embeddings'])
                logits = torch.matmul(q[:, :, -1:], implementation.repeat_kv(
                    k, module.num_key_value_groups).transpose(-2, -1)) * scale
                expected = torch.softmax(logits, -1, dtype=torch.float32).to(reference.dtype)
                delta = (reference[:, :, -1:].float() - expected.float()).abs().max().item()
            if delta > 1e-5:
                raise ValueError(f'Official eager attention row mismatch: {delta}')
            self.validation = {'official_eager_row_max_abs_error': delta,
                               'implementation_path': inspect.getfile(implementation)}

    def array(self):
        indices = sorted(self.rows)
        if len(indices) != len(self.modules):
            raise ValueError('Missing or duplicate layer observations')
        return np.stack([self.rows[i] for i in indices]), np.array(indices)
