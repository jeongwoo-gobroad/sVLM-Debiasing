# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
from collections import defaultdict
import logging
import re
from typing import Optional

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Role tokens
AI = "AI: "
HUMAN = "Human: "
_AI = "\n" + AI
_HUMAN = "\n" + HUMAN

# special media tokens
IMAGE = "<image>"
IMAGE_ROW_SEPARATOR = "\n"
IMAGE_GLOBAL_LOCAL_SEPARATOR = "\n"
MEDIA_TOKENS = {
    "image": [IMAGE],
}

_INFINITE = int(1e12)  # infinite token length for no-truncation

logger = logging.getLogger("kanana-1.5-v")


class AttrDict(dict):
    __slots__ = ()

    def __getattr__(self, name):
        try:
            val = self[name]
        except KeyError:
            raise AttributeError(name) from None

        if isinstance(val, dict) and not isinstance(val, AttrDict):
            val = AttrDict(val)
            self[name] = val
        return val

    def __setattr__(self, name, value):
        if name.startswith('_'):
            return super().__setattr__(name, value)
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None


def to_attrdict(obj):
    if isinstance(obj, dict) and not isinstance(obj, AttrDict):
        return AttrDict({k: to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attrdict(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_attrdict(x) for x in obj)
    return obj


def _pad_trunc(
    x: list[list[int]],
    padding: str,
    padding_side: str,
    pad_value: int,
    max_length: int,
) -> torch.LongTensor:
    """Pad and truncate sequences to the same length

    Args:
        x (list[list[int]])
        padding ("longest" or "max_length")
        padding_side ("left" or "right")
        pad_value (int)
        max_length (int or None): if padding == "max_length", max_length should be given.
    """
    assert padding in ["longest", "max_length"]
    assert padding_side in ["left", "right"]

    lengths = [len(sample) for sample in x]
    if padding == "longest":
        max_length = max(lengths)

    new_x = []
    for sample, length in zip(x, lengths):
        if torch.is_tensor(sample):
            sample = sample.tolist()

        if length >= max_length:
            new_x.append(sample[:max_length])
            continue

        padding_size = max_length - length
        pads = [pad_value] * padding_size
        if padding_side == "right":
            new_x.append(sample + pads)
        else:
            new_x.append(pads + sample)

    return torch.as_tensor(new_x, dtype=torch.long)


class KananaVTokenizerMixin:
    def mllm_setup(self, num_visual_tokens: int):
        self.num_visual_tokens = num_visual_tokens

        # Currently we only support the image modality for media modality.
        self.media_tokens = {k: -int(i + 1) for i, k in enumerate(MEDIA_TOKENS["image"])}
        self.media_lengths = {MEDIA_TOKENS["image"][0]: num_visual_tokens}

    def repeat_image_tokens(
        self, hw_tokens, with_row_separator=True, add_global_local_separator=False
    ):
        if len(hw_tokens) == 3:
            T, H, W = hw_tokens
        else:
            H, W = hw_tokens

        repeated_tokens = []

        if add_global_local_separator:
            global_local_separator = self(IMAGE_GLOBAL_LOCAL_SEPARATOR, add_special_tokens=False)[
                "input_ids"
            ]

            repeated_tokens += global_local_separator

        if with_row_separator:
            row_sep = self(IMAGE_ROW_SEPARATOR, add_special_tokens=False)["input_ids"]

        for h_idx in range(H):
            repeated_tokens += [self.media_tokens[IMAGE]] * W
            if with_row_separator and h_idx != H - 1:
                repeated_tokens += row_sep

        return repeated_tokens

    def encode_prompt(
        self, prompt: str, max_length: int | None = None, image_meta: dict | None = None
    ) -> dict:
        """Tokenize prompt which consists of image-text or text only, with role tokens.
        Role pattern is "AI: " or "Human: ".

        Args:
            prompt
            max_length (int or None): here, max_length is used for truncation.
                If max_length is None, no truncation is applied.
        """
        max_length = max_length or _INFINITE  # if None, set to infinite for no-truncation

        # output enc_chunk
        enc_chunk = []

        # Text-only or Image-Text Data
        # split prompt into chunks by media and role tokens
        tokens_to_split = list(self.media_tokens.keys()) + [_AI, _HUMAN]
        pattern = "|".join(map(re.escape, tokens_to_split))
        chunk_strs = re.split(f"({pattern})", prompt)
        chunk_strs = [x for x in chunk_strs if len(x) > 0]
        # tokenize chunks
        img_idx = 0  # for sync with image_meta
        for idx, chunk_str in enumerate(chunk_strs):
            if chunk_str in self.media_tokens:
                if chunk_str == IMAGE:
                    image_token_thw = (
                        image_meta["image_token_thw"][img_idx]
                        if image_meta.get("image_token_thw")
                        else None
                    )

                    media_tokens = self.repeat_image_tokens(
                        image_token_thw,
                        with_row_separator=True,
                        add_global_local_separator=True,
                    )
                    # increment image index
                    img_idx += 1

                else:
                    raise ValueError("Unknown chunk str", chunk_str)

                enc_chunk += media_tokens

            else:
                curr_chunk = self(chunk_str, add_special_tokens=False)["input_ids"]
                enc_chunk += curr_chunk

        L = len(enc_chunk)

        input_ids = torch.as_tensor(enc_chunk, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        assert L <= max_length, (
            f"[Length exceeded] Input sequence length ({L}) is greater than "
            f"the allowed max_length ({max_length}). "
            "Please truncate the sequence or increase max_length."
        )

        return {
            "input_ids": input_ids,  # [L]
            "seq_length": L,  # int
            "attention_mask": attention_mask,  # [L]
        }

    def batch_collate_pad(
        self,
        batch: list,
        padding: str,
        padding_side: str,
        max_length: int | None,
    ) -> dict[str, torch.LongTensor]:
        """Collate batch and pad/truncate to the same length

        Args:
            batch
            padding ("longest" or "max_length")
            padding_side ("left" or "right")
            pad_value (int)
            max_length (int or None): if padding == "max_length", max_length should be given
        """
        if padding == "max_length":
            assert max_length is not None, "max_length should be given if padding == 'max_length'"
        else:
            # if padding == 'longest' and max_length is None, set to infinite for no-truncation
            max_length = max_length or _INFINITE

        input_ids = [sample["input_ids"] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        seq_length = [sample["seq_length"] for sample in batch]

        input_ids = _pad_trunc(input_ids, padding, padding_side, self.pad_token_id, max_length)
        attention_mask = _pad_trunc(attention_mask, padding, padding_side, 0, max_length)
        seq_length = torch.as_tensor(seq_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_length": seq_length,
        }

    def get_chat_template(self) -> str:
        """Method for bw-compat: old HF transformers (e.g., 4.41.0) does not have get_chat_template
        """
        return self.chat_template


class KananaVTokenizer(PreTrainedTokenizer, KananaVTokenizerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode_text_only(self, prompt: str) -> dict:
        """Tokenize prompt which consists of text only, with role tokens."""
        tokens_to_split = [_AI, _HUMAN]
        pattern = "|".join(map(re.escape, tokens_to_split))
        chunk_strs = re.split(f"({pattern})", prompt)
        chunk_strs = [x for x in chunk_strs if len(x) > 0]

        # encode chunk strs
        encodings = defaultdict(list)
        for chunk_str in chunk_strs:
            encoding = super().__call__(chunk_str, add_special_tokens=False)
            for k, v in encoding.items():
                encodings[k].extend(v)
        encodings = to_attrdict(encodings)
        return encodings

    def __call__(self, text, *args, **kwargs) -> dict[str, list[int]]:
        tokens_to_split = [_AI, _HUMAN]
        pattern = "|".join(map(re.escape, tokens_to_split))

        if isinstance(text, str):
            if re.search(pattern, text):
                return self.encode_text_only(prompt=text)
            else:
                return super().__call__(text, *args, **kwargs)

        elif isinstance(text, (list, tuple)):
            encodings = defaultdict(list)
            for t in text:
                if re.search(pattern, t):
                    encoding = self.encode_text_only(prompt=t)
                else:
                    encoding = super().__call__(t, *args, **kwargs)

                for k, v in encoding.items():
                    encodings[k].append(v)
            encodings = to_attrdict(encodings)
            return encodings
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")

    def encode(self, *args, **kwargs) -> list[int]:
        input_ids = self.__call__(*args, **kwargs)["input_ids"]
        return input_ids if not any(isinstance(e, list) for e in input_ids) else [i for sub in input_ids for i in sub]


class KananaVTokenizerFast(PreTrainedTokenizerFast, KananaVTokenizerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode_text_only(self, prompt: str) -> dict:
        """Tokenize prompt which consists of text only, with role tokens."""
        tokens_to_split = [_AI, _HUMAN]
        pattern = "|".join(map(re.escape, tokens_to_split))
        chunk_strs = re.split(f"({pattern})", prompt)
        chunk_strs = [x for x in chunk_strs if len(x) > 0]

        # encode chunk strs
        encodings = defaultdict(list)
        for chunk_str in chunk_strs:
            encoding = super().__call__(chunk_str, add_special_tokens=False)
            for k, v in encoding.items():
                encodings[k].extend(v)
        encodings = to_attrdict(encodings)
        return encodings

    def __call__(self, text, *args, **kwargs) -> dict[str, list[int]]:
        tokens_to_split = [_AI, _HUMAN]
        pattern = "|".join(map(re.escape, tokens_to_split))

        if isinstance(text, str):
            if re.search(pattern, text):
                return self.encode_text_only(prompt=text)
            else:
                return super().__call__(text, *args, **kwargs)

        elif isinstance(text, (list, tuple)):
            encodings = defaultdict(list)
            for t in text:
                if re.search(pattern, t):
                    encoding = self.encode_text_only(prompt=t)
                else:
                    encoding = super().__call__(t, *args, **kwargs)

                for k, v in encoding.items():
                    encodings[k].append(v)
            encodings = to_attrdict(encodings)
            return encodings
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")

    def encode(self, *args, **kwargs) -> list[int]:
        input_ids = self.__call__(*args, **kwargs)["input_ids"]
        return input_ids if not any(isinstance(e, list) for e in input_ids) else [i for sub in input_ids for i in sub]
