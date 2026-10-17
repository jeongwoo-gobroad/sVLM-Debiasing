# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
import logging

import torch
from PIL.Image import Image
from transformers.processing_utils import ProcessorMixin


logger = logging.getLogger("kanana-1.5-v")


HUMAN = "Human: "
AI = "AI: "
CHAT_TEMPLATE = (
"""
{%- if bos_token is defined and bos_token %}
    {{- bos_token }}
{%- endif %}
{%- set intro %}
The following is a conversation between a curious human and AI assistant. 당신은 Kakao에서 개발된 인공지능 언어모델이고 이름은 kanana입니다.
Knowledge Cutoff Date: June30, 2024.
Capabilities and Limitations:
   - I cannot search for external content such as weather, news, or the current date and time.
   - If a URL is provided, I cannot access it directly. Insteaed, please copy and provide the relevant content for me to process.
{%- endset %}
{{ intro }}
{{- '\n' }}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {%- if message['content'] is string %}
            {{- message['content'] }}
        {%- else %}
            {% for content_item in message['content'] %}
                {% if content_item['type'] == 'text' %}
                    {{- content_item['text'] }}
                {% endif %}
            {% endfor %}            
        {%- endif %}
    {%- elif message['role'] == 'user' %}
        {%- if message['content'] is string %}
            {{- '<|USER|>' -}}
            {{- message['content'] }}
        {%- else %}
            {% set ns = namespace(has_images=false, has_text=false) %}
            {% for content_item in message['content'] %}
                {% if content_item['type'] in ['image', 'image_url'] %}
                    {% set ns.has_images = true %}
                {% elif content_item['type'] == 'text' %}
                    {% set ns.has_text = true %}
                {% endif %}
            {% endfor %}
            {% if ns.has_images %}
                {{- '<|USER|>' -}}
                {% for content_item in message['content'] %}
                    {% if content_item['type'] in ['image', 'image_url'] %}
                        {{- '<image>' }}
                    {% endif %}
                {% endfor %}
            {% endif %}
            {% if ns.has_text %}
                {{- '<|USER|>' -}}
                {% for content_item in message['content'] %}
                    {% if content_item['type'] == 'text' %}
                        {{- content_item['text'] }}
                    {% endif %}
                {% endfor %}
            {% endif %}
        {%- endif %}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|ASSISTANT|>' -}}
        {%- if message['content'] is string %}
            {{- message['content'] + eos_token }}
        {%- else %}
            {% for content_item in message['content'] %}
                {% if content_item['type'] == 'text' %}
                    {{- content_item['text'] }}
                {% endif %}
            {% endfor %}
        {%- endif %}
    {%- endif %}
    {%- if not loop.last %}
        {{- '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '\n<|ASSISTANT|>' }}
{%- endif %}
""".strip()
    .replace("<|USER|>", HUMAN)
    .replace("<|ASSISTANT|>", AI)
)


class KananaVProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = []
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tokenizer.mllm_setup("dynamic")

    def conv2prompt(
        self,
        conv: list[dict] | str,
        chat_template=CHAT_TEMPLATE,
        add_generation_prompt=False,
    ) -> str:
        """Convert conversation to prompt"""
        if isinstance(conv, list):
            prompt = self.tokenizer.apply_chat_template(
                conversation=conv,
                tokenize=False,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
            )
        elif isinstance(conv, str):
            prompt = conv
        else:
            raise TypeError(f"conv must be list or str, but got {type(conv)}")

        return prompt

    def __call__(self, data: dict, max_length, add_generation_prompt=False):
        return self.encode(data, max_length, add_generation_prompt=add_generation_prompt)

    def encode(self, data: dict, max_length, add_generation_prompt=False) -> dict:
        """
        Args:
            data (dict): {
                "conv": [
                    {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
                    {"role": "user", "content": IMAGE},
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
                    ...
                ],
                "image": [
                    PIL.Image,
                    ...
                ]
            }

        Return:
            data (dict): {
                "text": text_tokens_from_tokenizer,
                "text_raw": prompt,
                "image": pixel_values,
                "image_meta": image_meta (dict of list) includes image resolution, etc.
            }
        """
        assert "images" not in data

        conv = data["conv"]
        images: list[Image] = data.get("image")  # PIL images

        data = {
            "text": None,
            "text_raw": None,
            "image": None,
            "image_meta": None,
        }

        # image
        if images:
            processor_output = [
                self.image_processor(image) for image in images if image
            ]
            pixel_values = [
                processor_output["pixel_values"] for processor_output in processor_output
            ]
            image_meta = [processor_output["image_meta"] for processor_output in processor_output]
            if pixel_values:
                pixel_values = torch.concat(pixel_values, dim=0)
                data["image"] = pixel_values
                data["image_meta"] = {k: [d[k] for d in image_meta] for k in image_meta[0]}

        # text
        prompt = self.conv2prompt(conv, add_generation_prompt=add_generation_prompt)
        text_tokens = self.tokenizer.encode_prompt(
            prompt,
            max_length,
            image_meta=data["image_meta"],
        )

        data["text"] = text_tokens
        data["text_raw"] = prompt

        return data

    def batch_encode_collate(
        self,
        data_list: list[dict],
        padding: str = "longest",
        padding_side: str = "right",
        max_length: int | None = None,
        add_generation_prompt=False,
    ):
        """Encode batch and collate them"""
        batch = [
            self.encode(data, max_length, add_generation_prompt=add_generation_prompt)
            for data in data_list
        ]
        batch = self.collate(
            batch,
            padding=padding,
            padding_side=padding_side,
            max_length=max_length,
        )

        return batch

    def collate(
        self,
        batch,
        padding,
        padding_side,
        max_length,
    ):
        """Collate encoded results to model inputs"""
        text_batch = [data["text"] for data in batch]

        text_batch = self.tokenizer.batch_collate_pad(
            text_batch,
            padding=padding,
            padding_side=padding_side,
            max_length=max_length,
        )

        image_list = [data["image"] for data in batch if data["image"] is not None]
        image_meta = [data["image_meta"] for data in batch if data["image_meta"] is not None]
        if len(image_meta) > 0:
            image_meta = {
                k: sum([d[k] for d in image_meta], []) for k in image_meta[0]
            }
            if image_meta.get("vision_grid_thw"):
                image_meta["vision_grid_thw"] = torch.tensor(image_meta["vision_grid_thw"])
        else:
            image_meta = None

        output_batch = text_batch

        output_batch["pixel_values"] = torch.cat(image_list, dim=0) if len(image_list) > 0 else None
        output_batch["image_metas"] = image_meta
        return output_batch

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
