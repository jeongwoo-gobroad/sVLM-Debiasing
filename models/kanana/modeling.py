# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
# Upstream implementation: Copyright KAKAO Corp.; see LICENSE and Notice here.
# Modified by Jeongwoo Kim: project/provenance comments only; implementation unchanged.
# The project header above does not claim ownership of the upstream implementation.
from functools import partial
import inspect
import logging
import re
from typing import Optional, Tuple, Union

from einops import rearrange
from timm.layers import LayerNorm, LayerNorm2d
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.regnet import RegStage
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModelForCausalLM
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLVisionConfig,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    Qwen2VLPreTrainedModel,
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLVisionBlock,
    VisionRotaryEmbedding
)

from .configuration import KananaVVisualProjectorConfig, KananaVConfig

logger = logging.getLogger("kanana-1.5-v")


def build_pos_embeds(
    config: KananaVVisualProjectorConfig, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: KananaVVisualProjectorConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: KananaVVisualProjectorConfig):
    if getattr(config, "prenorm", False):
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth: int, hidden_size: int, output_hidden_size: int):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class PatchMerge(nn.Module):
    def __init__(self, merge_size):
        super().__init__()
        self.merge_size = merge_size

    def forward(self, x, channel_last=False):
        if channel_last:
            x = rearrange(x, "B H W D -> B D H W")
        _, D, H, W = x.shape
        merged_x = rearrange(
            x, "B D (H h2) (W w2) -> B (D h2 w2) H W", h2=self.merge_size, w2=self.merge_size
        )
        return merged_x


class DynamicCAbstractor(nn.Module):
    """Dynamic C-Abstractor based on RegBlock"""

    def __init__(self, config: KananaVVisualProjectorConfig, num_input_tokens: int):
        super().__init__()
        self.config = config
        if num_input_tokens == -1:
            num_input_tokens = config.pos_emb_size
        self.num_input_tokens = num_input_tokens

        self.merge_size = config.merge_size
        self.pos_emb_size = config.pos_emb_size

        self.eos_tokens = build_eos_tokens(config, config.output_hidden_size)
        self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)
        self.prenorm = build_prenorm(config)

        self.build_net()

    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.config.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = PatchMerge(merge_size=self.merge_size)
        s2 = RegBlock(
            depth,
            self.merge_size**2 * hidden_size,
            hidden_size,
        )

        if depth:
            self.net = nn.ModuleList([s1, sampler, s2])
            self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        else:
            self.net = sampler
            self.readout = build_mlp(mlp_depth, encoder_hidden_size, output_hidden_size)

    def forward(self, flattened_visual_embeds, grid_thw, **unused_kwargs):
        n_token_loc = torch.prod(grid_thw, dim=1)
        split_visual_embeds = torch.split(flattened_visual_embeds, n_token_loc.tolist())

        flattened_visual_embeds = []
        for _visual_embeds, _grid_thw in zip(split_visual_embeds, grid_thw):
            T, H, W = _grid_thw
            assert T == 1, "T must be 1. Video is not supported yet."
            reshaped_visual_embeds = rearrange(
                _visual_embeds, "(t h w) d -> 1 t h w d", t=T, h=H, w=W
            )
            # remove temporal dim
            reshaped_visual_embeds = reshaped_visual_embeds[:, 0]

            if self.prenorm is not None:
                reshaped_visual_embeds = self.prenorm(reshaped_visual_embeds)

            if self.pos_emb is not None:
                # interpolate pos emb and add to visual embeds
                _local_pos_emb = resample_abs_pos_embed(
                    posemb=self.pos_emb,
                    old_size=tuple([int(self.pos_emb_size**0.5)] * 2),
                    new_size=(H, W),
                    num_prefix_tokens=0,
                )
                _local_pos_emb = rearrange(
                    _local_pos_emb,
                    "1 (h w) d -> 1 h w d",
                    h=H,
                    w=W,
                )
                reshaped_visual_embeds = reshaped_visual_embeds + _local_pos_emb

            reshaped_visual_embeds = self._forward(
                reshaped_visual_embeds,
                input_size=(H, W),
            )
            flattened_visual_embeds.append(reshaped_visual_embeds)
        reshaped_visual_embeds = torch.cat(flattened_visual_embeds, dim=0)
        output = BaseModelOutput(last_hidden_state=reshaped_visual_embeds)
        return output

    def _forward(self, x, input_size):
        h, w = input_size
        x = rearrange(x, "1 h w d -> 1 d h w", h=h, w=w)
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        x = rearrange(x, "1 d h w -> (h w) d")
        x = self.readout(x)
        return x


class CustomQwen2VLVE(Qwen2VisionTransformerPretrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config) -> None:
        Qwen2VLPreTrainedModel.__init__(self, config)
        self.spatial_merge_size = config.spatial_merge_size
        self.gradient_checkpointing = False

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        assert return_dict, "Only return_dict=True is supported."

        encoder_states = () if output_hidden_states else None

        hidden_states = self.patch_embed(pixel_values)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if "position_embeddings" in inspect.signature(blk.forward).parameters:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                )
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)

    def get_num_tokens(self):
        return -1


class KananaVPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = KananaVConfig
    base_model_prefix = "kanana-1.5-v"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False

    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"language_model.encoder.embed_tokens.weight",
        r"language_model.decoder.embed_tokens.weight",
        r"language_model.lm_head.weight",
    ]
    _no_split_modules = [
        "CustomQwen2VLVE",
        "DynamicCAbstractor",
        "LlamaForCausalLM",
        "Parameter",
    ]

    def _init_weights(self, module):
        """Initialize the weights"""
        if (
            isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Embedding)
            or isinstance(module, nn.Linear)
        ):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            raise ValueError()


class KananaVForConditionalGeneration(KananaVPreTrainedModel):
    config_class = KananaVConfig

    def __init__(self, config: KananaVConfig):
        super().__init__(config)

        logger.info("Build vision model ...")
        try:
            self.vision_model = CustomQwen2VLVE._from_config(
                config.vision_config,
                attn_implementation="flash_attention_2",
            )
        except Exception as e:
            logger.error(e)
            logger.info("Failed to load Vision Encoder with flash_attention_2. Try without it ...")
            self.vision_model = CustomQwen2VLVE._from_config(config.vision_config)

        logger.info("Build projector ...")
        self.abstractor = DynamicCAbstractor(config.projector_config,
                                             num_input_tokens=self.vision_model.get_num_tokens())

        logger.info("Build language model ...")
        self.language_model = LlamaForCausalLM._from_config(config=config.text_config)

        self.post_init()

    def forward_vision(self, pixel_values, image_metas: Optional[dict] = None):
        vision_model_args = {
            "pixel_values": pixel_values,
            "return_dict": True,
            "output_hidden_states": True,
            "grid_thw": image_metas["vision_grid_thw"],
        }
        v_outputs = self.vision_model(**vision_model_args)
        layer_index = self.config.projector_config.feature_layer_index
        visual_features = self._get_visual_feature_at(v_outputs.hidden_states, layer_index)
        return visual_features

    def forward_projector(self, visual_features, image_metas: Optional[dict] = None):
        assert image_metas is not None
        visual_embeds = self.abstractor(
            visual_features,
            grid_thw=image_metas["vision_grid_thw"],
        )["last_hidden_state"]
        return visual_embeds

    def forward_and_project_vision(self, pixel_values, image_metas: Optional[dict] = None):
        assert pixel_values is not None
        visual_features = self.forward_vision(pixel_values, image_metas=image_metas)
        visual_embeds = self.forward_projector(visual_features, image_metas=image_metas)
        return visual_embeds

    def _get_visual_feature_at(self, v_output, layer_index):
        if isinstance(layer_index, list):
            visual_features = torch.stack(v_output, dim=1)[:, layer_index]  # [B, n_scales, L, dim]
        else:
            visual_features = v_output[layer_index]  # [B, L, dim]
        return visual_features

    def embed_text_tokens(self, input_ids):
        """Embed input_ids into text_embeds, ignoring media tokens (negative values)."""
        input_ids = input_ids.clone()
        input_ids[input_ids < 0] = 0

        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        if hasattr(self.language_model, "transformer") and hasattr(
            self.language_model.transformer, "word_embeddings_layernorm"
        ):
            text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds)

        return text_embeds

    def prepare_mm_inputs(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: Optional[list[torch.FloatTensor]] = None,
        image_metas: Optional[dict] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """Prepare multimodal inputs from input_ids and pixel_values."""
        if pixel_values is not None:
            pixel_values = pixel_values.to(self._get_input_dtype())

        if attention_mask is None:
            attention_mask = input_ids.new_ones(*input_ids.shape)

        # Get Text Embeddings
        text_embeds = self.embed_text_tokens(input_ids)
        flattened_text_embeds = rearrange(text_embeds, "b l d -> (b l) d")
        flattened_input_ids = rearrange(input_ids, "b l -> (b l)")

        # Get Visual Embeddings
        if pixel_values is not None:
            flattened_visual_embeds = self.forward_and_project_vision(
                pixel_values, image_metas
            )
            flattened_text_embeds[flattened_input_ids == -1] = flattened_visual_embeds

        input_embeds = rearrange(
            flattened_text_embeds, "(b l) d -> b l d", b=input_ids.shape[0]
        )
        return_inputs = {
            "inputs_embeds": input_embeds,
            "attention_mask": attention_mask,
        }
        return return_inputs

    def forward(
        self,
        pixel_values: list[torch.FloatTensor],
        image_metas: dict[list],
        input_ids: torch.FloatTensor,
        seq_length: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs = self.prepare_mm_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_metas=image_metas,
            attention_mask=attention_mask,
        )

        outputs = self.language_model(
            **inputs,
            labels=labels,
            position_ids=None,
            return_dict=return_dict,
            output_attentions=self.config.output_attentions,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor = None,
        image_metas: dict[list] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        seq_length: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if input_ids is None:
            return self.language_model.generate(attention_mask=attention_mask, **generate_kwargs)
        if pixel_values is None:
            return self.language_model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)

        if (
            image_metas is not None
            and image_metas.get("vision_grid_thw") is not None
            and isinstance(image_metas.get("vision_grid_thw"), torch.Tensor)
        ):
            image_metas["vision_grid_thw"] = image_metas["vision_grid_thw"].to(input_ids.device)

        inputs = self.prepare_mm_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_metas=image_metas,
            attention_mask=attention_mask,
        )

        outputs = self.language_model.generate(
            **inputs,
            **generate_kwargs,
        )

        return outputs

    def _get_input_dtype(self):
        dtype = next(self.vision_model.parameters()).dtype
        return dtype
