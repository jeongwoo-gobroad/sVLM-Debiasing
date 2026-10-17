# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
import logging

from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

logger = logging.getLogger("kanana-1.5-v")


class KananaVVisionConfig(PretrainedConfig):
    model_type = "kanana-1.5-v-visual-encoder"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        mlp_ratio=4,
        num_heads=16,
        in_chans=3,
        hidden_size=1280,
        patch_size=14,
        spatial_merge_size=2,
        spatial_patch_size=14,
        temporal_patch_size=2,
        initializer_range=0.02,
        image_size="dynamic",
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_chans = in_chans
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.initializer_range = initializer_range
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std


class KananaVVisualProjectorConfig(PretrainedConfig):
    model_type = "kanana-1.5-v-visual_projector"
    base_config_key = "projector_config"

    def __init__(
        self,
        depth=2,
        encoder_hidden_size=1280,
        feature_layer_index=-1,
        hidden_size=1024,
        merge_size=2,
        mlp_depth=2,
        num_eos_tokens=0,
        output_hidden_size=2048,
        pos_emb=True,
        pos_emb_size=576,
        prenorm=False,
        projector_type="dynamic-c-abs",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.encoder_hidden_size = encoder_hidden_size
        self.feature_layer_index = feature_layer_index
        self.hidden_size = hidden_size
        self.merge_size = merge_size
        self.mlp_depth = mlp_depth
        self.num_eos_tokens = num_eos_tokens
        self.output_hidden_size = output_hidden_size
        self.pos_emb = pos_emb
        self.pos_emb_size = pos_emb_size
        self.prenorm = prenorm
        self.projector_type = projector_type


class KananaLanguageConfig(LlamaConfig):
    model_type = "kanana-1.5-3b-instruct"
    base_config_key = "text_config"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class KananaVConfig(PretrainedConfig):
    model_type = "kanana-1.5-v"
    is_composition = True

    def __init__(
        self,
        vision_config: dict = {},
        projector_config: dict = {},
        text_config: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Vision config
        self.vision_config = KananaVVisionConfig(**vision_config)

        # Visual projector config
        self.projector_config = KananaVVisualProjectorConfig(**projector_config)

        # Language model config
        self.text_config = KananaLanguageConfig(**text_config)

    @property
    def num_visual_tokens(self):
        return "dynamic"

    @property
    def hidden_size(self):
        return self.text_config.hidden_size
