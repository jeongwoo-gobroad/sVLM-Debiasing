# (c) 2026 Jeongwoo Kim from KNU CSE, All rights reserved
# sVLM Inference-time Debiasing Research
import logging
import math
from typing import Optional, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import convert_to_rgb, resize
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
)
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

logger = logging.getLogger("kanana-1.5-v")


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class KananaVImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = OPENAI_CLIP_MEAN,
        image_std: Optional[Union[float, list[float]]] = OPENAI_CLIP_STD,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 4 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = Image.BICUBIC
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        self.do_convert_rgb = do_convert_rgb
        self.input_data_format = ChannelDimension.LAST

    def _preprocess(
        self,
        images: Union[ImageInput],
        do_resize: bool = True,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.
        (samuel) From image_processing_qwen2_vl.py

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    input_data_format=input_data_format,
                )

            if do_rescale:
                image = self.rescale(
                    image, scale=rescale_factor, input_data_format=input_data_format
                )

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            # Convert from (num_images, height, width, num_channels) format.
            patches = rearrange(patches, "N H W C -> N C H W")
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        flatten_patches = rearrange(
            patches,
            "(nT T) C (nH sH H) (nW sW W) -> (nT nH nW sH sW) (C T H W)",
            T=self.temporal_patch_size,
            H=self.patch_size,
            W=self.patch_size,
            nH=grid_h // self.merge_size,
            nW=grid_w // self.merge_size,
            sH=self.merge_size,
            sW=self.merge_size,
        )
        return (
            flatten_patches,
            (grid_t, grid_h, grid_w),
            (resized_height, resized_width),
            (height, width),
        )

    def resize_pil_image(self, image):
        """
        if width * height < self.min_pixels:
            expansion_ratio = np.ceil(1 / np.sqrt((width * height / self.min_pixels)))
            width, height = (width * expansion_ratio, height * expansion_ratio)
        """
        ori_width, ori_height = image.size
        width, height = (ori_width, ori_height)
        if min(width, height) < self.patch_size * self.merge_size:
            scale = self.patch_size * self.merge_size / min(width, height)
            width, height = (int(width * scale), int(height * scale))
        if (width, height) != (ori_width, ori_height):
            image = image.resize((width, height), resample=Image.BICUBIC)

        return image

    def __call__(self, image):
        """
        Args:
            image:

        Return:
            image_input (tensors): (number of tiles, 3, H, W)
            hw_tiles (tuple): (height, width) of the tiles
            hw_best_resolution (tuple): (height, width) of the best resolution (with padding)
            hw_orig_resolution (tuple): (height, width) of the original resolution
        """
        do_resize = self.do_resize
        resample = self.resample
        do_rescale = self.do_rescale
        rescale_factor = self.rescale_factor
        do_normalize = self.do_normalize
        image_mean = self.image_mean
        image_std = self.image_std
        do_convert_rgb = self.do_convert_rgb
        input_data_format = self.input_data_format

        if image is not None:
            # resize imagee if the shortest side is smaller than patch_size * merge_size
            image = self.resize_pil_image(image)

            patches, image_grid_thw, resized_hw, original_hw = self._preprocess(
                images=image,
                do_resize=do_resize,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                data_format=ChannelDimension.LAST,
            )

            pixel_values = torch.tensor(patches)
            image_meta = {
                "vision_grid_thw": image_grid_thw,
                "hw_best_resolution": resized_hw,
                "hw_orig_resolution": original_hw,
                "image_token_thw": (
                    image_grid_thw[0],
                    image_grid_thw[1] // self.merge_size,
                    image_grid_thw[2] // self.merge_size,
                ),
            }
        else:
            pixel_values = None
            image_meta = None

        return {
            "pixel_values": pixel_values,
            "image_meta": image_meta,
        }
