from collections import OrderedDict
from cv2 import transform

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
# from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
#                                     ToTensor)
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
import os

# try:
#     from torchvision.transforms import InterpolationMode
#     BICUBIC = InterpolationMode.BICUBIC
# except ImportError:
#     BICUBIC = Image.BICUBIC

# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for CLIP."""

from typing import Dict, List, Optional, Union

import numpy as np

from transformers.modeling_utils import PreTrainedModel,PretrainedConfig
from transformers.models.clip import CLIPVisionConfig
from transformers import CLIPImageProcessor
from transformers.image_processing_utils import BaseBatchFeature, BaseImageProcessor, get_size_dict, BatchFeature

from transformers.image_transforms import (
    center_crop,
    convert_to_rgb,
    get_resize_output_image_size,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    import PIL

# class UnicomImageProcessor(BaseImageProcessor):
#     r"""
#     Constructs a CLIP image processor.

#     Args:
#         do_resize (`bool`, *optional*, defaults to `True`):
#             Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
#             `do_resize` in the `preprocess` method.
#         size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
#             Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
#             the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
#             method.
#         resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
#             Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
#         do_center_crop (`bool`, *optional*, defaults to `True`):
#             Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
#             `preprocess` method.
#         crop_size (`Dict[str, int]` *optional*, defaults to 224):
#             Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
#             method.
#         do_rescale (`bool`, *optional*, defaults to `True`):
#             Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
#             the `preprocess` method.
#         rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
#             Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
#             method.
#         do_normalize:
#             Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
#         image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
#             Mean to use if normalizing the image. This is a float or list of floats the length of the number of
#             channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
#         image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
#             Image standard deviation.
#         do_convert_rgb (`bool`, *optional*, defaults to `True`):
#             Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
#             number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
#     """

#     model_input_names = ["pixel_values"]

#     def __init__(
#         self,
#         do_resize: bool = True,
#         size: Dict[str, int] = {"shortest_edge": 336},
#         resample: PILImageResampling = PILImageResampling.BICUBIC,
#         do_center_crop: bool = True,
#         crop_size: Dict[str, int] = {"height": 336, "width": 336},
#         do_rescale: bool = True,
#         rescale_factor: Union[int, float] = 1 / 255,
#         do_normalize: bool = True,
#         image_mean: Optional[Union[float, List[float]]] = (0.48145466, 0.4578275, 0.40821073),
#         image_std: Optional[Union[float, List[float]]] = (0.26862954, 0.26130258, 0.27577711),
#         do_convert_rgb: bool = True,
#         **kwargs,
#     ) -> None:
#         super().__init__(**kwargs)
#         size = size if size is not None else {"shortest_edge": 336}
#         size = get_size_dict(size, default_to_square=False)
#         crop_size = crop_size if crop_size is not None else {"height": 336, "width": 336}
#         crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

#         self.do_resize = do_resize
#         self.size = size
#         self.resample = resample
#         self.do_center_crop = do_center_crop
#         self.crop_size = crop_size
#         self.do_rescale = do_rescale
#         self.rescale_factor = rescale_factor
#         self.do_normalize = do_normalize
#         self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
#         self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
#         self.do_convert_rgb = do_convert_rgb

#     def resize(
#         self,
#         image: np.ndarray,
#         size: Dict[str, int],
#         resample: PILImageResampling = PILImageResampling.BICUBIC,
#         data_format: Optional[Union[str, ChannelDimension]] = None,
#         **kwargs,
#     ) -> np.ndarray:
#         """
#         Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
#         resized to keep the input aspect ratio.

#         Args:
#             image (`np.ndarray`):
#                 Image to resize.
#             size (`Dict[str, int]`):
#                 Size of the output image.
#             resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
#                 Resampling filter to use when resiizing the image.
#             data_format (`str` or `ChannelDimension`, *optional*):
#                 The channel dimension format of the image. If not provided, it will be the same as the input image.
#         """
#         size = get_size_dict(size, default_to_square=False)
#         if "shortest_edge" not in size:
#             raise ValueError(f"The `size` parameter must contain the key `shortest_edge`. Got {size.keys()}")
#         output_size = get_resize_output_image_size(image, size=size["shortest_edge"], default_to_square=False)
#         return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)

#     def center_crop(
#         self,
#         image: np.ndarray,
#         size: Dict[str, int],
#         data_format: Optional[Union[str, ChannelDimension]] = None,
#         **kwargs,
#     ) -> np.ndarray:
#         """
#         Center crop an image. If the image is too small to be cropped to the size given, it will be padded (so the
#         returned result will always be of size `size`).

#         Args:
#             image (`np.ndarray`):
#                 Image to center crop.
#             size (`Dict[str, int]`):
#                 Size of the output image in the form of a dictionary with keys `height` and `width`.
#             data_format (`str` or `ChannelDimension`, *optional*):
#                 The channel dimension format of the image. If not provided, it will be the same as the input image.
#         """
#         size = get_size_dict(size)
#         if "height" not in size or "width" not in size:
#             raise ValueError(f"The `size` parameter must contain the keys (height, width). Got {size.keys()}")
#         return center_crop(image, size=(size["height"], size["width"]), data_format=data_format, **kwargs)

#     def rescale(
#         self,
#         image: np.ndarray,
#         scale: Union[int, float],
#         data_format: Optional[Union[str, ChannelDimension]] = None,
#         **kwargs,
#     ):
#         """
#         Rescale an image by a scale factor. image = image * scale.

#         Args:
#             image (`np.ndarray`):
#                 Image to rescale.
#             scale (`int` or `float`):
#                 Scale to apply to the image.
#             data_format (`str` or `ChannelDimension`, *optional*):
#                 The channel dimension format of the image. If not provided, it will be the same as the input image.
#         """
#         return rescale(image, scale=scale, data_format=data_format, **kwargs)

#     def normalize(
#         self,
#         image: np.ndarray,
#         mean: Union[float, List[float]],
#         std: Union[float, List[float]],
#         data_format: Optional[Union[str, ChannelDimension]] = None,
#         **kwargs,
#     ) -> np.ndarray:
#         """
#         Normalize an image. image = (image - image_mean) / image_std.

#         Args:
#             image (`np.ndarray`):
#                 Image to normalize.
#             image_mean (`float` or `List[float]`):
#                 Image mean.
#             image_std (`float` or `List[float]`):
#                 Image standard deviation.
#             data_format (`str` or `ChannelDimension`, *optional*):
#                 The channel dimension format of the image. If not provided, it will be the same as the input image.
#         """
#         return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)

#     def preprocess(
#         self,
#         images: ImageInput,
#         do_resize: bool = None,
#         size: Dict[str, int] = None,
#         resample: PILImageResampling = None,
#         do_center_crop: bool = None,
#         crop_size: int = None,
#         do_rescale: bool = None,
#         rescale_factor: float = None,
#         do_normalize: bool = None,
#         image_mean: Optional[Union[float, List[float]]] = None,
#         image_std: Optional[Union[float, List[float]]] = None,
#         do_convert_rgb: bool = None,
#         return_tensors: Optional[Union[str, TensorType]] = None,
#         data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
#         **kwargs,
#     ) -> PIL.Image.Image:
#         """
#         Preprocess an image or batch of images.

#         Args:
#             images (`ImageInput`):
#                 Image to preprocess.
#             do_resize (`bool`, *optional*, defaults to `self.do_resize`):
#                 Whether to resize the image.
#             size (`Dict[str, int]`, *optional*, defaults to `self.size`):
#                 Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
#                 the longest edge resized to keep the input aspect ratio.
#             resample (`int`, *optional*, defaults to `self.resample`):
#                 Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
#                 has an effect if `do_resize` is set to `True`.
#             do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
#                 Whether to center crop the image.
#             crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
#                 Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
#             do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
#                 Whether to rescale the image.
#             rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
#                 Rescale factor to rescale the image by if `do_rescale` is set to `True`.
#             do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
#                 Whether to normalize the image.
#             image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
#                 Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
#             image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
#                 Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
#                 `True`.
#             do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
#                 Whether to convert the image to RGB.
#             return_tensors (`str` or `TensorType`, *optional*):
#                 The type of tensors to return. Can be one of:
#                 - Unset: Return a list of `np.ndarray`.
#                 - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
#                 - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
#                 - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
#                 - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
#             data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
#                 The channel dimension format for the output image. Can be one of:
#                 - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
#                 - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
#                 - Unset: defaults to the channel dimension format of the input image.
#         """
#         do_resize = do_resize if do_resize is not None else self.do_resize
#         size = size if size is not None else self.size
#         size = get_size_dict(size, param_name="size", default_to_square=False)
#         resample = resample if resample is not None else self.resample
#         do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
#         crop_size = crop_size if crop_size is not None else self.crop_size
#         crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
#         do_rescale = do_rescale if do_rescale is not None else self.do_rescale
#         rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
#         do_normalize = do_normalize if do_normalize is not None else self.do_normalize
#         image_mean = image_mean if image_mean is not None else self.image_mean
#         image_std = image_std if image_std is not None else self.image_std
#         do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

#         images = make_list_of_images(images)

#         if not valid_images(images):
#             raise ValueError(
#                 "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
#                 "torch.Tensor, tf.Tensor or jax.ndarray."
#             )

#         if do_resize and size is None:
#             raise ValueError("Size must be specified if do_resize is True.")

#         if do_center_crop and crop_size is None:
#             raise ValueError("Crop size must be specified if do_center_crop is True.")

#         if do_rescale and rescale_factor is None:
#             raise ValueError("Rescale factor must be specified if do_rescale is True.")

#         if do_normalize and (image_mean is None or image_std is None):
#             raise ValueError("Image mean and std must be specified if do_normalize is True.")

#         # PIL RGBA images are converted to RGB
#         if do_convert_rgb:
#             images = [convert_to_rgb(image) for image in images]

#         # All transformations expect numpy arrays.
#         images = [to_numpy_array(image) for image in images]

#         if do_resize:
#             images = [self.resize(image=image, size=size, resample=resample) for image in images]

#         if do_center_crop:
#             images = [self.center_crop(image=image, size=crop_size) for image in images]

#         if do_rescale:
#             images = [self.rescale(image=image, scale=rescale_factor) for image in images]

#         if do_normalize:
#             images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

#         images = [to_channel_dimension_format(image, data_format) for image in images]

#         data = {"pixel_values": images}
#         return BatchFeature(data=data, tensor_type=return_tensors)


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :]  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward_impl(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x):
        if self.training:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        image_features = []
        for blk in self.resblocks:
            x = blk(x)
            image_features.append(x.clone().permute(1,0,2))
        return image_features


class VisionTransformer(PreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.input_resolution = config.image_size
        self.patch_size = config.patch_size
        self.output_dim = config.projection_dim
        layers = config.num_hidden_layers
        width = config.hidden_size
        heads = config.num_attention_heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=self.patch_size, stride=self.patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((self.input_resolution // self.patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, self.output_dim))

    def forward(self, x: torch.Tensor, output_hidden_states: bool = True):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        image_features = self.transformer(x)
        return image_features
    
class UnicomVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        # self.merge = args.merge
        # self.mul_layers = [5,11,17,23]
        # self.is_mul_layers = args.mul_layers
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        

    def load_model(self):
        model_config = CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, projection_dim= 1024, 
                                  num_hidden_layers=24, num_attention_heads= 16, image_size= 336, 
                                  layer_norm_eps=1e-6, patch_size= 14)
        self.vision_tower =  VisionTransformer(config=model_config)
        weight = torch.load(os.path.join(self.vision_tower_name, "pytorch_model.pt"))
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower.load_state_dict(weight)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # if self.is_mul_layers:
        #     image_features = torch.stack(image_forward_outs)[self.mul_layers].reshape((image_forward_outs[0].shape[0],image_forward_outs[0].shape[1],-1))
        # else:
        #     image_features = image_forward_outs[self.select_layer]
        image_features = image_forward_outs[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            # if self.merge: # 将特征个数由1024拼接为512
            #     image_features = image_features.reshape(image_features.shape[0],image_features.shape[1]//2,-1)
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
