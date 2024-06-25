from typing import Dict, Iterable, List, Literal, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from PIL import Image
import math

# TODO(xwjiang): We should port CLIPVisionModel's code over to not depend on
# transformers' impl.
from transformers import CLIPVisionModel, LlavaNextConfig, Qwen2Config
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape, unpad_image)
from typing_extensions import NotRequired

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, VisionLanguageConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalData
from vllm.multimodal.image import ImagePixelData, get_dummy_image_data
from vllm.sequence import SamplerOutput, SequenceData

from .vlm_base import VisionLanguageModelBase

from vllm.logger import init_logger

from vllm.model_executor.models.unicom_encode import UnicomVisionTower
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
from vllm.model_executor.layers.layernorm import RMSNorm

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "model.mm_projector.0": "multi_modal_projector.linear_1",
    "model.mm_projector.2": "multi_modal_projector.linear_2",
    "model.vision_tower": "vision_tower",
    "model.": "language_model."
}

class LlavaNextImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, 1 + num_patches, num_channels, height, width)"""

    image_sizes: NotRequired[torch.Tensor]
    """Shape: (batch_size, 2)"""


class LlavaNextImageFeatureInputs(TypedDict):
    type: Literal["image_features"]
    data: torch.Tensor
    """Shape: (batch_size, 1 + num_patches, image_feature_size, hidden_size)"""

    image_sizes: NotRequired[torch.Tensor]
    """Shape: (batch_size, 2)"""

# TODO(xwjiang): Run benchmark and decide if TP.
class LlavaMultiModalProjector(nn.Module):

    def __init__(self, vision_hidden_size: int, text_hidden_size: int,
                 projector_hidden_act: str):
        super().__init__()
        logger.info(f'vit hidden size {vision_hidden_size}')
        logger.info(f'text hidden size {text_hidden_size}')
        self.linear_1 = nn.Linear(vision_hidden_size,
                                  text_hidden_size,
                                  bias=True)
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = nn.Linear(text_hidden_size,
                                  text_hidden_size,
                                  bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def merge_vision_embeddings(input_ids: torch.Tensor,
                            inputs_embeds: torch.Tensor,
                            vision_embeddings: torch.Tensor,
                            image_token_id: int) -> torch.Tensor:
    """In place merges in vision_embeddings with inputs_embeds."""
    # logger.info(f'image_token_id {image_token_id}')
    # logger.info(f'input_ids {input_ids}')
    mask = (input_ids == image_token_id)
    logger.info(f'vision_embeddings shape {vision_embeddings.shape}')
    image_feature_size = vision_embeddings.shape[0] * vision_embeddings.shape[1]
    if mask.sum() != image_feature_size:
        raise ValueError(f"image_feature_size should be {image_feature_size}, "
                         f"but found: {mask.sum()}")

    inputs_embeds[mask] = vision_embeddings.view(image_feature_size,
                                                 vision_embeddings.shape[-1])

    return inputs_embeds


class LlavaNextImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, 1 + num_patches, num_channels, height, width)"""

    image_sizes: NotRequired[torch.Tensor]
    """Shape: (batch_size, 2)"""


class LlavaNextImageFeatureInputs(TypedDict):
    type: Literal["image_features"]
    data: torch.Tensor
    """Shape: (batch_size, 1 + num_patches, image_feature_size, hidden_size)"""

    image_sizes: NotRequired[torch.Tensor]
    """Shape: (batch_size, 2)"""


LlavaNextImageInputs = Union[LlavaNextImagePixelInputs,
                             LlavaNextImageFeatureInputs]


def _get_dummy_image_data(
    seq_len: int,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
) -> Tuple[SequenceData, MultiModalData]:
    seq_data, fake_mm_data = get_dummy_image_data(seq_len, model_config,
                                                  vlm_config)

    config_input_type = vlm_config.image_input_type
    ImageInputType = VisionLanguageConfig.ImageInputType

    if config_input_type == ImageInputType.PIXEL_VALUES:
        _, c, h, w = vlm_config.image_input_shape
        mode = {1: "L", 3: "RGB"}[c]
        fake_mm_data = ImagePixelData(Image.new(mode, (w, h), color=0))

    return seq_data, fake_mm_data

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches
    
def process_anyres_image(image,  grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        raise ValueError(f"grid_pinpoints type should be list")

    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, 336)

    image_original_resize = image.resize((336, 336))

    image_patches = [image_original_resize] + patches
    
    return image_patches

def _image_pixel_processor_bk(
    data: ImagePixelData,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig
) -> Dict[str, torch.Tensor]:
    image = data.image
    
    if isinstance(image, Image.Image):
        images = process_anyres_image(image, model_config.hf_config.image_grid_pinpoints)
    else:
        assert 0, "should be image"  

    # process each data.image
    processed_images = []    
    for img in images:
        each_data = ImagePixelData
        each_data.image = img
        processed_img = MULTIMODAL_REGISTRY._get_plugin_for_data_type(ImagePixelData) \
        ._default_input_processor(each_data, model_config, vlm_config)
        processed_images.append(processed_img)
    
    return processed_images

def _image_pixel_processor(
    data: ImagePixelData,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
) -> Dict[str, torch.Tensor]:
    image = data.image

    if isinstance(image, torch.Tensor):
        pixel_values = image.to(model_config.dtype)
        batch_size, _, _, h, w = pixel_values.shape
        image_sizes = torch.tensor([(w, h) for _ in range(batch_size)])

        return {"pixel_values": pixel_values, "image_sizes": image_sizes}

    # Temporary patch before dynamic number of image tokens is supported
    _, _, h, w = vlm_config.image_input_shape
    if (w, h) != (image.width, image.height):
        logger.warning(
            "Dynamic image shape is currently not supported. "
            "Resizing input image to (%d, %d).", w, h)

        data.image = image.resize((w, h))

    return MULTIMODAL_REGISTRY._get_plugin_for_data_type(ImagePixelData) \
        ._default_input_processor(data, model_config, vlm_config)


class LlavaQwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )
        print(f'embed_tokens {self.embed_tokens}')
        print(f'Qwen2 config {config}')
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, cache_config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            # print(f'inputs_embeds {inputs_embeds.shape}: {inputs_embeds}')
            hidden_states = inputs_embeds
        else:
            assert 0, "must set input embeds"
            hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

@MULTIMODAL_REGISTRY.register_image_pixel_input(_image_pixel_processor)
@MULTIMODAL_REGISTRY.register_dummy_data(_get_dummy_image_data)
class LlavaNextQwen2ForConditionalGeneration(VisionLanguageModelBase):

    def __init__(self,
                 config: LlavaNextConfig,
                 vision_language_config: VisionLanguageConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__(vision_language_config)

        self.config = config

        if self.vision_language_config.image_input_type == (
                VisionLanguageConfig.ImageInputType.PIXEL_VALUES):
            vision_tower = getattr(config, 'mm_vision_tower',
                                   getattr(config, 'vision_tower', None))
            self.vision_tower = UnicomVisionTower(vision_tower, config)
        else:
            raise TypeError("Image features are not supported by DG-VLM-HD")

        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=config.mm_hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act)

        self.quant_config = quant_config
        self.language_model = LlavaQwen2Model(config, cache_config, quant_config)

        if config.tie_word_embeddings:
            self.lm_head_weight = self.model.embed_tokens.weight
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size)
            self.lm_head_weight = self.lm_head.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size))

        logger.info(f'vision_config: {self.config.vision_config}')

    def _validate_image_pixels(self, data: torch.Tensor) -> torch.Tensor:
        _, num_channels, _, _ = self.vision_language_config.image_input_shape

        # Note that this is different from that of vLLM vision_language_config
        # since the image is resized by the HuggingFace preprocessor
        height = width = self.config.vision_config.image_size

        if list(data.shape[2:]) != [num_channels, height, width]:
            raise ValueError(
                f"The expected image tensor shape is batch dimension plus "
                f"num_patches plus {[num_channels, height, width]}. "
                f"You supplied {data.shape}. "
                f"If you are using vLLM's entrypoint, make sure your "
                f"supplied image input is consistent with "
                f"image_input_shape in engine args.")

        return data

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        if list(data.shape[1:]) != [2]:
            raise ValueError(
                f"The expected image sizes shape is batch dimension plus "
                f"{[2]}. You supplied {data.shape}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaNextImageInputs]:
        # logger.info(f'kwargs: {kwargs}')
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_features = kwargs.pop("image_features", None)

        expected_input_type = self.vision_language_config.image_input_type
        ImageInputType = VisionLanguageConfig.ImageInputType

        if expected_input_type == ImageInputType.PIXEL_VALUES:
            if image_features is not None:
                raise ValueError(
                    "Expected pixel values but got image features")
            if pixel_values is None:
                return None

            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, torch.Tensor):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            return LlavaNextImagePixelInputs(
                type="pixel_values",
                data=self._validate_image_pixels(pixel_values),
                image_sizes=self._validate_image_sizes(image_sizes),
            )

        assert expected_input_type != ImageInputType.IMAGE_FEATURES, (
            "Failed to validate this at initialization time")
        
        return None

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(self, vision_tower: CLIPVisionModel,
                                  pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = vision_tower(pixel_values.to(vision_tower.device))

        #has been select in vision_tower
        return image_features

    def _merge_image_patch_embeddings(self, image_size: torch.Tensor,
                                      patch_embeddings: torch.Tensor, *,
                                      strategy: str) -> torch.Tensor:
        # Based on: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            orig_width, orig_height = image_size
            height = width = self.config.vision_config.image_size \
                // self.config.vision_config.patch_size

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the "
                    "image size.")

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # image_aspect_ratio == "anyres"
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    (orig_width, orig_height),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                other_patch_embeds = other_patch_embeds \
                    .view(num_patch_width, num_patch_height, height, width, -1)

                if "unpad" in strategy:
                    other_patch_embeds = other_patch_embeds \
                        .permute(4, 0, 2, 1, 3).contiguous() \
                        .flatten(1, 2).flatten(2, 3)
                    other_patch_embeds = unpad_image(other_patch_embeds,
                                                     image_size)
                    other_patch_embeds = torch.cat((
                        other_patch_embeds,
                        self.image_newline[:, None, None] \
                            .expand(*other_patch_embeds.shape[:-1], 1) \
                            .to(other_patch_embeds.device),
                    ), dim=-1)
                    other_patch_embeds = other_patch_embeds \
                        .flatten(1, 2).transpose(0, 1)
                else:
                    other_patch_embeds = other_patch_embeds \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .flatten(0, 3)

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0)
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (base_patch_embeds,
                         self.image_newline[None] \
                            .to(base_patch_embeds.device)
                    ), dim=0)
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")
    
    def _process_image_pixels(
            self, inputs: LlavaNextImagePixelInputs) -> torch.Tensor:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]
        logger.info(f'image input shape {pixel_values.shape}')

        b, num_patches, c, h, w = pixel_values.shape
        stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)

        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values)

        return stacked_image_features.view(b, num_patches,
                                           *stacked_image_features.shape[-2:])


    def _process_image_input(
            self, image_input: LlavaNextImageInputs) -> torch.Tensor:
        if image_input["type"] == "pixel_values":
            assert self.vision_tower is not None
            image_features = self._process_image_pixels(image_input)
        else:
            image_features = image_input["data"]

        logger.info(f'after vit feature shape {image_features.shape}')
        patch_embeddings = self.multi_modal_projector(image_features)
        logger.info(f'after projector feature shape {patch_embeddings.shape}')

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = image_input["data"].shape[0]
            vision_config = self.config.vision_config
            default_width = default_height = vision_config.image_size
            image_sizes = torch.as_tensor([[default_width, default_height]
                                           for _ in range(batch_size)])

        patch_strategy = self.config.mm_patch_merge_type
        logger.info(f'mm_patch_merge_type {patch_strategy}')
        merged_patch_embeddings = [
            self._merge_image_patch_embeddings(image_sizes[i],
                                               patch_features,
                                               strategy=patch_strategy)
            for i, patch_features in enumerate(patch_embeddings)
        ]

        return torch.stack(merged_patch_embeddings, dim=0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> SamplerOutput:
        """Run forward pass for Llava 1.5.

        One key thing to understalnd is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.
        Concretely, consider a text prompt:
        "<image>\nUSER: What's the content of the image?\nASSISTANT:".
        Tokenizer outputs:
        [1, 32000, 29871, 13, 11889, 29901, 1724, 29915, 29879, 278,
        2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566, 29901].
        The to-be-inserted image has a size of 576 (24 * 24) along the context
        length dimension.
        `input_ids` is thus [1, 32000, ..., 32000, 29871, 13, 11889, 29901,
        1724, 29915, 29879, 278, 2793, 310, 278, 1967, 29973, 13, 22933,
        9047, 13566, 29901].
        There will be 576 `32000` in the `input_ids`.
        (32000 is the token id for `<image>`.)

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        The model takes two types of image inputs:
        PIXEL_VALUES and IMAGE_FEATURES.
        The following shows how each maps to huggingface implementation.
        PIXEL_VALUES:
        - https://github.com/huggingface/transformers/blob/07bdbeb/src/transformers/models/llava/modeling_llava.py#L353
        IMAGE_FEATURES:
        - https://github.com/huggingface/transformers/blob/07bdbeb/src/transformers/models/llava/modeling_llava.py#L430
        before going through the multi modal projector.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: For PIXEL_VALUES, expects a batch with shape
                [1, 3, 336, 336].
            image_features: For IMAGE_FEATURES, expects a batch with shape
                [1, 576, 1024].
        """
        image_input = self._parse_and_validate_image_input(**kwargs)

        if image_input is not None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
            logger.info(f'inputs_embeds {inputs_embeds.shape}')
            vision_embeddings = self._process_image_input(image_input)
            inputs_embeds = merge_vision_embeddings(
                input_ids, inputs_embeds, vision_embeddings,
                self.vision_language_config.image_token_id)

            input_ids = None
        else:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
            input_ids = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            kv_caches,
                                            attn_metadata,
                                            inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)

            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                # print(f'dict name {name} {loaded_weight.size()}')

                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)