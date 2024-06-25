from typing import Dict, Iterable, List, Literal, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
# TODO(xwjiang): We should port CLIPVisionModel's code over to not depend on
# transformers' impl.
from transformers import CLIPVisionModel, LlavaConfig, Qwen2Config

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
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import ImagePixelData, get_dummy_image_data
from vllm.sequence import SamplerOutput

from .vlm_base import VisionLanguageModelBase

from vllm.logger import init_logger

from vllm.model_executor.models.unicom_encode import UnicomVisionTower
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from PIL import Image

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "model.mm_projector.0": "multi_modal_projector.linear_1",
    "model.mm_projector.2": "multi_modal_projector.linear_2",
    "model.vision_tower": "vision_tower",
    "model.": "language_model."
}

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


class LlavaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, num_channels, height, width)"""


class LlavaImageFeatureInputs(TypedDict):
    type: Literal["image_features"]
    data: torch.Tensor
    """Shape: (batch_size, image_feature_size, hidden_size)"""


LlavaImageInputs = Union[LlavaImagePixelInputs, LlavaImageFeatureInputs]

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result.unsqueeze(0)
        
def _image_pixel_processor(
    data: ImagePixelData,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
    image_mean: Optional[Union[float, List[float]]] = (0.48145466, 0.4578275, 0.40821073)
) -> Dict[str, torch.Tensor]:
    image = data.image
    
    #warm_up input is tensor with matched size, do NOT expand2square
    if isinstance(image, Image.Image):
        data.image = expand2square(image, tuple(int(x*255) for x in image_mean))

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

# @MULTIMODAL_REGISTRY.register_image_pixel_input()
@MULTIMODAL_REGISTRY.register_image_pixel_input(_image_pixel_processor)
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class LlavaQwen2ForConditionalGeneration(VisionLanguageModelBase):

    def __init__(self,
                 config: LlavaConfig,
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
            self.vision_tower = None

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


    def _validate_image_data(self, data: torch.Tensor) -> torch.Tensor:
        if list(data.shape[1:]) != list(
                self.vision_language_config.image_input_shape[1:]):
            raise ValueError(
                f"The expected image tensor shape is batch dimension plus "
                f"{self.vision_language_config.image_input_shape[1:]}. "
                f"You supplied {data.shape}. "
                f"If you are using vLLM's entrypoint, make sure your "
                f"supplied image input is consistent with "
                f"image_input_shape in engine args.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
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

            print(f'input pixel value {pixel_values.shape}')
            return LlavaImagePixelInputs(
                type="pixel_values",
                data=self._validate_image_data(pixel_values),
            )

        if expected_input_type == ImageInputType.IMAGE_FEATURES:
            if pixel_values is not None:
                raise ValueError(
                    "Expected image features but got pixel values")
            if image_features is None:
                return None

            if not isinstance(image_features, torch.Tensor):
                raise ValueError("Incorrect type of image features. "
                                 f"Got type: {type(image_features)}")

            return LlavaImageFeatureInputs(
                type="image_features",
                data=self._validate_image_data(image_features),
            )

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

    def _process_image_pixels(self,
                              inputs: LlavaImagePixelInputs) -> torch.Tensor:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]

        return self._image_pixels_to_features(self.vision_tower, pixel_values)

    def _process_image_input(self,
                             image_input: LlavaImageInputs) -> torch.Tensor:
        if image_input["type"] == "pixel_values":
            assert self.vision_tower is not None
            image_features = self._process_image_pixels(image_input)
        else:
            image_features = image_input["data"]

        logger.info(f'after vit feature shape {image_features.shape}')
        new_feature = self.multi_modal_projector(image_features)
        logger.info(f'after projector feature shape {new_feature.shape}')
        return new_feature
    
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
            print(f'input_ids {input_ids.shape}')
            mask = (input_ids == 32000)
            print(f'image token cnt {mask.sum()}')

            inputs_embeds = self.language_model.embed_tokens(input_ids)
            print(f'inputs_embeds {inputs_embeds.shape}')
            vision_embeddings = self._process_image_input(image_input)
            inputs_embeds = merge_vision_embeddings(
                input_ids, inputs_embeds, vision_embeddings,
                self.vision_language_config.image_token_id)

            print(f'merged_inputs_embeds {inputs_embeds.shape}')

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