from typing import Dict, Iterable, List, Literal, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionConfig, LlavaNextConfig, Qwen2Config
from transformers.models.llava_next.modeling_llava_next import (
    image_size_to_num_patches, get_anyres_image_grid_shape, unpad_image)
from typing_extensions import NotRequired

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig

from vllm.distributed import (get_pp_group, get_pp_indices,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)

from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, BatchedTensors
from vllm.sequence import IntermediateTensors, SamplerOutput

from vllm.logger import init_logger

from vllm.model_executor.models.unicom_encode import UnicomVisionTower
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
from vllm.model_executor.layers.layernorm import RMSNorm

from .clip import (dummy_image_for_clip, dummy_seq_data_for_clip,
                   get_clip_patch_grid_length, input_processor_for_clip)

from .interfaces import SupportsVision
from .llava import LlavaMultiModalProjector
from .utils import merge_vision_embeddings

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "model.mm_projector.0": "multi_modal_projector.linear_1",
    "model.mm_projector.2": "multi_modal_projector.linear_2",
    "model.vision_tower": "vision_tower",
    "model.": "language_model."
}

# Result in the max possible feature size (2x2 grid of 336x336px tiles)
MAX_IMAGE_FEATURE_SIZE_HEIGHT = MAX_IMAGE_FEATURE_SIZE_WIDTH = 448


class LlavaNextImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: BatchedTensors
    """
    Shape: `(batch_size, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different for each batch, in which case
    the data is passed as a list instead of a batched tensor.
    """

    image_sizes: NotRequired[torch.Tensor]
    """
    Shape: `(batch_size, 2)`

    This should be in `(height, width)` format.
    """

LlavaNextImageInputs = LlavaNextImagePixelInputs

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
        logger.info(f'embed_tokens {self.embed_tokens}')
        
        self.start_layer, self.end_layer = get_pp_indices(
            config.num_hidden_layers,
            get_pp_group().rank_in_group,
            get_pp_group().world_size)
        self.layers = nn.ModuleList(
            [nn.Identity() for _ in range(self.start_layer)] + [
                Qwen2DecoderLayer(config=config,
                                  cache_config=cache_config,
                                  quant_config=quant_config)
                for _ in range(self.start_layer, self.end_layer)
            ] + [
                nn.Identity()
                for _ in range(self.end_layer, config.num_hidden_layers)
            ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                # print(f'inputs_embeds {inputs_embeds.shape}: {inputs_embeds}')
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

def get_image_feature_size(
    hf_config: LlavaNextConfig,
    *,
    input_height: int,
    input_width: int,
) -> int:
    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        num_patches = get_clip_patch_grid_length(
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
        )
        base_feature_size = num_patches * num_patches

        # logger.info(f'grid_pinpoints {hf_config.image_grid_pinpoints}')
        patches_count = image_size_to_num_patches(
            image_size=(input_height, input_width),
            grid_pinpoints=hf_config.image_grid_pinpoints,
            patch_size=vision_config.image_size,
        )
        
        return patches_count * base_feature_size

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)

def get_max_image_tokens(ctx: InputContext):

    return get_image_feature_size(
        ctx.get_hf_config(LlavaNextConfig),
        input_height=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
        input_width=MAX_IMAGE_FEATURE_SIZE_WIDTH,
    )

def dummy_data(ctx: InputContext, seq_len: int):
    hf_config = ctx.get_hf_config(LlavaNextConfig)
    # TODO read from tokenizer
    hf_config.image_token_index = 151646
    vision_config = hf_config.vision_config

    image_feature_size = get_max_image_tokens(ctx)

    if isinstance(vision_config, CLIPVisionConfig):
        seq_data = dummy_seq_data_for_clip(
            vision_config,
            seq_len,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

        mm_data = dummy_image_for_clip(
            vision_config,
            image_width_override=MAX_IMAGE_FEATURE_SIZE_WIDTH,
            image_height_override=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
        )

        return seq_data, mm_data

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)

def input_processor(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(LlavaNextConfig)
    hf_config.image_token_index = 151646
    vision_config = hf_config.vision_config
    # logger.info(f'vision config {vision_config}')

    image_data = multi_modal_data["image"]
    if isinstance(image_data, Image.Image):
        width, height = image_data.size

        image_feature_size = get_image_feature_size(
            hf_config,
            input_height=height,
            input_width=width,
        )
        # logger.info(f'image_feature_size: {image_feature_size}')
    elif isinstance(image_data, torch.Tensor):
        raise NotImplementedError("Embeddings input is not supported yet")
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        return input_processor_for_clip(
            model_config,
            vision_config,
            llm_inputs,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)

@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data)
@INPUT_REGISTRY.register_input_processor(input_processor)
class LlavaNextQwen2ForConditionalGeneration(nn.Module, SupportsVision):

    def __init__(self,
                 config: LlavaNextConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config

        vision_tower = getattr(config, 'mm_vision_tower',getattr(config, 'vision_tower', None))
        last_slash_index = vision_tower.rfind('/')
        if last_slash_index != -1:
            vision_tower = config._name_or_path + vision_tower[last_slash_index:]
        else:
            vision_tower = config._name_or_path + '/' + vision_tower

        self.vision_tower = UnicomVisionTower(vision_tower, config)
        logger.info(f'vision_tower {vision_tower}')

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

    def _validate_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[1:])

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values in each batch element "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        # logger.info(f'image_sizes data shape: {data.shape}, {data}')

        if list(data.shape[1:]) != [2]:
            raise ValueError(
                f"The expected image sizes shape is batch dimension plus "
                f"{[2]}. You supplied {data.shape}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaNextImagePixelInputs]:
        # logger.info(f'kwargs: {kwargs}')
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)

        if pixel_values is None:
            return None

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        if not isinstance(image_sizes, torch.Tensor):
            raise ValueError("Incorrect type of image sizes. "
                             f"Got type: {type(image_sizes)}")

        return LlavaNextImagePixelInputs(
            type="pixel_values",
            data=self._validate_pixel_values(pixel_values),
            image_sizes=self._validate_image_sizes(image_sizes),
        )


    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(self, vision_tower: UnicomVisionTower,
                                  pixel_values: torch.Tensor) -> torch.Tensor:
        logger.info(f'pixel_values shape {pixel_values.shape} {pixel_values.dtype}')
        image_features = vision_tower(pixel_values.to(vision_tower.device))
        logger.info(f'vit feature shape {image_features.shape} {image_features.dtype}')

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
        self,
        inputs: LlavaNextImagePixelInputs,
    ) -> BatchedTensors:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]
        logger.info(f'pixel_values shape {pixel_values.shape}, {pixel_values.dtype}')
        if isinstance(pixel_values, torch.Tensor):
            b, num_patches, c, h, w = pixel_values.shape
            stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)
            stacked_image_features = self._image_pixels_to_features(
                self.vision_tower, stacked_pixel_values)
            stacked_patch_embeddings = self.multi_modal_projector(
                stacked_image_features)

            return stacked_patch_embeddings.view(
                b, num_patches, *stacked_patch_embeddings.shape[1:])

        num_patches_per_batch = [v.shape[0] for v in pixel_values]
        stacked_pixel_values = torch.cat(pixel_values)
        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values)

        return [
            self.multi_modal_projector(image_features) for image_features in
            torch.split(stacked_image_features, num_patches_per_batch)
        ]


    def _process_image_input(
            self, image_input: LlavaNextImageInputs) -> BatchedTensors:

        patch_embeddings = self._process_image_pixels(image_input)
        logger.info(f'after projector feature shape {patch_embeddings.shape}')

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = len(image_input["data"])
            vision_config = self.config.vision_config
            default_height = default_width = vision_config.image_size
            image_sizes = torch.as_tensor([[default_height, default_width]
                                           for _ in range(batch_size)])

        patch_strategy = self.config.mm_patch_merge_type
        logger.info(f'mm_patch_merge_type {patch_strategy}')
        merged_patch_embeddings = [
            self._merge_image_patch_embeddings(image_sizes[i],
                                               patch_features_batch,
                                               strategy=patch_strategy)
            for i, patch_features_batch in enumerate(patch_embeddings)
        ]

        return torch.stack(merged_patch_embeddings, dim=0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> SamplerOutput:
        """Run forward pass for LlaVA-NeXT.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"A chat between a curious human and an artificial intelligence
        assistant. The assistant gives helpful, detailed, and polite answers to
        the human's questions.
        USER: <image>\\nWhat is shown in this image? ASSISTANT:"`.

        Tokenizer outputs:
        `[1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082, 20255,
        29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568,
        6089, 304, 278, 5199, 29915, 29879, 5155, 29889, 3148, 1001, 29901,
        29871, 32000, 13, 5618, 338, 4318, 297, 445, 1967, 29973, 319, 1799,
        9047, 13566, 29901]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends 
        additional image tokens (denoted as `32000`), resulting in:
        `[1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082, 20255,
        29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568,
        6089, 304, 278, 5199, 29915, 29879, 5155, 29889, 3148, 1001, 29901,
        29871, 32000, ..., 32000, 13, 5618, 338, 4318, 297, 445, 1967, 29973,
        319, 1799, 9047, 13566, 29901]`.

        Unlike in LLaVA-1.5, the number of image tokens inputted to the language
        model depends on the original size of the input image. Including the
        original image token in the input, the required number of image tokens
        is given by :func:`get_llava_next_image_feature_size`.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: The pixels in each grid patch for each input image.
            image_sizes: The original `(height, width)` for each input image.
        
        See also:
            :class:`LlavaNextImageInputs`
        """
        image_input = self._parse_and_validate_image_input(**kwargs)

        if image_input is not None:
            vision_embeddings = self._process_image_input(image_input)
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)

            inputs_embeds = merge_vision_embeddings(
                input_ids, inputs_embeds, vision_embeddings,
                self.config.image_token_index)

            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            kv_caches,
                                            attn_metadata,
                                            None,
                                            inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
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