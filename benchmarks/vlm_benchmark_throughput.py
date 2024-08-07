"""Benchmark offline inference throughput."""
import argparse
import json
import random
import os
import time
import threading
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import FlexibleArgumentParser
from vllm.model_executor.models.dg_vlm_hd import get_image_feature_size
from vllm.model_executor.models.clip import dummy_image_for_clip
from vllm.inputs import TokensPrompt

# Result in the max possible feature size (2x2 grid of 336x336px tiles)
MAX_IMAGE_FEATURE_SIZE_HEIGHT = MAX_IMAGE_FEATURE_SIZE_WIDTH = 448

# 存储 GPU 显存使用的最大值
max_used = {}
max_used_lock = threading.Lock()

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def run_vllm(
    loop:int,
    num_prompts:int,
    input_len:int,
    output_len:int,
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
    load_format: str = EngineArgs.load_format,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=load_format
    )

    hf_config = llm.llm_engine.get_model_config().hf_config
    image_feature_size = get_image_feature_size(hf_config,         
                                                input_height=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
                                                input_width=MAX_IMAGE_FEATURE_SIZE_WIDTH)
    print(f'hf_config.image_token_index: {hf_config.image_token_index}, image_feature_size {image_feature_size}')
    
    mm_data= None
    if(args.input_len > image_feature_size):
        prompt_token_ids = [hf_config.image_token_index] + [random.randint(10, 10000) for _ in range(input_len - 1)]
        mm_data = dummy_image_for_clip(
            hf_config.vision_config,
            image_width_override=MAX_IMAGE_FEATURE_SIZE_WIDTH,
            image_height_override=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
        )
    else:
        prompt_token_ids = [random.randint(10, 10000) for _ in range(input_len)]

    tokens_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, multi_modal_data=mm_data)
    prompts = [tokens_prompt] * num_prompts 

    # Add the requests to the engine.
    sampling_param = SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
            )
    sampling_params = [sampling_param] * num_prompts

    start = time.perf_counter()
    for _ in range(loop):
        llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return (end - start) / loop

# 获取 NVIDIA GPU 显存信息
def get_gpu_memory_info(gpus):
    try:
        output = os.popen('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,nounits,noheader').read().strip()
        gpu_info = []
        for line in output.split('\n'):
            index, used, total = line.split(',')
            if int(index) in gpus:
                gpu_info.append((int(index), int(used), int(total)))
        return gpu_info
    except:
        return []

# 后台线程，统计 GPU 显存使用情况
def monitor_gpu(gpu_id):
    gpus = list(map(int, gpu_id.split(",")))
    print(f'used gpus {gpus}')
    while True:
        gpu_info = get_gpu_memory_info(gpus)
        with max_used_lock:
            for index, used, total in gpu_info:
                if index not in max_used or used > max_used[index][0]:
                    max_used[index] = (used, total)
        time.sleep(2)  # 每 2 秒记录一次

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # 启动后台线程监控 GPU 显存使用情况
    monitor_thread = threading.Thread(target=monitor_gpu,args=(args.gpu_id,))
    monitor_thread.daemon = True
    monitor_thread.start()

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            args.loop, args.num_prompts, args.input_len, args.output_len, args.model, args.tokenizer, args.quantization,
            args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
            args.trust_remote_code, args.dtype, args.max_model_len,
            args.enforce_eager, args.kv_cache_dtype,
            args.quantization_param_path, args.device,
            args.enable_prefix_caching, args.enable_chunked_prefill,
            args.max_num_batched_tokens, args.distributed_executor_backend,
            args.gpu_memory_utilization, args.download_dir, args.load_format)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = args.output_len * args.num_prompts
    print(f"Throughput: {args.num_prompts / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")


    with max_used_lock:
        for index, (used, total) in max_used.items():
            print(f"GPU {index}: {used} MB / {total} MB")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": args.num_prompts,
            "total_num_tokens": total_num_tokens,
            "requests_per_second": args.num_prompts / elapsed_time,
            "tokens_per_second": round(total_num_tokens / elapsed_time, 2),
        }
        with max_used_lock:
            for index, (used, total) in max_used.items():
                results[f'GPU {index}']=used
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="/models/llava-7b-unicom-qwen2-hd-800k/")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=2)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--loop", type=int, default=10)
    parser.add_argument("--gpu_id", type=str, default="0,1",help="gpu id")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "openvino", "tpu", "xpu"],
        help='device type for vLLM execution, supporting CUDA, OpenVINO and '
        'CPU.')
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument(
        '--load-format',
        type=str,
        default=EngineArgs.load_format,
        choices=[
            'auto', 'pt', 'safetensors', 'npcache', 'dummy', 'tensorizer',
            'bitsandbytes'
        ],
        help='The format of the model weights to load.\n\n'
        '* "auto" will try to load the weights in the safetensors format '
        'and fall back to the pytorch bin format if safetensors format '
        'is not available.\n'
        '* "pt" will load the weights in the pytorch bin format.\n'
        '* "safetensors" will load the weights in the safetensors format.\n'
        '* "npcache" will load the weights in pytorch format and store '
        'a numpy cache to speed up the loading.\n'
        '* "dummy" will initialize the weights with random values, '
        'which is mainly for profiling.\n'
        '* "tensorizer" will load the weights using tensorizer from '
        'CoreWeave. See the Tensorize vLLM Model script in the Examples'
        'section for more information.\n'
        '* "bitsandbytes" will load the weights using bitsandbytes '
        'quantization.\n')
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")

    main(args)
