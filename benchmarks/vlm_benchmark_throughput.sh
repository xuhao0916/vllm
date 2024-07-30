#!/bin/bash
root=$(dirname $(dirname $(realpath $0)))

rm $root/throughput-*.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 1 --input-len 512 --output-len 128   --output-json throughput-1-512-128.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 1 --input-len 512 --output-len 1024  --output-json throughput-1-512-1024.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 1 --input-len 2048 --output-len 128  --output-json throughput-1-2048-128.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 1 --input-len 2048 --output-len 1024 --output-json throughput-1-2048-1024.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 1 --input-len 4096 --output-len 128  --output-json throughput-1-4096-128.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 1 --input-len 4096 --output-len 1024 --output-json throughput-1-4096-1024.json

CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 8 --input-len 512 --output-len 128   --output-json throughput-8-512-128.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 8 --input-len 512 --output-len 1024  --output-json throughput-8-512-1024.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 8 --input-len 2048 --output-len 128  --output-json throughput-8-2048-128.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 8 --input-len 2048 --output-len 1024 --output-json throughput-8-2048-1024.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 8 --input-len 4096 --output-len 128  --output-json throughput-8-4096-128.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 8 --input-len 4096 --output-len 1024 --output-json throughput-8-4096-1024.json

CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 16 --input-len 512 --output-len 128   --output-json throughput-16-512-128.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 16 --input-len 512 --output-len 1024  --output-json throughput-16-512-1024.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 16 --input-len 2048 --output-len 128  --output-json throughput-16-2048-128.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 16 --input-len 2048 --output-len 1024 --output-json throughput-16-2048-1024.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 16 --input-len 4096 --output-len 128  --output-json throughput-16-4096-128.json
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/vlm_benchmark_throughput.py --num-prompts 16 --input-len 4096 --output-len 1024 --output-json throughput-16-4096-1024.json
