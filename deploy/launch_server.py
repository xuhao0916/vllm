import argparse
import base64
import json
import os
import signal
import sys
import pathlib
import subprocess
from pathlib import Path
import time
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        required=True,
        help="Config file",
    )
    # docker will build a default one inside
    parser.add_argument(
        "--gpu-mem-fraction",
        type=float,
        default=None,
        help="how much GPU memory should be used, value range 0~1",
    )
    parser.add_argument(
        "--oaip",
        default=None,
        required=False,
        help="path to oaip, default: /app/oaip, set to `none` to disable",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="specify cuda devices to use, like `0,1,2,3`",
    )
    return parser.parse_args()

def get_cmd(world_size, tritonserver, model_repo, http_port, devices):
    cmd = ""
    if devices and len(devices) > 0:
        cmd = "CUDA_VISIBLE_DEVICES=" + devices + " "
    cmd += "mpirun --allow-run-as-root "
    for i in range(world_size):
        cmd += " -n 1 {} --allow-grpc false --grpc-port 8788 --allow-metrics false --http-port {} --model-repository={} --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix{}_ : ".format(
            tritonserver, http_port, model_repo, i
        )
    # print(cmd)
    return cmd

def get_vllm_tp_size(devices):
        total_cnt = torch.cuda.device_count()
        assert total_cnt >= 1, f"can NOT get cuda device"
        if devices:
            devids = [int(s) for s in devices.split(",")]
            assert (
                len(devids) <= total_cnt
            ), f"specified cuda devices {devices} more than total device {total_cnt}"
            tp = len(devids)
        else:
            tp = total_cnt

        print(f'vllm tp size {tp}')
        return tp

def get_vllm_cmd(tp_size, model, address, devices, image_size, feature_size, gpu_memory_utilization):
    cmd = ""
    if devices and len(devices) > 0:
        cmd = "CUDA_VISIBLE_DEVICES=" + devices + " "

    host = address.split(':')[0]
    port = address.split(':')[1]
    input_shape_list = [1,3,image_size[0],image_size[1]]
    input_shape =  ','.join(str(x) for x in input_shape_list)

    cmd += "python -m vllm.entrypoints.openai.api_server \
                --model {} --host {} --port {} --tensor-parallel-size {} \
                --image-input-type pixel_values --image-token-id 151646 \
                --image-input-shape {} --image-feature-size {} \
                --disable-log-requests --gpu-memory-utilization {} \
                --max-model-len 4096 \
                --chat-template /vllm/examples/template_chatml.jinja".format(
                    model,host, port, tp_size, input_shape, feature_size, gpu_memory_utilization
                )
    # print(cmd)
    return cmd

def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit

def divide_to_patches(image_size, patch_size: int):
    patch_cnt = 0
    height, width = image_size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch_cnt += 1
            
    return patch_cnt

def get_llava_feature_size(model, image_size):
    model = model.strip()
    print(f'model {model}, image size {image_size}')
    config_file_path = os.path.join(model, 'config.json')
    assert os.path.exists(config_file_path) , f'{config_file_path} no config.json ?!'
    with open(config_file_path, 'r') as fp:
        js = json.load(fp)

    image_grid_pinpoints = (
        js['image_grid_pinpoints'] if js['image_grid_pinpoints'] is not None
        else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    )

    patch_size = 336
    preprocessor_config = os.path.join(model, 'preprocessor_config.json')
    if os.path.exists(preprocessor_config):
        with open(preprocessor_config, 'r') as fp:
            js = json.load(fp)
            patch_size = js['crop_size']


    best_resolution = select_best_resolution(image_size, image_grid_pinpoints)
    patches_cnt = divide_to_patches(best_resolution, patch_size)
    print(f'patch cnt: {patches_cnt}')
    feature_size = (1 + patches_cnt) * 576
    return feature_size

def decide_mem_fraction():
    if torch.cuda.is_available():
        print("GPU类型：", torch.cuda.get_device_name(0))
        gpu_type = torch.cuda.get_device_name(0).lower()
        if 'T4' in gpu_type:
            print("T4: set frac = 0.9")
            frac = 0.9
        elif '3090' in gpu_type:
            print("3090: set frac = 0.9")
            frac = 0.9
        else:
            print("Default: set frac = 0.9")
            frac = 0.9
    return frac

class Args:
    def __init__(self) -> None:
        pass
    def add(self, key, value):
        setattr(self, key, value)

def main(args):
    processes = {}
    # startup oaip if required, before triton startup
    oaip = args.oaip if args.oaip else "/vllm/oaip"
    if oaip != 'none':
        assert os.path.exists(oaip), f"oaip not found: {oaip}"
        oaip = f"{oaip} -config {args.config}"
        print(">>> ", oaip)
        processes['oaip'] = subprocess.Popen(oaip, shell=True, preexec_fn=os.setsid)

    if args.engine_name.lower() != "vllm":
        print('only support vllm, launch vllm')

    #set timeout for vllm
    if args.timeout is not None and args.timeout > 60: 
        os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = str(args.timeout)

    feature_size = get_llava_feature_size(args.model, args.process_image_size)
    tp_size = get_vllm_tp_size(args.devices)

    gpu_mem_fraction = decide_mem_fraction() if args.gpu_mem_fraction is None or args.gpu_mem_fraction == 0 \
        else args.gpu_mem_fraction

    cmd = get_vllm_cmd(tp_size, args.model, args.address, args.devices, args.process_image_size, feature_size, gpu_mem_fraction)
    print(">>> ", cmd)
    processes['vllm'] = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

    # this program monitor triton/vllm and oaip and any processes started by myself
    # any exit subprocess will terminate everything
    exits = False
    try:
        while not exits:
            for k, v in processes.items():
                ret = v.poll()
                if ret is not None:
                    print(f'subprocess {k} exited: {ret}')
                    exits = True
            time.sleep(3)
    except Exception as e:
        print(e)
        print('now terminating everything')
    except KeyboardInterrupt:
        print("terminating signal received")


    for k, v in processes.items():
        ret = v.poll()
        if ret is None:
            print(f'terminating {k}, pid {v.pid}')
            try:
                os.killpg(v.pid,signal.SIGTERM) 
            except Exception as e:
                print(e)
            print(f'stop terminating')
    

def set_value(js, args, name, defval, *aargs):
    """if args.name is None, load from js[*args], if still not present, use defval
    """
    v = getattr(args, name, None)
    if v is not None:
        return
    if len(aargs) > 0:
        v = js
        for param in aargs:
            v = v.get(param, None)
            if v is None:
                break
    if v is None:
        v = defval
    setattr(args, name, v)
    
if __name__ == "__main__":
    print(f">> cmd line: {' '.join(sys.argv)}")
    args = parse_arguments()
    assert len(args.config) > 0 and os.path.exists(args.config)
    with open(args.config, 'r') as fp:
        js = json.load(fp)
    def setval(*a):
        set_value(js, args, *a)
    setval("oaip", None, "sys", "oaip")
    setval("devices", None, "sys", "devices")
    setval("oaip", None, "sys", "oaip")
    setval("engine_name", None, "engine", "name")

    #vllm args
    print(f'select vllm engine')
    setval("address", None, "engine", "vllm", "address")
    setval("timeout", 60, "engine", "vllm", "timeout")
    setval("model", None, "engine", "vllm", "model")
    setval("gpu_mem_fraction", None, "engine", "vllm", "gpuMemFraction")
    setval("process_image_size", [1920, 1080], "engine", "vllm", "process_image_size")
    assert args.model is not None and len(args.model) > 0
    assert args.address is not None and len(args.address) > 0

    main(args)
    