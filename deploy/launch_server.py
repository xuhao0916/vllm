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

def get_vllm_cmd(tp_size, model, address, devices, gpu_memory_utilization, max_model_len=None, template=None):
    cmd = ""
    if devices and len(devices) > 0:
        cmd = "CUDA_VISIBLE_DEVICES=" + devices + " "

    host = address.split(':')[0]
    port = address.split(':')[1]

    chat_template = (
        template if template is not None
        else "/vllm/examples/template_chatml.jinja"
    )
    cmd += "python -m vllm.entrypoints.openai.api_server \
                --model {} --host {} --port {} --tensor-parallel-size {} \
                --disable-log-requests --gpu-memory-utilization {} \
                --chat-template {}".format(
                    model, host, port, tp_size, gpu_memory_utilization, chat_template
                )
    if max_model_len and max_model_len != 0:
        cmd += " --max-model-len {}".format(max_model_len)

    # print(cmd)
    return cmd

def decide_mem_fraction():
    assert torch.cuda.is_available(), f"cuda is not available"
    gpu_type = torch.cuda.get_device_name(0)
    print("GPU类型：", gpu_type)
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

    tp_size = get_vllm_tp_size(args.devices)

    gpu_mem_fraction = decide_mem_fraction() if args.gpu_mem_fraction is None or args.gpu_mem_fraction == 0 \
        else args.gpu_mem_fraction

    cmd = get_vllm_cmd(tp_size, args.model, args.address, args.devices, gpu_mem_fraction, args.max_model_len, args.chat_template)
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
    setval("max_model_len", None, "engine", "vllm", "maxLen")
    setval("chat_template", None, "engine", "vllm", "template")

    assert args.model is not None and len(args.model) > 0
    assert args.address is not None and len(args.address) > 0

    main(args)
    
