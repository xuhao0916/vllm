#!/bin/bash
set -ex

url="dockerhub.deepglint.com/lse/vllm"
tag="0.0.2"
extra_args=""

if [ ! -z $http_proxy ]; then
    extra_args="--build-arg proxy_val=$http_proxy"
    echo $extra_args
fi

DOCKER_BUILDKIT=1 docker build \
 --progress=plain \
 -t $url:$tag \
 $extra_args \
 --build-arg IMAGE=$url:$tag \
 -f Dockerfile .


