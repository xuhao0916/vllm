#!/bin/bash
set -ex



root=$(dirname $(dirname $(realpath $0)))

url=$(jq -r .url $root/deploy/VERSION.json)
tag=$(jq -r .tag $root/deploy/VERSION.json)
base_image=$(jq -r .base $root/deploy/VERSION.json)

pushd $root
# get commit ids
vllm_commitid=$(git rev-parse HEAD)
pushd oaip
oaip_commitid=$(git rev-parse HEAD)
popd

# prepare docker workspace
rm -rf tmp && mkdir tmp

# build oaip, it's ok to build on host and copy to image
pushd oaip
go generate ./...
popd
pushd oaip/cmd/oaip
go build -o $root/tmp/oaip .
popd

# copy files to workspace
pushd tmp
cp -r $root/vllm .
cp -r $root/examples .
cp -r $root/deploy .
cp -r $root/oaip/thirdparty .

extra_args=""
if [ ! -z $http_proxy ]; then
    extra_args="--build-arg proxy_val=$http_proxy"
fi

# build
DOCKER_BUILDKIT=1 docker build \
 --progress=plain \
 -t $url:$tag \
 $extra_args \
 --build-arg IMAGE=$url:$tag \
 --build-arg BASE_IMAGE=$base_image \
 --build-arg VLLM_COMMITID=$vllm_commitid \
 --build-arg OAIP_COMMITID=$oaip_commitid \
 -f $root/deploy/Dockerfile.deploy .
popd

rm -rf $root/tmp
