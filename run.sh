#!/bin/bash

USER="4thhia"

docker container run --rm -it \
    --net=host --group-add=sudo --group-add=video \
    -u $(id -u):$(id -g) \
    -v $(pwd):/home/${USER}/workdir \
    --shm-size=32g \
    --gpus=all \
    settai_jax