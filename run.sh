#!/bin/bash

USER="your_username"

docker container run --rm -it \
    -u $(id -u):$(id -g) \
    -v $(pwd):/home/${USER}/workdir \
    --shm-size=32g \
    --gpus=all \
    "${USER}_dtd"