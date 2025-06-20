#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES=0 #,1,2


ENV_NAME="humanoid" # "hopper" "halfcheetah" "ant" "humanoid" 
TD="dtd"
RUN_TIME="1747207725" # "1747248866"  "1747173792"  "1747307855" "1747207725"
STEP="1"
NUM_FRAMES="5000"

echo "=== Run #$ENV_NAME ==="

python3 common/render.py "$ENV_NAME" "$TD" "$RUN_TIME" "$STEP" "$NUM_FRAMES"