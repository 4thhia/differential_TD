#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES=0


UNIXTIME=$(date +%s)
ALGO="ppo"
ENV_NAME="hopper"
MAX_BUDGET="2500000"

TD="dtd" # baseline / naive / dtd
NOISE_LVL="0.01"
NOISE_LVL_STR=$(echo $NOISE_LVL | sed 's/\.//g')

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/incumbent/${ENV_NAME}/${TD}/noise_lvl${NOISE_LVL_STR}.sh"


echo "Running experiment with:"
echo "ENV_NAME: $ENV_NAME"
echo "NOISE_LVL: $NOISE_LVL"
echo "TD: $TD"
echo "=============================="


python3 ${ALGO}/main.py \
    hydra.run.dir="configs/logs/${ALGO}/${ENV_NAME}/${TD}/noise${NOISE_LVL_STR}/${UNIXTIME}" \
    env.name=${ENV_NAME} \
    env.noise_lvl=${NOISE_LVL} \
    algorithm=${ALGO}_${TD} \
    algorithm.total_timesteps=${MAX_BUDGET} \
    algorithm.num_env_steps_per_update=${NUM_ENV_STEPS_PER_UPDATE} \
    algorithm.num_epochs_per_update=${NUM_EPOCHS_PER_UPDATE} \
    algorithm.minibatch_size=${MINIBATCH_SIZE} \
    algorithm.model_kwargs.learning_rate=${LEARNING_RATE} \
    algorithm.model_kwargs.mix_ratio=${MIX_RATIO} \
    algorithm.model_kwargs.gae_lambda=${GAE_LAMBDA} \
    algorithm.model_kwargs.clip_range=${CLIP_RANGE} \
    algorithm.model_kwargs.normalize_advantage=${NORMALIZE_ADVANTAGE} \
    algorithm.model_kwargs.vf_coef=${VF_COEF} \
    algorithm.model_kwargs.ent_coef=${ENT_COEF} \
    run_time=${UNIXTIME}