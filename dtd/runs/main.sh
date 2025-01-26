#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.85
export CUDA_VISIBLE_DEVICES=0 #,1,2


UNIXTIME=$(date +%s)
ALGO="ppo"
ENV_NAME="ant"
MAX_BUDGET="2500000"

TD="mix" # baseline / sde / mix
NOISE_LVL="0.01"
NOISE_LVL_STR=$(echo $NOISE_LVL | sed 's/\.//g')

source ./incumbent/${ENV_NAME}/${TD}/noise${NOISE_LVL_STR}.sh


python3 ${ALGO}/main.py \
    hydra.run.dir="configs/logs/${ALGO}/${ENV_NAME}/${TD}/${UNIXTIME}" \
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