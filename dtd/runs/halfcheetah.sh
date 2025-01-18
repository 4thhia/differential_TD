#!/bin/bash
# tuning parameters: lmb.lr / regularization coef / weightunet.lr

export XLA_PYTHON_CLIENT_MEM_FRACTION=.70
export CUDA_VISIBLE_DEVICES=1


UNIXTIME=$(date +%s)
MAX_BUDGET="1e7"
MIN_BUDGET="1e5"

ENV_NAME="halfcheetah" # ant / halfcheetah / humanoid
TD="mix" # baseline / sde / mix
NOISE_LVL="0.05"
NOISE_LVL_STR=$(echo $NOISE_LVL | sed 's/\.//g')

ALGO="ppo"

python3 ${ALGO}/tune.py --multirun \
    hydra.run.dir="configs/logs/${ALGO}/${ENV_NAME}/${TD}/${UNIXTIME}" \
    algorithm.total_timesteps=${MAX_BUDGET} \
    hydra.sweeper.dehb_kwargs.min_budget=${MIN_BUDGET} \
    env.name=${ENV_NAME} \
    env.noise_lvl=${NOISE_LVL} \
    algorithm.TD=${TD} \
    run_time=${UNIXTIME} > hpo_results/${ENV_NAME}/${ENV_NAME}_${TD}_noise_lvl${NOISE_LVL_STR}.txt 2>&1