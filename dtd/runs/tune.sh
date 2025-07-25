#!/bin/bash
# tuning parameters: lmb.lr / regularization coef / weightunet.lr

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES=0


UNIXTIME=$(date +%s)
AGENT_CLASS="ppo"
ENV_NAME="hopper"
MAX_BUDGET="1e7"
MIN_BUDGET="1e5"

TD="dtd" # baseline / naive / dtd
NOISE_LVL="0.01"
NOISE_LVL_STR=$(echo $NOISE_LVL | sed 's/\.//g')


echo "Running experiment with:"
echo "ENV_NAME: $ENV_NAME"
echo "NOISE_LVL: $NOISE_LVL"
echo "TD: $TD"
echo "=============================="


python3 ${AGENT_CLASS}/tune.py --multirun \
    hydra.run.dir="configs/logs/${AGENT_CLASS}/${ENV_NAME}/${TD}/${UNIXTIME}" \
    algorithm=${AGENT_CLASS}_${TD} \
    algorithm.total_timesteps=${MAX_BUDGET} \
    hydra.sweeper.dehb_kwargs.min_budget=${MIN_BUDGET} \
    env.name=${ENV_NAME} \
    env.noise_lvl=${NOISE_LVL} \
    run_time=${UNIXTIME} > hpo_results/${ENV_NAME}/${ENV_NAME}_${TD}_noise_lvl${NOISE_LVL_STR}.txt 2>&1
