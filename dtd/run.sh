#!/bin/bash
# tuning parameters: lmb.lr / regularization coef / weightunet.lr

export XLA_PYTHON_CLIENT_MEM_FRACTION=.70
export CUDA_VISIBLE_DEVICES=1


UNIXTIME=$(date +%s)
ENV_NAME="hopper"
TD="sde"
NOISE_LVL="0.05"
NOISE_LVL_STR=$(echo $NOISE_LVL | sed 's/\.//g')

ALGO="ppo"

python3 ${ALGO}/tune.py --multirun \
    hydra.run.dir="configs/logs/${ALGO}/${ENV_NAME}/${TD}/${UNIXTIME}" \
    env.name=${ENV_NAME} \
    env.noise_lvl=${NOISE_LVL} \
    algorithm.TD=${TD} \
    run_time=${UNIXTIME} > out/tune_${TD}_${ENV_NAME}_noise_lvl${NOISE_LVL_STR}.txt 2>&1