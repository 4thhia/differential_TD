#!/bin/bash
# tuning parameters: lmb.lr / regularization coef / weightunet.lr

#export XLA_PYTHON_CLIENT_MEM_FRACTION=.70
#export CUDA_VISIBLE_DEVICES=0,1,2


UNIXTIME=$(date +%s)
ALGO="ppo"
ENV_NAME="halfcheetah" # ant / halfcheetah / humanoid
MAX_BUDGET="1e7"
MIN_BUDGET="1e5"

TD="mix" # baseline / sde / mix
NOISE_LVL="0.01"
NOISE_LVL_STR=$(echo $NOISE_LVL | sed 's/\.//g')

if [ "$TD" = "baseline" ]; then
    # Best score seen/Incumbent score: -6915.67

    NUM_ENV_STEPS_PER_UPDATE="8"
    NUM_EPOCHS_PER_UPDATE="19"
    MINIBATCH_SIZE="256"
    LEARNING_RATE="0.00008629229179020549"
    MIX_RATIO="0.6547194057090427"
    GAE_LAMBDA="0.9045341626018957"
    CLIP_RANGE="0.4024707811363749"
    NORMALIZE_ADVANTAGE="0.0"
    VF_COEF="0.5685394219224662"
    ENT_COEF="0.10448839678231656"


    python3 ${ALGO}/main.py \
        hydra.run.dir="configs/logs/${ALGO}/${ENV_NAME}/${TD}/${UNIXTIME}" \
        algorithm=${ALGO}_${TD} \
        algorithm.total_timesteps=${MAX_BUDGET} \
        algorithm.num_env_steps_per_update=${NUM_ENV_STEPS_PER_UPDATE} \
        algorithm.num_epochs_per_update=${NUM_EPOCHS_PER_UPDATE} \
        algorithm.minibatch_size=${MINIBATCH_SIZE} \
        hydra.sweeper.dehb_kwargs.min_budget=${MIN_BUDGET} \
        env.name=${ENV_NAME} \
        env.noise_lvl=${NOISE_LVL} \
        algorithm.model_kwargs.learning_rate=${LEARNING_RATE} \
        algorithm.model_kwargs.gae_lambda=${GAE_LAMBDA} \
        algorithm.model_kwargs.clip_range=${CLIP_RANGE} \
        algorithm.model_kwargs.normalize_advantage=${NORMALIZE_ADVANTAGE} \
        algorithm.model_kwargs.vf_coef=${VF_COEF} \
        algorithm.model_kwargs.ent_coef=${ENT_COEF} \
        run_time=${UNIXTIME}

elif [ "$TD" = "mix" ]; then
    # Best score seen/Incumbent score: -13279.12
    NUM_ENV_STEPS_PER_UPDATE="8"
    NUM_EPOCHS_PER_UPDATE="8"
    MINIBATCH_SIZE="512"
    LEARNING_RATE="0.0006585676462596407"
    MIX_RATIO="0.6547194057090427"
    GAE_LAMBDA="0.8438960401775698"
    CLIP_RANGE="0.34265129604640837"
    NORMALIZE_ADVANTAGE="0.0"
    VF_COEF="0.9592619112875895"
    ENT_COEF="0.032421711330095936"


    python3 ${ALGO}/main.py \
        hydra.run.dir="configs/logs/${ALGO}/${ENV_NAME}/${TD}/${UNIXTIME}" \
        algorithm=${ALGO}_${TD} \
        algorithm.total_timesteps=${MAX_BUDGET} \
        algorithm.num_env_steps_per_update=${NUM_ENV_STEPS_PER_UPDATE} \
        algorithm.num_epochs_per_update=${NUM_EPOCHS_PER_UPDATE} \
        algorithm.minibatch_size=${MINIBATCH_SIZE} \
        hydra.sweeper.dehb_kwargs.min_budget=${MIN_BUDGET} \
        env.name=${ENV_NAME} \
        env.noise_lvl=${NOISE_LVL} \
        algorithm.model_kwargs.learning_rate=${LEARNING_RATE} \
        algorithm.model_kwargs.mix_ratio=${MIX_RATIO} \
        algorithm.model_kwargs.gae_lambda=${GAE_LAMBDA} \
        algorithm.model_kwargs.clip_range=${CLIP_RANGE} \
        algorithm.model_kwargs.normalize_advantage=${NORMALIZE_ADVANTAGE} \
        algorithm.model_kwargs.vf_coef=${VF_COEF} \
        algorithm.model_kwargs.ent_coef=${ENT_COEF} \
        run_time=${UNIXTIME}

fi