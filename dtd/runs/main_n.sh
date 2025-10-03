#!/usr/bin/env bash
# Batch execution of main.sh experiments

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES=3

AGENT_CLASS="a2c"
ENV_NAME="hopper"
MAX_BUDGET="2500000"
TD="dtd"            # baseline / naive / dtd
NOISE_LVL="0.01"
NOISE_LVL_STR=$(echo $NOISE_LVL | sed 's/\.//g')

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# load incumbent
source "${SCRIPT_DIR}/incumbent/${AGENT_CLASS}/${ENV_NAME}/${TD}/noise_lvl${NOISE_LVL_STR}.sh"

# 出力先ディレクトリを作成
mkdir -p results/${AGENT_CLASS}/${ENV_NAME}

N_RUNS=20

for i in $(seq 0 $((N_RUNS-1))); do
  UNIXTIME=$(date +%s)
  RUN_DIR="configs/logs/${AGENT_CLASS}/${ENV_NAME}/${TD}/${UNIXTIME}"
  mkdir -p "$RUN_DIR"

  echo "=== Run #$i at $(date) ==="
  echo "  Run dir: $RUN_DIR"
  echo "-----------------------------"

  # 実行前のタイムスタンプ
  start_time=$(date +%s)

  # 12時間タイムアウト付きで main.py を実行
  timeout 12h python3 ${AGENT_CLASS}/main.py \
    hydra.run.dir="$RUN_DIR" \
    env.name=${ENV_NAME} \
    env.noise_lvl=${NOISE_LVL} \
    algorithm=${AGENT_CLASS}_${TD} \
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

  EXIT_CODE=$?

  # 実行後のタイムスタンプ
  end_time=$(date +%s)
  duration=$(( end_time - start_time ))
  hours=$(( duration / 3600 ))
  minutes=$((( duration % 3600 ) / 60 ))
  seconds=$(( duration % 60 ))

  # 実行時間をログに追記
  printf "\nElapsed time: %02dh %02dm %02ds\n" \
    "$hours" "$minutes" "$seconds" >> "$OUT_FILE"

  # タイムアウト／正常終了メッセージ
  if [ $EXIT_CODE -eq 124 ]; then
    echo "!!! Run #$i timed out after 12 hours (exit code $EXIT_CODE) !!!"
  else
    echo "=== Finished run #$i (exit code $EXIT_CODE) ==="
  fi
  echo
done

echo "All ${N_RUNS} runs completed."
