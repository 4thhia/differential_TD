#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES=0

Agent_CLASS="ppo"
ENV_NAME="pusher" # inverted_pendulum 'walker2d' "reacher" "pusher"
MAX_BUDGET="1e7"
MIN_BUDGET="1e5"
TD="dtd"
NOISE_LVL="0.01"
NOISE_LVL_STR=$(echo $NOISE_LVL | sed 's/\.//g')

mkdir -p hpo_results/${Agent_CLASS}/${ENV_NAME}

N_RUNS=10

for i in $(seq 0 $((N_RUNS-1))); do
  UNIXTIME=$(date +%s)
  RUN_DIR="configs/logs/${Agent_CLASS}/${ENV_NAME}/${TD}/${UNIXTIME}"
  mkdir -p "$RUN_DIR"

  OUT_FILE="hpo_results/${Agent_CLASS}/${ENV_NAME}/${ENV_NAME}_${TD}_noise_lvl${NOISE_LVL_STR}_${i}.txt"

  echo "=== Run #$i at $(date) ==="
  echo "  Run dir: $RUN_DIR"
  echo "  Log file: $OUT_FILE"
  echo "-----------------------------"

  # 実行前のタイムスタンプ
  start_time=$(date +%s)

  # 12時間タイムアウト付きで実行
  timeout 12h \
    python3 ${Agent_CLASS}/tune.py --multirun \
      hydra.run.dir="$RUN_DIR" \
      algorithm=${Agent_CLASS}_${TD} \
      algorithm.total_timesteps=${MAX_BUDGET} \
      hydra.sweeper.dehb_kwargs.min_budget=${MIN_BUDGET} \
      env.name=${ENV_NAME} \
      env.noise_lvl=${NOISE_LVL} \
      run_time=${UNIXTIME} \
    > "$OUT_FILE" 2>&1

  EXIT_CODE=$?

  # 実行後のタイムスタンプ
  end_time=$(date +%s)

  # 経過秒数を計算
  duration=$(( end_time - start_time ))
  hours=$(( duration / 3600 ))
  minutes=$((( duration % 3600 ) / 60 ))
  seconds=$(( duration % 60 ))

  # 実行時間を OUT_FILE の最後に追記
  printf "\nElapsed time: %02dh %02dm %02ds\n" \
    "$hours" "$minutes" "$seconds" >> "$OUT_FILE"

  # タイムアウト／正常終了のメッセージはコンソールに出力
  if [ $EXIT_CODE -eq 124 ]; then
    echo "!!! Run #$i timed out after 12 hours (exit code $EXIT_CODE) !!!"
  else
    echo "=== Finished run #$i (exit code $EXIT_CODE) ==="
  fi
  echo
done

echo "All ${N_RUNS} runs completed."
