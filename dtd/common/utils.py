import os
import json
from omegaconf import OmegaConf
import matplotlib.pyplot as plt


import numpy as np
import jax
import orbax.checkpoint as ocp


def create_ckpt_mngr(cfg) -> ocp.CheckpointManager:
    # Orbax checkpoint manager setup
    checkpoint_dir = f"models/{cfg.algorithm.agent_class}/{cfg.experiment_name}/{cfg.sub_experiment_name}/{cfg.run_time}/"

    os.makedirs(checkpoint_dir, exist_ok=True)
    # Configure CheckpointManager with the new format
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=cfg.training.save_interval,
        max_to_keep=cfg.training.save_top_k,
        step_format_fixed_length=3,
        enable_async_checkpointing=True,
        cleanup_tmp_directories=True,
    )
    ckpt_mngr = ocp.CheckpointManager(
        directory=os.path.abspath(checkpoint_dir),
        options=ckpt_options,
    )
    return ckpt_mngr


def load_model(cfg_path: str, network:str, step: int):
    """
    Loads parameters, batch_stats, and opt_state from a checkpoint manager initialized with `cfg`.

    Parameters:
    - cfg: Configuration file used to save models.
    - step: The step at which to restore the checkpoint.
    """
    # Reinitialize the checkpoint manager
    cfg = OmegaConf.load(cfg_path)
    ckpt_mngr = create_ckpt_mngr(cfg)
    # Restore the checkpoint at the specified step
    loaded_state = ckpt_mngr.restore(step)[network]

    # Update the current TrainState with loaded values
    return loaded_state


def plot_return(baseline_dir, mix_dir, output_dir, filename, title, mix_ratio):

    def func(lst):
        result = [lst[0]]  # 最初の値を結果リストに追加
        for i in range(1, len(lst)):
            if lst[i] != lst[i - 1]:  # 連続した値でなければ追加
                result.append(lst[i])
        return result

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(baseline_dir, "means.json"), "r") as f:
        baseline_means = json.load(f)
        baseline_means = func(baseline_means)
    with open(os.path.join(baseline_dir, "stds.json"), "r") as f:
        baseline_stds = json.load(f)
        baseline_stds = func(baseline_stds)
    with open(os.path.join(mix_dir, "means.json"), "r") as f:
        mix_means = json.load(f)
        mix_means = func(mix_means)
    with open(os.path.join(mix_dir, "stds.json"), "r") as f:
        mix_stds = json.load(f)
        mix_stds = func(mix_stds)

    def interpolate(original_list, new_length):
        original_indices = np.linspace(0, 1, len(original_list))  # 元のインデックス（正規化）
        new_indices = np.linspace(0, 1, new_length)  # 新しいインデックス（正規化）

        interpolated_list = np.interp(new_indices, original_indices, original_list)
        return interpolated_list

    baseline_means = interpolate(baseline_means, 2500000)
    baseline_stds = interpolate(baseline_stds, 2500000)
    mix_means = interpolate(mix_means, 2500000)
    mix_stds = interpolate(mix_stds, 2500000)

    print(f'baseline_means:{len(baseline_means)}')
    print(f'baseline_stds:{len(baseline_stds)}')
    print(f'mix_means:{len(mix_means)}')
    print(f'mix_stds:{len(mix_stds)}')

    x = range(len(baseline_means))

    plt.figure(figsize=(8, 5))

    plt.plot(x, baseline_means, label="TD", color="red", linewidth=2)
    plt.fill_between(x, [a - b for a, b in zip(baseline_means, baseline_stds)], [a + b for a, b in zip(baseline_means, baseline_stds)], color="red", alpha=0.15, edgecolor="none")

    plt.plot(x, mix_means, label=f"dTD{round(mix_ratio, 2):.2f}", color="dodgerblue", linewidth=2)
    plt.fill_between(x, [a - b for a, b in zip(mix_means, mix_stds)], [a + b for a, b in zip(mix_means, mix_stds)], color="dodgerblue", alpha=0.15, edgecolor="none")

    # グラフの装飾
    plt.title(title)
    plt.xlabel("total episode step")
    plt.ylabel("Average Episodic Return")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
