import os
import json
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import numpy as np
import jax
import orbax.checkpoint as ocp


def create_ckpt_mngr(
    agent_class: str,
    env_name: str,
    TD: str,
    noise_lvl: str,
    run_time: int,
) -> ocp.CheckpointManager:
    # Orbax checkpoint manager setup
    noise_lvl_str = str(noise_lvl).replace('.', '').zfill(3)

    checkpoint_dir = f"models/{agent_class}/{env_name}/{TD}/noise_lvl{noise_lvl_str}/{run_time}"

    os.makedirs(checkpoint_dir, exist_ok=True)
    # Configure CheckpointManager with the new format
    ckpt_options = ocp.CheckpointManagerOptions(
        step_format_fixed_length=1,
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
    ckpt_mngr = create_ckpt_mngr(
        agent_class=cfg.algorithm.agent_class,
        env_name=cfg.env.name,
        TD=cfg.algorithm.TD,
        noise_lvl=cfg.env.noise_lvl,
        run_time=cfg.run_time,
    )
    # Restore the checkpoint at the specified step
    loaded_state = ckpt_mngr.restore(step)[network]

    # Update the current TrainState with loaded values
    return loaded_state