import os
from omegaconf import OmegaConf

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
