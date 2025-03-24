import hydra
from omegaconf import DictConfig

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from dtd.common.env_wrappers import create_env
from dtd.common.train import evaluate_policy
from dtd.ppo.networks import setup_network
from dtd.ppo.train import train_baseline, train_dtd, train_shjb


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    cfg.run_name = (
        f"{cfg.algorithm.agent_class}__{cfg.env.name}__{cfg.algorithm.TD}"
        f"__seed={cfg.algorithm.seed}__noise_lvl={cfg.env.noise_lvl}"
        f"__lr={cfg.algorithm.model_kwargs.learning_rate}__gamma={cfg.algorithm.model_kwargs.gamma}__lambda={cfg.algorithm.model_kwargs.gae_lambda}__clip_range{cfg.algorithm.model_kwargs.clip_range}"
        f"__ent_coef={cfg.algorithm.model_kwargs.ent_coef}__vf_coef={cfg.algorithm.model_kwargs.vf_coef}"
    )

    cfg.algorithm.num_updates = (
        cfg.algorithm.total_timesteps // cfg.algorithm.num_env_steps_per_update // cfg.env.num_envs
    )

    print(f'num_updates:{cfg.algorithm.num_updates}')

    cfg.algorithm.num_minibatches = (
        cfg.algorithm.num_env_steps_per_update * cfg.env.num_envs // cfg.algorithm.minibatch_size
    )

    env = create_env(
        env_name=cfg.env.name,
        backend=cfg.env.backend,
        noise_lvl=cfg.env.noise_lvl,
        batch_size=cfg.env.num_envs,
    )
    cfg.env.action_size = env.action_size
    cfg.env.observation_size = env.observation_size
    cfg.env.dt = float(env.dt)

    rng = jax.random.PRNGKey(cfg.algorithm.seed)
    rng, setup_rng = jax.random.split(rng)
    network = setup_network(
        rng=setup_rng,
        action_size=cfg.env.action_size,
        observation_size=cfg.env.observation_size,
        activation=cfg.algorithm.model_kwargs.activation,
        learning_rate=cfg.algorithm.model_kwargs.learning_rate,
        max_grad_norm=cfg.algorithm.max_grad_norm,
        anneal_lr=cfg.algorithm.anneal_lr,
        num_minibatches=cfg.algorithm.num_minibatches,
        num_epochs_per_update=cfg.algorithm.num_epochs_per_update,
        num_updates=cfg.algorithm.num_updates,
    )

    if cfg.algorithm.TD=="baseline":
        print(f'TD type: baseline')
        (rng, network, _, _), metrics = train_baseline(
            rng=rng,
            env=env,
            num_envs=cfg.env.num_envs,
            noise_lvl=cfg.env.noise_lvl,
            network=network,
            num_updates=cfg.algorithm.num_updates,
            num_env_steps_per_update=cfg.algorithm.num_env_steps_per_update,
            num_epochs_per_update=cfg.algorithm.num_epochs_per_update,
            minibatch_size=cfg.algorithm.minibatch_size,
            num_minibatches=cfg.algorithm.num_minibatches,
            gamma=cfg.algorithm.model_kwargs.gamma,
            gae_lambda=cfg.algorithm.model_kwargs.gae_lambda,
            clip_range=cfg.algorithm.model_kwargs.clip_range,
            ent_coef=cfg.algorithm.model_kwargs.ent_coef,
            vf_coef=cfg.algorithm.model_kwargs.vf_coef,
            normalize_advantage=cfg.algorithm.model_kwargs.normalize_advantage,
        )
    elif cfg.algorithm.TD=="shjb":
        print(f'TD type: shjb')
        (rng, network, _, _), metrics = train_shjb(
            rng=rng,
            env=env,
            num_envs=cfg.env.num_envs,
            noise_lvl=cfg.env.noise_lvl,
            network=network,
            num_updates=cfg.algorithm.num_updates,
            num_env_steps_per_update=cfg.algorithm.num_env_steps_per_update,
            num_epochs_per_update=cfg.algorithm.num_epochs_per_update,
            minibatch_size=cfg.algorithm.minibatch_size,
            num_minibatches=cfg.algorithm.num_minibatches,
            gamma=cfg.algorithm.model_kwargs.gamma,
            gae_lambda=cfg.algorithm.model_kwargs.gae_lambda,
            clip_range=cfg.algorithm.model_kwargs.clip_range,
            ent_coef=cfg.algorithm.model_kwargs.ent_coef,
            vf_coef=cfg.algorithm.model_kwargs.vf_coef,
            normalize_advantage=cfg.algorithm.model_kwargs.normalize_advantage,
            mix_ratio=cfg.algorithm.model_kwargs.mix_ratio,
        )
    elif cfg.algorithm.TD=="dtd":
        print(f'TD type: dtd')
        (rng, network, _, _), metrics = train_dtd(
            rng=rng,
            env=env,
            num_envs=cfg.env.num_envs,
            noise_lvl=cfg.env.noise_lvl,
            network=network,
            num_updates=cfg.algorithm.num_updates,
            num_env_steps_per_update=cfg.algorithm.num_env_steps_per_update,
            num_epochs_per_update=cfg.algorithm.num_epochs_per_update,
            minibatch_size=cfg.algorithm.minibatch_size,
            num_minibatches=cfg.algorithm.num_minibatches,
            gamma=cfg.algorithm.model_kwargs.gamma,
            gae_lambda=cfg.algorithm.model_kwargs.gae_lambda,
            clip_range=cfg.algorithm.model_kwargs.clip_range,
            ent_coef=cfg.algorithm.model_kwargs.ent_coef,
            vf_coef=cfg.algorithm.model_kwargs.vf_coef,
            normalize_advantage=cfg.algorithm.model_kwargs.normalize_advantage,
            mix_ratio=cfg.algorithm.model_kwargs.mix_ratio,
        )
    else:
        raise ValueError(
        f"Invalid value for cfg.algorithm.TD: {cfg.algorithm.TD}. "
        "Expected 'baseline' or 'dtd'."
    )


    eval_env = create_env(
        env_name=cfg.env.name,
        backend=cfg.env.backend,
        noise_lvl=cfg.env.noise_lvl,
        batch_size=cfg.env.num_envs_for_eval,
    )

    minus_mean_reward = evaluate_policy(rng, eval_env, network, cfg.algorithm.num_env_steps_for_eval)

    return minus_mean_reward


if __name__ == "__main__":
    main()