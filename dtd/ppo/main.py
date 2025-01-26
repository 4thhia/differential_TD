import os
import hydra
from omegaconf import DictConfig

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from dtd.common.env_wrappers import create_env
from dtd.common.train import evaluate_policy, calculate_return_stats_per_update
from dtd.ppo.networks import setup_network
from dtd.ppo.train import train_baseline, train_mix

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

    metricses = []

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
        metricses.append(metrics)
    elif cfg.algorithm.TD=="mix":
        print(f'TD type: mix')
        cfg.run_name += f"__mix_ratio={cfg.algorithm.model_kwargs.mix_ratio}"
        for i in range(4):
            (rng, network, _, _), metrics = train_mix(
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
            metricses.append(metrics)
    else:
        raise ValueError(
        f"Invalid value for cfg.algorithm.TD: {cfg.algorithm.TD}. "
        "Expected 'baseline' or 'sde'."
    )

    meanses = []
    stdses = []
    for i in range(4):
        means, stds = calculate_return_stats_per_update(metricses[i]["returned_episode_returns"])
        meanses += means
        stdses += stds

    save_dir = f"result/metrics/{cfg.algorithm.agent_class}/{cfg.env.name}/{cfg.algorithm.TD}/{cfg.run_name}"
    os.makedirs(save_metrics_dir, exist_ok=True)
    with open(os.path.join(save_dir, "means.json"), "w") as f:
        json.dump(means, f)

    with open(os.path.join(save_dir, "stds.json"), "w") as f:
        json.dump(stds, f)


if __name__ == "__main__":
    main()