import os
import json
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt


class Transition(NamedTuple):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    info: jnp.ndarray


def dsV_s_fn(value_fn, params, state_t_plus_dt, state_t):
    ds = state_t_plus_dt - state_t
    _, dsV_s = jax.jvp(
        lambda state: value_fn(params, state),
        (state_t,), (ds,)
    )
    return dsV_s


def dsV_ssds_fn(value_fn, params, state_t_plus_dt, state_t):
    ds = state_t_plus_dt - state_t
    dsV_ss_fn = jax.grad(dsV_s_fn, argnums=-1)
    batch_dsV_ss_fn = jax.vmap(dsV_ss_fn, in_axes=(None, None, 0, 0))
    dsV_ss = batch_dsV_ss_fn(value_fn, params, state_t_plus_dt, state_t)
    dsV_ssds = jnp.vecdot(dsV_ss, ds, axis=-1)
    return dsV_ssds


def evaluate_policy(rng, env, network, num_env_steps_for_eval):
    @jax.jit
    def env_step(runner_state, unused):
        rng, network, state, obs = runner_state

        # SELECT ACTION
        rng, action_rng = jax.random.split(rng)
        pi = network.actor.apply_fn(network.actor.params, obs)
        value = network.critic.apply_fn(network.critic.params, obs)
        action = pi.sample(seed=action_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, env_rng = jax.random.split(rng)
        next_state, next_obs, reward, done, info = env.step(env_rng, state, action)

        transition = Transition(
            obs, next_obs, action, reward, done, value, log_prob, info,
        )
        runner_state = (rng, network, next_state, next_obs)
        return runner_state, transition

    rng, env_reset_rng = jax.random.split(rng)
    state = env.reset(env_reset_rng)
    runner_state = (rng, network, state, state.env_state.obs)

    runner_state, traj_batch = jax.lax.scan(
        env_step, runner_state, None, num_env_steps_for_eval,
    )

    unique_returns = []

    for returns_per_env in traj_batch.info["returned_episode_returns"].T:
        prev_val = None
        for val in returns_per_env:
            if (prev_val is not None) or (val != 0):
                if val != prev_val:
                    unique_returns.append(val)
                prev_val = val

    mean_reward = jnp.mean(jnp.array(unique_returns))

    return - mean_reward


def calculate_return_stats_per_update(returns):

    means = []
    stds = []

    for returns_per_step in returns:

        unique_returns = []

        for returns_per_env in returns_per_step.T:
            prev_return = None
            for val in returns_per_env:
                if (prev_return is not None) or (val != 0):
                    if val != prev_return:
                        unique_returns.append(val)
                    prev_return = val

        unique_returns = np.array(unique_returns)
        means.append(float(np.mean(unique_returns)))
        stds.append(float(np.std(unique_returns)))

    return means, stds


def plot_return(baseline_dir, mix_dir, output_dir, title):
    with open(os.path.join(baseline_dir, "means.json"), "r") as f:
        baseline_means = json.load(f)
    with open(os.path.join(baseline_dir, "stds.json"), "r") as f:
        baseline_stds = json.load(f)

    with open(os.path.join(mix_dir, "means.json"), "r") as f:
        mix_means = json.load(f)
    with open(os.path.join(mix_dir, "stds.json"), "r") as f:
        mix_stds = json.load(f)

    x = range(len(baseline_means))


    plt.figure(figsize=(8, 5))

    plt.plot(x, baseline_means, label="TD", color="red", linewidth=2)
    plt.fill_between(x, [a - b for a, b in zip(baseline_means, baseline_stds)], [a + b for a, b in zip(baseline_means, baseline_stds)], color="red", alpha=0.15, edgecolor="none")

    plt.plot(x, mix_means, label="dTD", color="dodgerblue", linewidth=2)
    plt.fill_between(x, [a - b for a, b in zip(mix_means, mix_stds)], [a + b for a, b in zip(mix_means, mix_stds)], color="dodgerblue", alpha=0.15, edgecolor="none")

    # グラフの装飾
    plt.title(f"title")
    plt.xlabel("Optimization Step")
    plt.ylabel("Average Episodic Return")
    plt.legend()
    plt.grid(True)

    # ファイル名の作成と保存
    filename = f"{output_dir}/returns.png"
    plt.savefig(filename)
