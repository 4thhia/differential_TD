import jax
import jax.numpy as jnp

from brax.envs.base import Env

from dtd.common.train import Transition, dsV_s_fn, dsV_ssds_fn
from dtd.ppo.networks import ActorCritic


def train_baseline(
    rng: jax.random.PRNGKey,
    env: Env,
    num_envs: int,
    noise_lvl: float,
    network: ActorCritic,
    num_updates: int,
    num_env_steps_per_update: int,
    num_epochs_per_update: int,
    minibatch_size: int,
    num_minibatches: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    normalize_advantage: bool,
):
    @jax.jit
    def update_step(runner_state, unused):
        # COLLECT TRAJECTORIES
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

        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, num_env_steps_per_update,
        )

        # CALCULATE ADVANTAGE AND TARGET
        rng, network, state, last_obs = runner_state
        last_val = network.critic.apply_fn(network.critic.params, last_obs)
        def calculate_advantages(traj_batch, last_val):
            def gae(advantage_and_next_value, transition):
                advantage, next_value = advantage_and_next_value
                value, reward, done = (
                    transition.value,
                    transition.reward,
                    transition.done,
                )
                delta = reward + gamma * next_value * (1 - done) - value
                advantage = delta + gamma * gae_lambda * advantage * (1 - done)
                return (advantage, value), advantage

            _, advantages = jax.lax.scan(
                gae, (jnp.zeros_like(last_val), last_val), traj_batch,
                reverse=True, unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = calculate_advantages(traj_batch, last_val)

        # UPDATE NETWORK
        def update_network(update_state, unused):
            def update_minibatch(network, batch_info):
                traj_minibatch, advantages_minibatch, targets_minibatch = batch_info

                def loss_fn(actor_params, critic_params):
                    # CALUCULATE ACTOR LOSS
                    pi = network.actor.apply_fn(actor_params, traj_minibatch.obs)
                    log_prob = pi.log_prob(traj_minibatch.action)
                    ratio = jnp.exp(log_prob - traj_minibatch.log_prob)

                    advantages = (1 - normalize_advantage) * advantages_minibatch + normalize_advantage * (advantages_minibatch - advantages_minibatch.mean()) / (advantages_minibatch.std() + 1e-8)
                    actor_loss1 = ratio * advantages
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - clip_range,
                            1.0 + clip_range,
                        ) * advantages
                    )
                    actor_loss = jnp.mean(- jnp.minimum(actor_loss1, actor_loss2))
                    entropy = pi.entropy().mean()

                    # CALUCULATE VALUE LOSS
                    value = network.critic.apply_fn(critic_params, traj_minibatch.obs)
                    value_loss = jnp.mean(jnp.square(targets_minibatch - value))
                    total_loss = (
                        actor_loss
                        - ent_coef * entropy
                        + vf_coef * value_loss
                    )
                    return total_loss, (actor_loss, entropy, value_loss)

                grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
                (total_loss, (actor_loss, entropy, value_loss)), (grads_actor, grads_critic) = grad_fn(network.actor.params, network.critic.params)

                new_actor = network.actor.apply_gradients(grads=grads_actor)
                new_critic = network.critic.apply_gradients(grads=grads_critic)
                network = network.replace(actor=new_actor, critic=new_critic)

                return network, (total_loss, actor_loss, entropy, value_loss)

            rng, network = update_state

            batch_size = minibatch_size * num_minibatches
            assert (
                batch_size == num_env_steps_per_update * num_envs
            ), f"batch_size:{batch_size}: The product of (number of environment steps per update {num_env_steps_per_update}) and (number of parallel environments {num_envs}) must be divisible by the number of minibatches{num_minibatches}."

            rng, batch_rng = jax.random.split(rng)
            permutation = jax.random.permutation(batch_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            network, (total_loss_epoch, actor_loss_epoch, entropy_epoch, value_loss_epoch) = jax.lax.scan(
                update_minibatch, network, minibatches
            )

            update_state = (rng, network)

            return update_state, (jnp.mean(total_loss_epoch), jnp.mean(actor_loss_epoch), jnp.mean(entropy_epoch), jnp.mean(value_loss_epoch))

        update_state = (rng, network)
        (rng, network), loss_info = jax.lax.scan(
            update_network, update_state, None, num_epochs_per_update
        )
        metric = traj_batch.info
        runner_state = (rng, network, state, last_obs)

        return runner_state, metric

    rng, env_reset_rng = jax.random.split(rng)
    state = env.reset(env_reset_rng)
    runner_state = (rng, network, state, state.env_state.obs)

    runner_state, metrics = jax.lax.scan(
        update_step, runner_state, None, num_updates,
    )

    return runner_state, metrics


def train_naive_dtd(
    rng: jax.random.PRNGKey,
    env: Env,
    num_envs: int,
    noise_lvl: float,
    network: ActorCritic,
    num_updates: int,
    num_env_steps_per_update: int,
    num_epochs_per_update: int,
    minibatch_size: int,
    num_minibatches: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    normalize_advantage: bool,
    mix_ratio: float,
):
    @jax.jit
    def update_step(runner_state, unused):
        # COLLECT TRAJECTORIES
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

        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, num_env_steps_per_update,
        )

        # CALCULATE ADVANTAGE
        rng, network, state, last_obs = runner_state
        last_val = network.critic.apply_fn(network.critic.params, last_obs)
        def calculate_advantages(traj_batch, last_val):
            def gae(advantage_and_next_value, transition):
                advantage, next_value = advantage_and_next_value
                value, reward, done = (
                    transition.value,
                    transition.reward,
                    transition.done,
                )
                delta = reward + gamma * next_value * (1 - done) - value
                advantage = delta + gamma * gae_lambda * advantage * (1 - done)
                return (advantage, value), advantage

            _, advantages = jax.lax.scan(
                gae, (jnp.zeros_like(last_val), last_val), traj_batch,
                reverse=True, unroll=16,
            )
            return advantages

        advantages = calculate_advantages(traj_batch, last_val)
        targets_baseline = advantages + traj_batch.value

        # CALCULATE TARGET
        def calculate_dtd_target(unused, transition):
            obs, next_obs, reward = (
                transition.obs,
                transition.next_obs,
                transition.reward,
            )
            target = - (
                reward
                + dsV_s_fn(
                    network.critic.apply_fn,
                    network.critic.params,
                    next_obs,
                    obs,
                )
                + jnp.where(noise_lvl <= 0.0, 0.0, 1.0) * (1 / 2) * dsV_ssds_fn(
                    network.critic.apply_fn,
                    network.critic.params,
                    next_obs,
                    obs,
                )
            ) / jnp.log(gamma)
            return None, target

        _, targets_dtd = jax.lax.scan(
            calculate_dtd_target, None, traj_batch, unroll=16,
        )

        # UPDATE NETWORK
        def update_network(update_state, unused):
            def update_minibatch(network, batch_info):
                traj_minibatch, advantages_minibatch, targets_baseline_minibatch, targets_dtd_minibatch = batch_info

                def loss_fn(actor_params, critic_params):
                    # CALUCULATE ACTOR LOSS
                    pi = network.actor.apply_fn(actor_params, traj_minibatch.obs)
                    log_prob = pi.log_prob(traj_minibatch.action)
                    ratio = jnp.exp(log_prob - traj_minibatch.log_prob)

                    advantages = (1 - normalize_advantage) * advantages_minibatch + normalize_advantage * (advantages_minibatch - advantages_minibatch.mean()) / (advantages_minibatch.std() + 1e-8)
                    actor_loss1 = ratio * advantages
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - clip_range,
                            1.0 + clip_range,
                        ) * advantages
                    )
                    actor_loss = jnp.mean(- jnp.minimum(actor_loss1, actor_loss2))
                    entropy = pi.entropy().mean()

                    # CALUCULATE VALUE LOSS
                    preds = network.critic.apply_fn(critic_params, traj_minibatch.obs)
                    value_loss_baseline = jnp.mean(jnp.square(targets_baseline_minibatch - preds))
                    value_loss_dtd = jnp.mean(jnp.square(targets_dtd_minibatch - preds))
                    value_loss = (1 - mix_ratio) * value_loss_baseline + mix_ratio * value_loss_dtd

                    total_loss = (
                        actor_loss
                        - ent_coef * entropy
                        + vf_coef * value_loss
                    )
                    return total_loss, (actor_loss, entropy, value_loss)

                grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
                (total_loss, (actor_loss, entropy, value_loss)), (grads_actor, grads_critic) = grad_fn(network.actor.params, network.critic.params)

                new_actor = network.actor.apply_gradients(grads=grads_actor)
                new_critic = network.critic.apply_gradients(grads=grads_critic)
                network = network.replace(actor=new_actor, critic=new_critic)

                return network, (total_loss, actor_loss, entropy, value_loss)

            rng, network = update_state

            batch_size = minibatch_size * num_minibatches
            assert (
                batch_size == num_env_steps_per_update * num_envs
            ), f"batch_size:{batch_size}: The product of (number of environment steps per update {num_env_steps_per_update}) and (number of parallel environments {num_envs}) must be divisible by the number of minibatches{num_minibatches}."

            rng, batch_rng = jax.random.split(rng)
            permutation = jax.random.permutation(batch_rng, batch_size)
            batch = (traj_batch, advantages, targets_baseline, targets_dtd)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            network, (total_loss_epoch, actor_loss_epoch, entropy_epoch, value_loss_epoch) = jax.lax.scan(
                update_minibatch, network, minibatches
            )

            update_state = (rng, network)

            return update_state, (jnp.mean(total_loss_epoch), jnp.mean(actor_loss_epoch), jnp.mean(entropy_epoch), jnp.mean(value_loss_epoch))

        update_state = (rng, network)
        (rng, network), loss_info = jax.lax.scan(
            update_network, update_state, None, num_epochs_per_update
        )
        metric = traj_batch.info
        runner_state = (rng, network, state, last_obs)

        return runner_state, metric

    rng, env_reset_rng = jax.random.split(rng)
    state = env.reset(env_reset_rng)
    runner_state = (rng, network, state, state.env_state.obs)

    runner_state, metrics = jax.lax.scan(
        update_step, runner_state, None, num_updates,
    )

    return runner_state, metrics


def train_dtd(
    rng: jax.random.PRNGKey,
    env: Env,
    num_envs: int,
    noise_lvl: float,
    network: ActorCritic,
    num_updates: int,
    num_env_steps_per_update: int,
    num_epochs_per_update: int,
    minibatch_size: int,
    num_minibatches: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    normalize_advantage: bool,
    mix_ratio: float,
):
    @jax.jit
    def update_step(runner_state, unused):
        # COLLECT TRAJECTORIES
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

        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, num_env_steps_per_update,
        )

        # CALCULATE ADVANTAGE
        rng, network, state, last_obs = runner_state
        last_val = network.critic.apply_fn(network.critic.params, last_obs)
        def calculate_advantages(traj_batch, last_val):
            def gae(advantage_and_next_value, transition):
                advantage, next_value = advantage_and_next_value
                value, reward, done = (
                    transition.value,
                    transition.reward,
                    transition.done,
                )
                delta = reward + gamma * next_value * (1 - done) - value
                advantage = delta + gamma * gae_lambda * advantage * (1 - done)
                return (advantage, value), advantage

            _, advantages = jax.lax.scan(
                gae, (jnp.zeros_like(last_val), last_val), traj_batch,
                reverse=True, unroll=16,
            )
            return advantages

        advantages = calculate_advantages(traj_batch, last_val)
        targets_baseline = advantages + traj_batch.value

        # CALCULATE TARGET
        def calculate_dtd_target(unused, transition):
            value, reward = (
                transition.value,
                transition.reward,
            )
            target = reward + jnp.log(gamma) * value
            return None, target

        _, targets_dtd = jax.lax.scan(
            calculate_dtd_target, None, traj_batch, unroll=16,
        )

        # UPDATE NETWORK
        def update_network(update_state, unused):
            def update_minibatch(network, batch_info):
                traj_minibatch, advantages_minibatch, targets_baseline_minibatch, targets_dtd_minibatch = batch_info

                def loss_fn(actor_params, critic_params):
                    # CALUCULATE ACTOR LOSS
                    pi = network.actor.apply_fn(actor_params, traj_minibatch.obs)
                    log_prob = pi.log_prob(traj_minibatch.action)
                    ratio = jnp.exp(log_prob - traj_minibatch.log_prob)

                    advantages = (1 - normalize_advantage) * advantages_minibatch + normalize_advantage * (advantages_minibatch - advantages_minibatch.mean()) / (advantages_minibatch.std() + 1e-8)
                    actor_loss1 = ratio * advantages
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - clip_range,
                            1.0 + clip_range,
                        ) * advantages
                    )
                    actor_loss = jnp.mean(- jnp.minimum(actor_loss1, actor_loss2))
                    entropy = pi.entropy().mean()

                    # CALUCULATE VALUE LOSS
                    preds_baseline = network.critic.apply_fn(critic_params, traj_minibatch.obs)
                    preds_dtd = (
                        - dsV_s_fn(
                            network.critic.apply_fn,
                            critic_params,
                            traj_minibatch.next_obs,
                            traj_minibatch.obs,
                        )
                        - (1 / 2) * dsV_ssds_fn( # jnp.where(noise_lvl <= 0.0, 0.0, 1.0) *
                            network.critic.apply_fn,
                            critic_params,
                            traj_minibatch.next_obs,
                            traj_minibatch.obs,
                        )
                    )
                    value_loss_baseline = jnp.mean(jnp.square(targets_baseline_minibatch - preds_baseline))
                    value_loss_dtd = jnp.mean(jnp.square(targets_dtd_minibatch - preds_dtd))
                    value_loss = (1 - mix_ratio) * value_loss_baseline + mix_ratio * value_loss_dtd

                    total_loss = (
                        actor_loss
                        - ent_coef * entropy
                        + vf_coef * value_loss
                    )
                    return total_loss, (actor_loss, entropy, value_loss)

                grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
                (total_loss, (actor_loss, entropy, value_loss)), (grads_actor, grads_critic) = grad_fn(network.actor.params, network.critic.params)

                new_actor = network.actor.apply_gradients(grads=grads_actor)
                new_critic = network.critic.apply_gradients(grads=grads_critic)
                network = network.replace(actor=new_actor, critic=new_critic)

                return network, (total_loss, actor_loss, entropy, value_loss)

            rng, network = update_state

            batch_size = minibatch_size * num_minibatches
            assert (
                batch_size == num_env_steps_per_update * num_envs
            ), f"batch_size:{batch_size}: The product of (number of environment steps per update {num_env_steps_per_update}) and (number of parallel environments {num_envs}) must be divisible by the number of minibatches{num_minibatches}."

            rng, batch_rng = jax.random.split(rng)
            permutation = jax.random.permutation(batch_rng, batch_size)
            batch = (traj_batch, advantages, targets_baseline, targets_dtd)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            network, (total_loss_epoch, actor_loss_epoch, entropy_epoch, value_loss_epoch) = jax.lax.scan(
                update_minibatch, network, minibatches
            )

            update_state = (rng, network)

            return update_state, (jnp.mean(total_loss_epoch), jnp.mean(actor_loss_epoch), jnp.mean(entropy_epoch), jnp.mean(value_loss_epoch))

        update_state = (rng, network)
        (rng, network), loss_info = jax.lax.scan(
            update_network, update_state, None, num_epochs_per_update
        )
        metric = traj_batch.info
        runner_state = (rng, network, state, last_obs)

        return runner_state, metric

    rng, env_reset_rng = jax.random.split(rng)
    state = env.reset(env_reset_rng)
    runner_state = (rng, network, state, state.env_state.obs)

    runner_state, metrics = jax.lax.scan(
        update_step, runner_state, None, num_updates,
    )

    return runner_state, metrics