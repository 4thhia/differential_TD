from typing import Dict, Any, Tuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from brax import envs
from brax.envs.base import Env, State, Wrapper


class NoiseWrapper(Wrapper):
    def __init__(self, env: Env, noise_lvl: float):
        super().__init__(env)
        self.noise_lvl = noise_lvl

    def step(self, rng: jax.random.PRNGKey, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        std_dev = jnp.abs(state.obs) * self.noise_lvl
        noise = (1 - state.done) * jax.random.normal(rng, shape=state.obs.shape) * std_dev
        new_obs = state.obs + noise

        return state.replace(obs=new_obs)


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, rng: jax.random.PRNGKey, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(rng, state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps
        return state.replace(done=done)


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        return state

    def step(self, rng: jax.random.PRNGKey, state: State, action: jax.Array) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(rng, state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.tree_util.tree_map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )
        obs = jax.tree_util.tree_map(where_done, state.info['first_obs'], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


@struct.dataclass
class LogEnvState:
    env_state: jax.Array
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.random.PRNGKey) -> LogEnvState:
        env_state = self.env.reset(rng)
        env_state.info["returned_episode_returns"] = 0
        env_state.info["returned_episode_lengths"] = 0
        env_state.info["returned_episode"] = 0
        state = LogEnvState(env_state, 0, 0, 0, 0)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng: jax.random.PRNGKey,
        state: LogEnvState,
        action: jax.Array,
    ) -> Tuple[jax.Array, LogEnvState, jax.Array, bool, Dict[Any, Any]]:
        """Step the environment.


        Args:
          rng: PRNG key.
          state: The current state of the environment.
          action: The action to take.
          params: The parameters of the environment.


        Returns:
          A tuple of (observation, state, reward, done, info).
        """
        env_state = self.env.step(rng, state.env_state, action)
        new_episode_return = state.episode_returns + env_state.reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - env_state.done),
            episode_lengths=new_episode_length * (1 - env_state.done),
            returned_episode_returns=state.returned_episode_returns * (1 - env_state.done)
            + new_episode_return * env_state.done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - env_state.done)
            + new_episode_length * env_state.done,
        )
        env_state.info["returned_episode_returns"] = state.returned_episode_returns
        env_state.info["returned_episode_lengths"] = state.returned_episode_lengths
        env_state.info["returned_episode"] = env_state.done
        return state, env_state.obs, env_state.reward, env_state.done, env_state.info


class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def step(self, rng: jax.random.PRNGKey, state: State, action: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.step)(rng, state, action)


def create_env(
    env_name: str,
    backend: str,
    noise_lvl: Optional[float] = 0,
    episode_length: int = 3000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    logging: bool = True,
    batch_size: Optional[int] = None,
):
    """Creates an environment from the registry.

    Args:
        env_name: environment name string
        episode_length: length of episode
        action_repeat: how many repeated actions to take per environment step
        auto_reset: whether to auto reset the environment after an episode is done
        batch_size: the number of environments to batch together
        **kwargs: keyword argments that get passed to the Env class constructor

    Returns:
        env: an environment
    """
    env = envs.get_environment(env_name=env_name, backend=backend)
    env = NoiseWrapper(env, noise_lvl)

    if episode_length is not None:
        env = EpisodeWrapper(env, episode_length, action_repeat)
    if auto_reset:
        env = AutoResetWrapper(env)
    if logging:
        env = LogWrapper(env)
    if batch_size:
        env = VmapWrapper(env, batch_size)

    return env