import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import optax
import distrax


class Actor(nn.Module):
    action_dim: int
    activation: str = "tanh"

    def setup(self):
        if self.activation == "tanh":
            self.activation_fn = nn.tanh
        elif self.activation == "relu":
            self.activation_fn = nn.relu
        elif self.activation == "swish":
            self.activation_fn = nn.swish
        else:
            raise ValueError(f"Unknown activation function: {self.activation}. Expected 'tanh' or 'relu' or 'swish'.")

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        h = self.activation_fn(h)
        h = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(h)
        h = self.activation_fn(h)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,)) # https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_continuous_action.py
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        return pi


class Critic(nn.Module):
    activation: str = "tanh"

    def setup(self):
        if self.activation == "tanh":
            self.activation_fn = nn.tanh
        elif self.activation == "relu":
            self.activation_fn = nn.relu
        elif self.activation == "swish":
            self.activation_fn = nn.swish
        else:
            raise ValueError(f"Unknown activation function: {self.activation}. Expected 'tanh' or 'relu' or 'swish'.")

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        h = self.activation_fn(h)
        h = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(h)
        h = self.activation_fn(h)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)

        return jnp.squeeze(critic, axis=-1)

@struct.dataclass
class ActorCritic:
    actor: TrainState
    critic: TrainState


def setup_network(
    rng: jax.random.PRNGKey,
    action_size: int,
    observation_size: int,
    activation: str,
    learning_rate: float,
    max_grad_norm: float,
    anneal_lr: bool,
    num_minibatches: int,
    num_epochs_per_update: int,
    num_updates: int,
) -> ActorCritic:
    actor = Actor(action_size, activation)
    critic = Critic(activation)

    dummy_input = jnp.zeros(observation_size)
    actor_params = actor.init(rng, dummy_input)
    critic_params = critic.init(rng, dummy_input)

    if anneal_lr:
        def linear_schedule(count):
            frac = (
                1.0 - (count // (num_minibatches * num_epochs_per_update)) / num_updates
            )
            return learning_rate * frac
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate, eps=1e-5),
        )

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=optimizer,
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=optimizer,
    )

    network = ActorCritic(actor_state, critic_state)

    return network