import os
import sys
from tqdm import trange

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from flax.serialization import from_state_dict
import optax
from brax import envs
from brax.io import html

from dtd.common.utils import load_model
from dtd.ppo.networks import Actor

from flax.core import unfreeze
from flax.traverse_util import flatten_dict
import pprint


def print_param_structure(params, name):
    print(f"\n=== Structure of {name} ===")
    flat = flatten_dict(unfreeze(params))
    for key_tuple in flat.keys():
        print("/".join(key_tuple))


def run_brax_ant_and_save_html(env_name: str, TD: str, run_time: int, step: int, num_frames: int):

    # create env
    env = envs.get_environment(env_name=env_name, backend="generalized")
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    
    # create model
    actor = Actor(env.action_size, "relu")
    dummy_input = jnp.zeros(env.observation_size)
    actor_params = actor.init(key, dummy_input)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=optax.adam(1e-3),
    )

    # load model
    cfg_path = f"configs/logs/ppo/{env_name}/{TD}/{run_time}/.hydra/config.yaml"
    state_dict = load_model(cfg_path, "actor", step)
    # print_param_structure(actor_state.params, "actor_state.params")
    # print_param_structure(state_dict["params"], "state_dict['params']")

    restored_params = from_state_dict(actor_state.params, state_dict["params"])
    actor = actor_state.replace(params=restored_params)

    frames = []
    obs = state.obs

    print("Rendering...")
    for t in trange(num_frames):
        key, subkey = jax.random.split(key)
        pi = actor.apply_fn(actor.params, jnp.array(obs))
        action = pi.mean()
        state = env.step(state, action)
        obs = state.obs

        frames.append(state.pipeline_state)

    # save as html
    os.makedirs("videos", exist_ok=True)
    html_path = f"videos/{env_name}_{TD}.html"
    html_str = html.render(env.sys, frames)
    with open(html_path, "w") as f:
        f.write(html_str)


if __name__ == "__main__":
    env_name = sys.argv[1]
    TD = sys.argv[2]
    run_time = sys.argv[3]
    step = int(sys.argv[4])
    num_frames = int(sys.argv[5])
    
    run_brax_ant_and_save_html(env_name, TD, run_time, step, num_frames)