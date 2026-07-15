# Paper: https://arxiv.org/pdf/1912.02877

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "jax[cuda13]",
#     "flax",
#     "optax",
#     "gymnax",
#     "wandb",
#     "tyro",
# ]
# ///

import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
import optax
import gymnax
import wandb
import tyro

@dataclass
class Config:
    # Upside-Down RL
    max_reward: int = 500
    horizon_scale: float = 0.02
    return_scale: float = 0.02
    replay_size: int = 1024
    n_warm_up_episodes: int = 64
    batch_size: int = 1024
    n_updates_per_iter: int = 100
    n_episodes_per_iter: int = 15
    last_few: int = 5  # how many to actually look at for determining command
    segments_per_episode: int = 1  # (t1,t2) segments mined per sampled episode
    full_horizon: bool = True  # force t2=T (episodic); False also mines t2<T segments
    depth: int = 2  # number of hidden layers in the behavior function
    activation: Literal["relu", "tanh"] = "tanh"  # hidden-layer nonlinearity
    layernorm: bool = True
    hidden_size: int = 512
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    seed: int = 0
    max_episodes: int = 500  # number of training iterations
    val_freq: int = 5  # run greedy validation this often
    n_val_episodes: int = 30
    wandb_project: str = "upside-down-rl-cartpole"

class BehaviorFunction(nn.Module):
    """Maps (state, command) -> action logits
    Works for a single example (state (obs_dim,), command (2,)) and for a batch (state (B, obs_dim), command (B, 2))
    """
    n_actions: int
    hidden_size: int
    depth: int
    layernorm: bool
    activation: Callable
    @nn.compact
    def __call__(self, state, command):
        out = nn.sigmoid(nn.Dense(self.hidden_size, name="fc1")(state))
        command_out = nn.sigmoid(nn.Dense(self.hidden_size, name="commands")(command))
        out = out * command_out  # sigmoid command-gating
        for i in range(self.depth):
            h = nn.Dense(self.hidden_size, name=f"hidden_{i}")(out)
            if self.layernorm:
                h = nn.LayerNorm(name=f"ln_{i}")(h)
            out = self.activation(h)
        return nn.Dense(self.n_actions, name="out")(out)

def make_command(desire, horizon):
    """Scaled command vector concatenating desired return and desired horizon"""
    return jnp.stack([desire * args.return_scale, horizon * args.horizon_scale], axis=-1)

@partial(jax.jit, static_argnames=("deterministic",))
def rollout_batch(params, keys, desires, horizons, deterministic=False):
    def one(key, desire, horizon):
        reset_key, scan_key = jax.random.split(key)
        obs0, state0 = env.reset(reset_key, env_params)
        def step(carry, k):
            obs, state, des, hor, done_prev = carry
            act_key, env_key = jax.random.split(k)
            logits = model.apply({"params": params}, obs, make_command(des, hor))
            if deterministic:
                action = jnp.argmax(logits)
            else:
                action = jax.random.categorical(act_key, logits)
            n_obs, n_state, reward, done, _ = env.step(env_key, state, action, env_params)
            active = ~done_prev  # this step is part of the real (first) episode
            rew_rec = jnp.where(active, reward, 0.0)
            n_des = jnp.where(active, jnp.minimum(des - reward, args.max_reward), des)
            n_hor = jnp.where(active, jnp.maximum(hor - 1.0, 1.0), hor)
            carry = (n_obs, n_state, n_des, n_hor, done_prev | done)
            return carry, (obs, action, rew_rec, active)
        ks = jax.random.split(scan_key, T)
        init = (obs0, state0, desire, horizon, jnp.bool_(False))
        _, (obs_s, act_s, rew_s, mask) = lax.scan(step, init, ks)
        return {"obs": obs_s, "act": act_s.astype(jnp.int32), "rew": rew_s, "length": mask.sum().astype(jnp.int32), "summed": rew_s.sum()}
    return jax.vmap(one)(keys, desires, horizons)

def empty_buffer():
    return {
        "obs": jnp.zeros((args.replay_size, T, obs_dim), jnp.float32),
        "act": jnp.zeros((args.replay_size, T), jnp.int32),
        "rew": jnp.zeros((args.replay_size, T), jnp.float32),
        "length": jnp.zeros((args.replay_size,), jnp.int32),
        "summed": jnp.full((args.replay_size,), -jnp.inf, jnp.float32),
    }

@jax.jit
def buffer_add(replay_buffer, new):
    cat = {k: jnp.concatenate([replay_buffer[k], new[k]], axis=0) for k in replay_buffer}
    order = jnp.argsort(-cat["summed"])[: args.replay_size]  # keep the best replay_size
    return {k: cat[k][order] for k in cat}

def sampling_exploration(replay_buffer, key):
    """New (desired_reward, desired_horizon) from the best `last_few` episodes

    Horizon = mean length of the top episodes; return = uniform sample in
    [mean, mean+std] of their returns. Buffer is sorted, so the top slice is [:last_few].
    """
    top_len = replay_buffer["length"][: args.last_few].astype(jnp.float32)
    top_ret = replay_buffer["summed"][: args.last_few]
    dh = top_len.mean()
    m, s = top_ret.mean(), top_ret.std()
    dr = jax.random.uniform(key, (), minval=m, maxval=m + s)
    return dr, dh

def make_example(replay_buffer, ep_i, key):
    """Mine one (state, desired_reward, time_horizon, action) training example."""
    L = replay_buffer["length"][ep_i]
    obs, act, rew = replay_buffer["obs"][ep_i], replay_buffer["act"][ep_i], replay_buffer["rew"][ep_i]
    k1, k2 = jax.random.split(key)
    Lc = jnp.maximum(L, 2)
    t1 = jax.random.randint(k1, (), 0, Lc - 1)  # start in [0, L-2]
    if args.full_horizon:
        t2 = L  # return-to-go over the whole tail (episodic task)
    else:
        t2 = jax.random.randint(k2, (), t1 + 1, jnp.maximum(t1 + 2, L))
    idx = jnp.arange(T)
    mask = (idx >= t1) & (idx < t2)
    desired_reward = jnp.sum(rew * mask)  # cumulative reward over [t1, t2)
    time_horizon = (t2 - t1).astype(jnp.float32)
    return obs[t1], desired_reward, time_horizon, act[t1]

def sample_training_batch(replay_buffer, n_valid, key):
    ek, tk = jax.random.split(key)
    ep_idx = jax.random.randint(ek, (n_eps,), 0, n_valid)  # only sample valid slots
    ep_idx = jnp.repeat(ep_idx, args.segments_per_episode)  # (train_B,)
    tks = jax.random.split(tk, train_B)
    return jax.vmap(make_example, in_axes=(None, 0, 0))(replay_buffer, ep_idx, tks)

@jax.jit
def train_epoch(params, opt_state, replay_buffer, n_valid, key):
    """`n_updates_per_iter` gradient steps, scanned. Buffer is constant this epoch."""
    keys = jax.random.split(key, args.n_updates_per_iter)
    def body(carry, k):
        params, opt_state = carry
        states, desires, horizons, actions = sample_training_batch(replay_buffer, n_valid, k)
        def loss_fn(p):
            logits = model.apply({"params": p}, states, make_command(desires, horizons))
            return optax.softmax_cross_entropy_with_integer_labels(logits, actions).mean()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss
    (params, opt_state), losses = lax.scan(body, (params, opt_state), keys)
    return params, opt_state, losses.mean()

@jax.jit
def validate(params, key):
    """Greedy rollouts at the fixed validation command -> (mean, std) return."""
    keys = jax.random.split(key, args.n_val_episodes)
    des = jnp.full((args.n_val_episodes,), float(args.max_reward))
    hor = jnp.full((args.n_val_episodes,), float(args.max_reward))
    ep = rollout_batch(params, keys, des, hor, deterministic=True)
    return ep["summed"].mean(), ep["summed"].std()

def main():
    global args, env, env_params, T, obs_dim, n_eps, train_B, model, optimizer
    args = tyro.cli(Config)
    env, env_params = gymnax.make("CartPole-v1")
    env_params = env_params.replace(max_steps_in_episode=args.max_reward)
    T = args.max_reward  # fixed rollout horizon
    obs_dim = env.observation_space(env_params).shape[0]
    n_actions = env.action_space(env_params).n
    # number of episodes sampled per training batch, mined segments_per_episode times each
    n_eps = max(1, args.batch_size // args.segments_per_episode)
    train_B = n_eps * args.segments_per_episode
    hidden_activation = {"relu": jax.nn.relu, "tanh": jnp.tanh}[args.activation]
    model = BehaviorFunction(n_actions, args.hidden_size, args.depth, args.layernorm, hidden_activation)
    _base_opt = optax.adam(args.lr)
    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), _base_opt) if args.max_grad_norm > 0 else _base_opt
    wandb.init(project=args.wandb_project, config={"env": "CartPole-v1", **vars(args)})
    key = jax.random.PRNGKey(args.seed)
    key, bf_key = jax.random.split(key)
    params = model.init(bf_key, jnp.zeros(obs_dim), jnp.zeros(2))["params"]
    opt_state = optimizer.init(params)
    replay_buffer = empty_buffer()
    # warmup
    key, wk = jax.random.split(key)
    wkeys = jax.random.split(wk, args.n_warm_up_episodes)
    # stochastic rollouts at a fixed init command
    warm = rollout_batch(params, wkeys, jnp.full((args.n_warm_up_episodes,), 1.0), jnp.full((args.n_warm_up_episodes,), 1.0), deterministic=False)
    replay_buffer = buffer_add(replay_buffer, warm)  # seed the replay buffer
    n_valid = min(args.replay_size, args.n_warm_up_episodes)  # number of real (non-empty) slots
    all_rewards = []
    global_step = 0
    for it in range(1, args.max_episodes + 1):
        iter_start = time.perf_counter()
        n_valid_dev = jnp.int32(n_valid)
        # 1) improve the behavior function on the current replay buffer
        key, tk = jax.random.split(key)
        params, opt_state, loss = train_epoch(params, opt_state, replay_buffer, n_valid_dev, tk)
        # 2) collect fresh exploratory episodes and add them to the replay buffer
        key, ck, rk = jax.random.split(key, 3)
        cks = jax.random.split(ck, args.n_episodes_per_iter)
        des, hor = jax.vmap(lambda k: sampling_exploration(replay_buffer, k))(cks)
        rkeys = jax.random.split(rk, args.n_episodes_per_iter)
        new = rollout_batch(params, rkeys, des, hor, deterministic=False)
        replay_buffer = buffer_add(replay_buffer, new)
        n_valid = min(args.replay_size, n_valid + args.n_episodes_per_iter)
        # log stuff
        loss, ep_rewards, steps_this_iter, des_m, hor_m = jax.device_get((loss, new["summed"].mean(), new["length"].sum(), des.mean(), hor.mean()))
        ep_rewards = float(ep_rewards)
        global_step += int(steps_this_iter)
        all_rewards.append(ep_rewards)
        mean_100 = float(np.mean(all_rewards[-100:]))
        sps = int(steps_this_iter) / (time.perf_counter() - iter_start)
        log = {
            "reward": ep_rewards,
            "mean_100_reward": mean_100,
            "loss": float(loss),
            "desired_reward": float(des_m),
            "desired_horizon": float(hor_m),
            "sps": sps,
            "global_step": global_step,
        }
        print(f"\rEpisode: {it} | Rewards: {ep_rewards:.2f} | Mean_100_Rewards: {mean_100:.2f} | Loss: {float(loss):.3f}", end="", flush=True)
        if it % 100 == 0:
            print()
        if it % args.val_freq == 0:
            key, vk = jax.random.split(key)
            v_mean, v_std = jax.device_get(validate(params, vk))
            log["val_mean_return"] = float(v_mean)
            log["val_std_return"] = float(v_std)
            print(f"\rEpisode: {it} | Validation (greedy, {args.n_val_episodes} eps) mean: {float(v_mean):.2f} +/- {float(v_std):.2f}")
        wandb.log(log, step=it)
    # final greedy eval
    key, ek = jax.random.split(key)
    final_mean, final_std = jax.device_get(validate(params, ek))
    wandb.log({"final_return": float(final_mean)})
    print(f"\nDesired {args.max_reward} | final greedy return {float(final_mean):.2f} +/- {float(final_std):.2f}")
    wandb.finish()

if __name__ == "__main__":
    main()
