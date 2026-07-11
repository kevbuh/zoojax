# Paper: https://arxiv.org/pdf/1912.02877

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "jax[cuda13]",
#     "optax",
#     "gymnasium[classic-control]",
#     "wandb",
#     "moviepy",
#     "tyro",
# ]
# ///

import os
import time
from dataclasses import dataclass
from typing import Literal

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # mirror single.py's device = "cpu"

import numpy as np
import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
import wandb
import copy
import pickle
import tyro

@dataclass
class Config:
    # Upside-Down RL
    max_reward: int = 500
    horizon_scale: float = 0.02
    return_scale: float = 0.02
    replay_size: int = 1024
    n_warm_up_episodes: int = replay_size
    batch_size: int = 1024
    n_updates_per_iter: int = 100
    n_episodes_per_iter: int = 15
    last_few: int = 5
    # Network / optimization
    arch: Literal["mlp", "resnet"] = "mlp"  # "resnet" adds residual skips around the hidden layers
    depth: int = 2  # number of hidden layers in the behavior function
    activation: Literal[
        "relu", "tanh", "elu", "gelu", "sigmoid",
        "relu6", "leaky_relu", "hard_sigmoid", "log_sigmoid", "softplus",
        "sparse_plus", "soft_sign", "silu", "hard_silu", "celu", "selu",
        "mish", "hard_tanh", "squareplus",
    ] = "hard_tanh"  # hidden-layer nonlinearity
    layernorm: bool = True  # apply LayerNorm to each hidden layer (pre-activation)
    batchnorm: bool = False  # apply BatchNorm to each hidden layer (pre-activation); mutually exclusive with layernorm
    bn_momentum: float = 0.99  # running-stat momentum for BatchNorm
    hidden_size: int = 512
    lr: float = 3e-4
    seed: int = 0
    max_episodes: int = 500
    val_freq: int = 10  # run cheap greedy validation this often (in training iters)
    n_val_episodes: int = 10
    wandb_project: str = "upside-down-rl-cartpole"
    video_fps: int = 30  # fps of the single rendered rollout saved at the end

args = tyro.cli(Config)

# init Environment
env = gym.make("CartPole-v1")
action_space = env.action_space.n
state_space = env.observation_space.shape[0]
max_reward = args.max_reward

arch = args.arch
# hidden-layer nonlinearity, selected by --activation (sigmoid command-gating is fixed)
hidden_activation = {
    "relu": jax.nn.relu, "tanh": jnp.tanh, "elu": jax.nn.elu, "gelu": jax.nn.gelu, "sigmoid": jax.nn.sigmoid,
    "relu6": jax.nn.relu6, "leaky_relu": jax.nn.leaky_relu, "hard_sigmoid": jax.nn.hard_sigmoid,
    "log_sigmoid": jax.nn.log_sigmoid, "softplus": jax.nn.softplus, "sparse_plus": jax.nn.sparse_plus,
    "soft_sign": jax.nn.soft_sign, "silu": jax.nn.silu, "hard_silu": jax.nn.hard_silu, "celu": jax.nn.celu,
    "selu": jax.nn.selu, "mish": jax.nn.mish, "hard_tanh": jax.nn.hard_tanh, "squareplus": jax.nn.squareplus,
}[args.activation]
layernorm = args.layernorm
batchnorm = args.batchnorm
bn_momentum = args.bn_momentum
assert not (layernorm and batchnorm), "--layernorm and --batchnorm are mutually exclusive"
horizon_scale = args.horizon_scale
return_scale = args.return_scale
replay_size = args.replay_size
n_warm_up_episodes = args.n_warm_up_episodes
n_updates_per_iter = args.n_updates_per_iter
n_episodes_per_iter = args.n_episodes_per_iter
last_few = args.last_few
batch_size = args.batch_size

# Validation command (fixed at the max reward)
val_freq = args.val_freq
n_val_episodes = args.n_val_episodes
val_desired_reward = float(max_reward)
val_time_horizon = float(max_reward)

# Logging
wandb_project = args.wandb_project
video_fps = args.video_fps

_key = jax.random.PRNGKey(args.seed)

def next_key():
    global _key
    _key, sub = jax.random.split(_key)
    return sub

def init_layer(key, fan_in, fan_out):
    """PyTorch-style nn.Linear init: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    bound = 1.0 / np.sqrt(fan_in)
    wk, bk = jax.random.split(key)
    w = jax.random.uniform(wk, (fan_out, fan_in), minval=-bound, maxval=bound)
    b = jax.random.uniform(bk, (fan_out,), minval=-bound, maxval=bound)
    return {"w": w, "b": b}

def layer_norm(x, p, eps=1e-5):
    """Standard LayerNorm over the feature axis with learnable scale/shift."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return p["gamma"] * (x - mean) / jnp.sqrt(var + eps) + p["beta"]

class BF:
    def __init__(self, state_space, action_space, hidden_size, depth, seed):
        # keys: fc1, commands, `depth` hidden layers, output
        keys = jax.random.split(jax.random.PRNGKey(seed), 3 + depth)
        self.actions = np.arange(action_space)
        self.action_space = action_space
        self.params = {
            "fc1": init_layer(keys[0], state_space, hidden_size),
            "commands": init_layer(keys[1], 2, hidden_size),
            "hidden": [init_layer(keys[2 + i], hidden_size, hidden_size) for i in range(depth)],
            "out": init_layer(keys[2 + depth], hidden_size, action_space),
        }
        if layernorm:
            # one LayerNorm (gamma=1, beta=0) per hidden layer
            self.params["ln"] = [{"gamma": jnp.ones((hidden_size,)), "beta": jnp.zeros((hidden_size,))} for _ in range(depth)]
        if batchnorm:
            # trainable scale/shift per hidden layer; running mean/var live outside
            # the trainable params (see bn_state) and are updated manually.
            self.params["bn"] = [{"gamma": jnp.ones((hidden_size,)), "beta": jnp.zeros((hidden_size,))} for _ in range(depth)]
    @staticmethod
    def forward(params, state, command, train=False, bn_state=None):
        # Returns (logits, new_bn_state). new_bn_state is None unless --batchnorm:
        # in train mode it holds the momentum-updated running stats to carry forward;
        # in eval mode it is bn_state unchanged. Inference normalizes with running
        # stats (batch stats are degenerate on a single-state rollout).
        def lin(p, x):
            return x @ p["w"].T + p["b"]
        out = jax.nn.sigmoid(lin(params["fc1"], state))
        command_out = jax.nn.sigmoid(lin(params["commands"], command))
        out = out * command_out
        new_bn = [] if batchnorm else None
        for i, layer in enumerate(params["hidden"]):
            h = lin(layer, out)
            if batchnorm:
                g = params["bn"][i]
                if train:
                    mean = h.mean(axis=0)
                    var = h.var(axis=0)
                    # momentum update of running stats (no gradient through them)
                    new_bn.append({
                        "mean": bn_momentum * bn_state[i]["mean"] + (1.0 - bn_momentum) * jax.lax.stop_gradient(mean),
                        "var": bn_momentum * bn_state[i]["var"] + (1.0 - bn_momentum) * jax.lax.stop_gradient(var),
                    })
                else:
                    mean, var = bn_state[i]["mean"], bn_state[i]["var"]
                    new_bn.append(bn_state[i])
                h = g["gamma"] * (h - mean) / jnp.sqrt(var + 1e-5) + g["beta"]  # pre-activation BatchNorm
            elif layernorm:
                h = layer_norm(h, params["ln"][i])  # pre-activation LayerNorm
            h = hidden_activation(h)
            if arch == "resnet":
                # Residual block: skip connection around each equal-width hidden
                # layer (out = out + act(fc(out))), as in rnd.py's extrahid.
                out = out + h
            else:
                out = h
        out = lin(params["out"], out)
        return out, new_bn
    def __call__(self, state, command):
        return BF.forward(self.params, state, command, False, bn_state)[0]
    def action(self, state, desire, horizon):
        """
        Samples the action based on their probability
        """
        command = jnp.concatenate((jnp.asarray(desire) * return_scale, jnp.asarray(horizon) * horizon_scale), axis=-1)
        action_prob, _ = BF.forward(self.params, jnp.asarray(state), command, False, bn_state)
        action = jax.random.categorical(next_key(), action_prob, axis=-1)
        return int(action)
    def greedy_action(self, state, desire, horizon):
        """
        Returns the greedy action
        """
        command = jnp.concatenate((jnp.asarray(desire) * return_scale, jnp.asarray(horizon) * horizon_scale), axis=-1)
        action_prob, _ = BF.forward(self.params, jnp.asarray(state), command, False, bn_state)
        action = int(jnp.argmax(action_prob, axis=-1))
        return action

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
    def add_sample(self, states, actions, rewards):
        episode = {"states": states, "actions": actions, "rewards": rewards, "summed_rewards": sum(rewards)}
        self.buffer.append(episode)
    def sort(self):
        # sort buffer
        self.buffer = sorted(self.buffer, key=lambda i: i["summed_rewards"], reverse=True)
        # keep the max buffer size
        self.buffer = self.buffer[: self.max_size]
    def get_random_samples(self, batch_size):
        self.sort()
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        return batch
    def get_nbest(self, n):
        self.sort()
        return self.buffer[:n]
    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer(replay_size)
bf = BF(state_space, action_space, args.hidden_size, args.depth, args.seed + 1)
optimizer = optax.adam(args.lr)
opt_state = optimizer.init(bf.params)

# BatchNorm running stats (mean=0, var=1 per hidden layer), updated during training.
# None when --batchnorm is off. Kept out of the trainable params on purpose.
bn_state = None
if batchnorm:
    bn_state = [{"mean": jnp.zeros((args.hidden_size,)), "var": jnp.ones((args.hidden_size,))} for _ in range(args.depth)]

# initial command
init_desired_reward = 1
init_time_horizon = 1

# FUNCTIONS FOR Sampling exploration commands

def sampling_exploration(top_X_eps=last_few):
    """
    This function calculates the new desired reward and new desired horizon based on the replay buffer.
    New desired horizon is calculted by the mean length of the best last X episodes.
    New desired reward is sampled from a uniform distribution given the mean and the std calculated
    from the last best X performances where X is the hyperparameter last_few.
    """
    top_X = buffer.get_nbest(last_few)
    # The exploratory desired horizon dh0 is set to the mean of the lengths of the selected episodes
    new_desired_horizon = np.mean([len(i["states"]) for i in top_X])
    # save all top_X cumulative returns in a list
    returns = [i["summed_rewards"] for i in top_X]
    # from these returns calc the mean and std
    mean_returns = np.mean(returns)
    std_returns = np.std(returns)
    # sample desired reward from a uniform distribution given the mean and the std
    new_desired_reward = np.random.uniform(mean_returns, mean_returns + std_returns)
    return np.array([new_desired_reward], dtype=np.float32), np.array([new_desired_horizon], dtype=np.float32)

# FUNCTIONS FOR TRAINING
def select_time_steps(saved_episode):
    """
    Given a saved episode from the replay buffer this function samples random time steps (t1 and t2) in that episode:
    T = max time horizon in that episode
    Returns t1, t2 and T
    """
    # Select times in the episode:
    T = len(saved_episode["states"])  # episode max horizon
    t1 = np.random.randint(0, T - 1)
    t2 = np.random.randint(t1 + 1, T)
    return t1, t2, T

def create_training_input(episode, t1, t2):
    """
    Based on the selected episode and the given time steps this function returns 4 values:
    1. state at t1
    2. the desired reward: sum over all rewards from t1 to t2
    3. the time horizont: t2 -t1

    4. the target action taken at t1

    buffer episodes are build like [cumulative episode reward, states, actions, rewards]
    """
    state = episode["states"][t1]
    desired_reward = sum(episode["rewards"][t1:t2])
    time_horizont = t2 - t1
    action = episode["actions"][t1]
    return state, desired_reward, time_horizont, action

def create_training_examples(batch_size):
    """
    Creates a data set of training examples that can be used to create a data loader for training.
    ============================================================
    1. for the given batch_size episode idx are randomly selected
    2. based on these episodes t1 and t2 are samples for each selected episode
    3. for the selected episode and sampled t1 and t2 trainings values are gathered
    ______________________________________________________________
    Output are two numpy arrays in the length of batch size:
    Input Array for the Behavior function - consisting of (state, desired_reward, time_horizon)
    Output Array with the taken actions
    """
    input_array = []
    output_array = []
    # select randomly episodes from the buffer
    episodes = buffer.get_random_samples(batch_size)
    for ep in episodes:
        # select time stamps
        t1, t2, T = select_time_steps(ep)
        # For episodic tasks they set t2 to T:
        t2 = T
        state, desired_reward, time_horizont, action = create_training_input(ep, t1, t2)
        input_array.append(
            np.concatenate([state, np.array([desired_reward], dtype=np.float32), np.array([time_horizont], dtype=np.float32)]).astype(np.float32)
        )
        output_array.append(action)
    return input_array, output_array

def _loss_fn(params, state, command, y, bn_state):
    logits, new_bn = BF.forward(params, state, command, True, bn_state)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss, new_bn

@jax.jit
def _update(params, opt_state, state, command, y, bn_state):
    (loss, new_bn), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params, state, command, y, bn_state)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, new_bn

def train_behavior_function(batch_size):
    """
    Trains the BF with on a cross entropy loss were the inputs are the action probabilities based on the state and command.
    The targets are the actions appropriate to the states from the replay buffer.
    """
    global opt_state, bn_state
    X, y = create_training_examples(batch_size)
    X = jnp.stack([jnp.asarray(x) for x in X])
    state = X[:, 0:state_space]
    d = X[:, state_space : state_space + 1]
    h = X[:, state_space + 1 : state_space + 2]
    command = jnp.concatenate((d * return_scale, h * horizon_scale), axis=-1)
    y = jnp.asarray(y, dtype=jnp.int32)
    bf.params, opt_state, pred_loss, bn_state = _update(bf.params, opt_state, state, command, y, bn_state)
    return np.asarray(pred_loss)

def evaluate(desired_return=np.array([init_desired_reward], dtype=np.float32), desired_time_horizon=np.array([init_time_horizon], dtype=np.float32)):
    """
    Runs one episode of the environment to evaluate the bf.
    """
    desired_return = np.array(desired_return, dtype=np.float32)
    desired_time_horizon = np.array(desired_time_horizon, dtype=np.float32)
    state, _ = env.reset()
    rewards = 0
    while True:
        state = np.asarray(state, dtype=np.float32)
        action = bf.action(state, desired_return, desired_time_horizon)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards += reward
        desired_return = np.minimum(desired_return - reward, np.array([max_reward], dtype=np.float32))
        desired_time_horizon = np.maximum(desired_time_horizon - 1, np.array([1], dtype=np.float32))
        if done:
            break
    return rewards

def validate(n_episodes=n_val_episodes):
    """
    Validation: runs the GREEDY (argmax) policy for n_episodes at a fixed command
    (val_desired_reward / val_time_horizon) and returns the per-episode returns.
    Unlike evaluate() this is deterministic given the weights (no action sampling),
    so it gives a cleaner signal of learning progress; we average over several
    episodes because the environment start state is still random.
    """
    returns = []
    for _ in range(n_episodes):
        desired_return = np.array([val_desired_reward], dtype=np.float32)
        desired_time_horizon = np.array([val_time_horizon], dtype=np.float32)
        state, _ = env.reset()
        rewards = 0
        while True:
            state = np.asarray(state, dtype=np.float32)
            action = bf.greedy_action(state, desired_return, desired_time_horizon)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
            desired_return = np.minimum(desired_return - reward, np.array([max_reward], dtype=np.float32))
            desired_time_horizon = np.maximum(desired_time_horizon - 1, np.array([1], dtype=np.float32))
            if done:
                break
        returns.append(rewards)
    return returns

def record_episode(desired_return=val_desired_reward, desired_time_horizon=val_time_horizon):
    """
    Runs one GREEDY episode in an rgb_array environment and returns the rendered
    frames as a uint8 array shaped (T, C, H, W) -- the layout wandb.Video expects
    -- together with the episode return.
    """
    render_env = gym.make("CartPole-v1", render_mode="rgb_array")
    dr = np.array([desired_return], dtype=np.float32)
    dh = np.array([desired_time_horizon], dtype=np.float32)
    state, _ = render_env.reset()
    frames = []
    rewards = 0
    while True:
        frames.append(render_env.render())
        state = np.asarray(state, dtype=np.float32)
        action = bf.greedy_action(state, dr, dh)
        state, reward, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        rewards += reward
        dr = np.minimum(dr - reward, np.array([max_reward], dtype=np.float32))
        dh = np.maximum(dh - 1, np.array([1], dtype=np.float32))
        if done:
            break
    render_env.close()
    frames = np.array(frames, dtype=np.uint8)  # (T, H, W, C)
    frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W) for wandb.Video
    return frames, rewards

# Training Loop

# Algorithm 2 - Generates an Episode using the Behavior Function:
def generate_episode(
    desired_return=np.array([init_desired_reward], dtype=np.float32), desired_time_horizon=np.array([init_time_horizon], dtype=np.float32)
):
    """
    Generates more samples for the replay buffer.
    """
    desired_return = np.array(desired_return, dtype=np.float32)
    desired_time_horizon = np.array(desired_time_horizon, dtype=np.float32)
    state, _ = env.reset()
    states = []
    actions = []
    rewards = []
    while True:
        state = np.asarray(state, dtype=np.float32)
        action = bf.action(state, desired_return, desired_time_horizon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        desired_return -= reward
        desired_time_horizon -= 1
        desired_time_horizon = np.array([np.maximum(desired_time_horizon, 1).item()], dtype=np.float32)
        if done:
            break
    return [states, actions, rewards]

# Algorithm 1
def run_upside_down(max_episodes):
    """ """
    all_rewards = []
    losses = []
    average_100_reward = []
    desired_rewards_history = []
    horizon_history = []
    val_episodes = []  # training-episode index at which validation was run
    val_means = []  # mean greedy return over n_val_episodes
    val_stds = []  # std greedy return over n_val_episodes
    global_step = 0  # cumulative environment steps collected during training
    for ep in range(1, max_episodes + 1):
        iter_start = time.perf_counter()
        steps_this_iter = 0  # env steps collected this iteration (for SPS)
        # improve|optimize bf based on replay buffer
        loss_buffer = []
        for i in range(n_updates_per_iter):
            bf_loss = train_behavior_function(batch_size)
            loss_buffer.append(bf_loss)
        bf_loss = np.mean(loss_buffer)
        losses.append(bf_loss)
        # run x new episode and add to buffer
        for i in range(n_episodes_per_iter):
            # Sample exploratory commands based on buffer
            new_desired_reward, new_desired_horizon = sampling_exploration()
            generated_episode = generate_episode(new_desired_reward, new_desired_horizon)
            buffer.add_sample(generated_episode[0], generated_episode[1], generated_episode[2])
            steps_this_iter += len(generated_episode[2])
        global_step += steps_this_iter
        sps = steps_this_iter / (time.perf_counter() - iter_start)
        new_desired_reward, new_desired_horizon = sampling_exploration()
        # monitoring desired reward and desired horizon
        desired_rewards_history.append(new_desired_reward.item())
        horizon_history.append(new_desired_horizon.item())
        ep_rewards = evaluate(new_desired_reward, new_desired_horizon)
        all_rewards.append(ep_rewards)
        average_100_reward.append(np.mean(all_rewards[-100:]))
        print(
            "\rEpisode: {} | Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f}".format(ep, ep_rewards, np.mean(all_rewards[-100:]), bf_loss),
            end="",
            flush=True,
        )
        if ep % 100 == 0:
            print(
                "\rEpisode: {} | Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f}".format(
                    ep, ep_rewards, np.mean(all_rewards[-100:]), bf_loss
                )
            )
        log = {
            "reward": ep_rewards,
            "mean_100_reward": np.mean(all_rewards[-100:]),
            "loss": float(bf_loss),
            "desired_reward": new_desired_reward.item(),
            "desired_horizon": new_desired_horizon.item(),
            "sps": sps,
            "global_step": global_step,
        }
        # periodic greedy validation
        if ep % val_freq == 0:
            val_returns = validate(n_val_episodes)
            val_episodes.append(ep)
            val_means.append(np.mean(val_returns))
            val_stds.append(np.std(val_returns))
            log["val_mean_return"] = val_means[-1]
            log["val_std_return"] = val_stds[-1]
            print("\rEpisode: {} | Validation (greedy, {} eps) mean: {:.2f} +/- {:.2f}".format(ep, n_val_episodes, val_means[-1], val_stds[-1]))
        wandb.log(log, step=ep)
    return all_rewards, average_100_reward, desired_rewards_history, horizon_history, losses, val_episodes, val_means, val_stds

def main():
    wandb.init(
        project=wandb_project,
        config={
            "env": "CartPole-v1",
            "arch": arch,
            "depth": args.depth,
            "activation": args.activation,
            "layernorm": args.layernorm,
            "batchnorm": args.batchnorm,
            "hidden_size": args.hidden_size,
            "max_reward": max_reward,
            "horizon_scale": horizon_scale,
            "return_scale": return_scale,
            "replay_size": replay_size,
            "n_warm_up_episodes": n_warm_up_episodes,
            "n_updates_per_iter": n_updates_per_iter,
            "n_episodes_per_iter": n_episodes_per_iter,
            "last_few": last_few,
            "batch_size": batch_size,
            "val_freq": val_freq,
            "n_val_episodes": n_val_episodes,
        },
    )
    # Warm-up: fill the replay buffer with initial-command rollouts.
    for i in range(n_warm_up_episodes):
        desired_return = np.array([init_desired_reward], dtype=np.float32)
        desired_time_horizon = np.array([init_time_horizon], dtype=np.float32)
        state, _ = env.reset()
        states = []
        actions = []
        rewards = []
        while True:
            action = bf.action(np.asarray(state, dtype=np.float32), desired_return, desired_time_horizon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            rewards.append(reward)
            state = next_state
            desired_return -= reward
            desired_time_horizon -= 1
            desired_time_horizon = np.array([np.maximum(desired_time_horizon, 1).item()], dtype=np.float32)
            if done:
                break
        buffer.add_sample(states, actions, rewards)
    rewards, average, d, h, loss, val_episodes, val_means, val_stds = run_upside_down(max_episodes=args.max_episodes)
    # EVALUATION RUN
    desired = float(max_reward)
    frames, rewards = record_episode(desired, desired)
    wandb.log({"final_rollout": wandb.Video(frames, fps=video_fps, format="mp4"), "final_rollout_return": rewards})
    print("Desired rewards: {} | after finishing one episode the agent received {} rewards".format(desired, rewards))
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()
