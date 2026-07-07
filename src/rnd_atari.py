# Paper:          https://arxiv.org/abs/1810.12894
# Reference impl: https://github.com/openai/random-network-distillation
# Wandb report:   https://wandb.ai/kevinbuhler/rnd-atari-jax/reports/ZOOJAX-Atari-RND--VmlldzoxNzQzOTYzOA
# Run with:  uv run src/rnd_atari.py
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jax[cuda12]",
#     "flax",
#     "optax",
#     "distrax",
#     "numpy",
#     "gymnasium[atari]",
#     "ale-py",
#     "envpool",
#     "opencv-python",
#     "tyro",
#     "wandb",
# ]
# ///

from dataclasses import dataclass
from typing import Any, NamedTuple

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

def _lrelu(x):
    """tf.nn.leaky_relu uses alpha=0.2 by default — JAX/Flax default is 0.01.
    The original RND target/predictor convs go through tf.nn.leaky_relu, so
    pin alpha=0.2 here to match.
    """
    return jax.nn.leaky_relu(x, negative_slope=0.2)

class CnnPolicy(nn.Module):
    """Conv policy with dual value heads. Categorical action distribution.

    Input:  obs in [B, 84, 84, 4] (uint8 or float32; cast + /255 inside).
    Output: (logits, vpred_ext, vpred_int).
    """
    num_actions: int
    hidsize: int = 256  # 128 * enlargement
    extrahid: bool = True
    @nn.compact
    def __call__(self, obs):
        x = obs.astype(jnp.float32) / 255.0
        x = nn.relu(nn.Conv(32, (8, 8), (4, 4), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="c1")(x))
        x = nn.relu(nn.Conv(64, (4, 4), (2, 2), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="c2")(x))
        x = nn.relu(nn.Conv(64, (4, 4), (1, 1), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="c3")(x))
        x = x.reshape(x.shape[0], -1)
        # fc1: hidsize, sqrt(2)
        x = nn.relu(nn.Dense(self.hidsize, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="fc1")(x))
        # fc_additional: 448, sqrt(2)
        x = nn.relu(nn.Dense(448, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="fc_additional")(x))
        if self.extrahid:
            # extrahid residual blocks, init_scale=0.1
            x_val = x + nn.relu(nn.Dense(448, kernel_init=orthogonal(0.1), bias_init=constant(0.0), name="fc2val")(x))
            x_act = x + nn.relu(nn.Dense(448, kernel_init=orthogonal(0.1), bias_init=constant(0.0), name="fc2act")(x))
        else:
            x_val = x
            x_act = x
        # heads, all init_scale=0.01
        logits = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="pd")(x_act)
        vpred_int = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="vf_int")(x_val)
        vpred_ext = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="vf_ext")(x_val)
        return logits, jnp.squeeze(vpred_ext, -1), jnp.squeeze(vpred_int, -1)

class RNDTargetCNN(nn.Module):
    """Frozen target net.

    Input:  pre-normalized + clipped last channel, [B, 84, 84, 1].
    Output: rep features [B, rep_size]. Linear last layer per the original.

    Architecture from cnn_policy_param_matched.py:141-145:
      3 leaky_relu convs (32, 64, 64) -> flatten -> linear fc(rep_size).
    """
    convfeat: int = 32  # 16 * enlargement
    rep_size: int = 512
    @nn.compact
    def __call__(self, x):
        x = _lrelu(nn.Conv(self.convfeat, (8, 8), (4, 4), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="c1r")(x))
        x = _lrelu(nn.Conv(self.convfeat * 2, (4, 4), (2, 2), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="c2r")(x))
        x = _lrelu(nn.Conv(self.convfeat * 2, (3, 3), (1, 1), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="c3r")(x))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.rep_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="fc1r")(x)
        return x

class RNDPredictorCNN(nn.Module):
    """Trained predictor net (deeper than the target).

    Input:  pre-normalized + clipped last channel, [B, 84, 84, 1].
    Output: rep features [B, rep_size].

    Architecture from cnn_policy_param_matched.py:156-163:
      3 leaky_relu convs (32, 64, 64) -> flatten ->
      relu fc(256*enlargement) -> relu fc(256*enlargement) -> linear fc(rep_size).
    """
    convfeat: int = 32
    rep_size: int = 512
    enlargement: int = 2
    @nn.compact
    def __call__(self, x):
        x = _lrelu(
            nn.Conv(self.convfeat, (8, 8), (4, 4), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="c1rp_pred")(x)
        )
        x = _lrelu(
            nn.Conv(
                self.convfeat * 2, (4, 4), (2, 2), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="c2rp_pred"
            )(x)
        )
        x = _lrelu(
            nn.Conv(
                self.convfeat * 2, (3, 3), (1, 1), padding="VALID", kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="c3rp_pred"
            )(x)
        )
        x = x.reshape(x.shape[0], -1)
        fc_width = 256 * self.enlargement
        x = nn.relu(nn.Dense(fc_width, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="fc1r_hat1_pred")(x))
        x = nn.relu(nn.Dense(fc_width, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="fc1r_hat2_pred")(x))
        x = nn.Dense(self.rep_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="fc1r_hat3_pred")(x)
        return x

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value_ext: jnp.ndarray
    value_int: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray

class RunningMoments(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

class RewNormState(NamedTuple):
    rewems: jnp.ndarray
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

def welford_update(state, batch):
    # RunningMeanStd.update_from_moments
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]
    delta = batch_mean - state.mean
    tot_count = state.count + batch_count
    new_mean = state.mean + delta * batch_count / tot_count
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
    new_var = M2 / tot_count
    return new_mean, new_var, tot_count

def normalize_obs(obs_last_channel, obs_norm):
    # clip((x - ph_mean) / ph_std, -5, 5)
    return jnp.clip((obs_last_channel - obs_norm.mean) / jnp.sqrt(obs_norm.var + 1e-8), -5.0, 5.0)

class RNDAtariAgent(flax.struct.PyTreeNode):
    train_state: TrainState
    rnd_target_params: Any
    obs_norm: RunningMoments  # per-pixel mean/var over (84, 84, 1)
    rew_norm: RewNormState
    config: dict = flax.struct.field(pytree_node=False)
    def loss_fn(agent, params, traj_batch, gae, targets_ext, targets_int, rng_mask):
        logits, vpred_ext, vpred_int = agent.train_state.apply_fn(params["policy"], traj_batch.obs)
        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(traj_batch.action)
        # vf_loss_{int,ext} = 0.5 * vf_coef * mean(square(...))
        vf_loss_ext = (0.5 * agent.config["vf_coef"]) * jnp.square(vpred_ext - targets_ext).mean()
        vf_loss_int = (0.5 * agent.config["vf_coef"]) * jnp.square(vpred_int - targets_int).mean()
        # clipped surrogate
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        if agent.config["normalize_adv"]:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1.0 - agent.config["clip_eps"], 1.0 + agent.config["clip_eps"]) * gae
        pg_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()
        entropy = pi.entropy().mean()
        # ent_loss = (-ent_coef) * entropy
        ent_loss = -agent.config["ent_coef"] * entropy
        # RND aux loss: only the last channel feeds RND, cast uint8 to float
        # before normalization for numeric safety
        last_chan = traj_batch.obs[..., -1:].astype(jnp.float32)
        rnd_obs = normalize_obs(last_chan, agent.obs_norm)
        target_feat = RNDTargetCNN().apply(agent.rnd_target_params, rnd_obs)
        pred_feat = RNDPredictorCNN().apply(params["rnd_predictor"], rnd_obs)
        pred_errors = jnp.mean(jnp.square(target_feat - pred_feat), axis=-1)
        # random-mask + safe-mean
        mask = jax.random.uniform(rng_mask, shape=pred_errors.shape) < agent.config["update_proportion"]
        aux_loss = jnp.sum(mask * pred_errors) / jnp.maximum(jnp.sum(mask), 1.0)
        # total = pg + ent + (vf_int + vf_ext) + aux
        total_loss = pg_loss + ent_loss + vf_loss_ext + vf_loss_int + aux_loss
        return total_loss, {
            "total_loss": total_loss,
            "pg_loss": pg_loss,
            "vf_loss_ext": vf_loss_ext,
            "vf_loss_int": vf_loss_int,
            "entropy": entropy,
            "aux_loss": aux_loss,
        }
    @jax.jit
    def update(agent, traj_batch, advantages, targets_ext, targets_int, rng):
        num_envs = agent.config["num_envs"]
        num_steps = agent.config["num_steps"]
        num_minibatches = agent.config["num_minibatches"]
        update_epochs = agent.config["update_epochs"]
        batch_size = num_steps * num_envs
        def _update_epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng, mask_rng = jax.random.split(rng, 3)
            permutation = jax.random.permutation(perm_rng, batch_size)
            batch = (traj_batch, advantages, targets_ext, targets_int)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])), shuffled)
            mask_rngs = jax.random.split(mask_rng, num_minibatches)
            def _update_mb(train_state, mb_with_rng):
                (mb_traj, mb_adv, mb_te, mb_ti), mb_mask_rng = mb_with_rng
                (_, info), grads = jax.value_and_grad(agent.loss_fn, has_aux=True)(train_state.params, mb_traj, mb_adv, mb_te, mb_ti, mb_mask_rng)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, info
            train_state, info = jax.lax.scan(_update_mb, train_state, (minibatches, mask_rngs))
            return (train_state, rng), info
        (new_train_state, rng), info = jax.lax.scan(_update_epoch, (agent.train_state, rng), None, update_epochs)
        info = jax.tree_util.tree_map(lambda x: x.mean(), info)
        return agent.replace(train_state=new_train_state), info, rng
    @jax.jit
    def act(agent, obs, rng):
        rng, act_rng = jax.random.split(rng)
        logits, value_ext, value_int = agent.train_state.apply_fn(agent.train_state.params["policy"], obs)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=act_rng)
        return action, pi.log_prob(action), value_ext, value_int, rng
    @jax.jit
    def value(agent, obs):
        _, ve, vi = agent.train_state.apply_fn(agent.train_state.params["policy"], obs)
        return ve, vi
    @jax.jit
    def intrinsic_reward(agent, obs):
        # obs: (B, 84, 84, 4) frame-stacked. RND only sees the last frame
        last_chan = obs[..., -1:].astype(jnp.float32)
        rnd_obs = normalize_obs(last_chan, agent.obs_norm)
        target_feat = RNDTargetCNN().apply(agent.rnd_target_params, rnd_obs)
        pred_feat = RNDPredictorCNN().apply(agent.train_state.params["rnd_predictor"], rnd_obs)
        return jnp.mean(jnp.square(target_feat - pred_feat), axis=-1)
    def update_obs_norm(agent, batch):
        """Update per-pixel obs RMS over the last channel only.

        ppo_agent.py:493-494: obs_.reshape((-1, 84, 84, 4))[:, :, :, -1:] feeds
        RunningMeanStd(shape=(84, 84, 1)). welford_update averages over axis=0,
        so passing a (N, 84, 84, 1) batch gets the per-pixel stats we want.
        """
        last = batch[..., -1:].astype(jnp.float32)
        last = last.reshape((-1,) + last.shape[-3:])
        mean, var, count = welford_update(agent.obs_norm, last)
        return agent.replace(obs_norm=RunningMoments(mean=mean, var=var, count=count))
    def update_rew_norm(agent, rffs, rewems_final):
        # RewardForwardFilter -> RunningMeanStd(scalar)
        mean, var, count = welford_update(agent.rew_norm, rffs.reshape(-1))
        return agent.replace(rew_norm=RewNormState(rewems=rewems_final, mean=mean, var=var, count=count))

def create_learner(config, rng, obs_shape, num_actions, num_total_updates=None):
    # obs_shape == (84, 84, 4)
    rng, rng_net, rng_target, rng_pred = jax.random.split(rng, 4)
    network = CnnPolicy(num_actions=num_actions)
    rnd_target = RNDTargetCNN()
    rnd_predictor = RNDPredictorCNN()
    init_x = jnp.zeros((1,) + tuple(obs_shape), dtype=jnp.uint8)
    init_x_rnd = jnp.zeros((1, obs_shape[0], obs_shape[1], 1), dtype=jnp.float32)
    rnd_target_params = rnd_target.init(rng_target, init_x_rnd)
    combined_params = {"policy": network.init(rng_net, init_x), "rnd_predictor": rnd_predictor.init(rng_pred, init_x_rnd)}
    # Paper default: max_grad_norm=0.0 == no clip. Anything > 0 clips
    chain_args = []
    if config.max_grad_norm > 0:
        chain_args.append(optax.clip_by_global_norm(config.max_grad_norm))
    if config.anneal_lr and num_total_updates is not None:
        def lr_schedule(count):
            frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / num_total_updates
            return config.lr * frac
        chain_args.append(optax.adam(learning_rate=lr_schedule, eps=1e-5))
    else:
        chain_args.append(optax.adam(config.lr, eps=1e-5))
    tx = optax.chain(*chain_args)
    train_state = TrainState.create(apply_fn=network.apply, params=combined_params, tx=tx)
    obs_h, obs_w, _ = obs_shape  # (84, 84, 4)
    obs_norm = RunningMoments(mean=jnp.zeros((obs_h, obs_w, 1)), var=jnp.ones((obs_h, obs_w, 1)), count=jnp.asarray(1e-4))
    rew_norm = RewNormState(rewems=jnp.zeros(config.num_envs), mean=jnp.asarray(0.0), var=jnp.asarray(1.0), count=jnp.asarray(1e-4))
    agent_config = flax.core.FrozenDict(
        dict(
            clip_eps=config.cliprange,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            num_envs=config.num_envs,
            num_steps=config.num_steps,
            num_minibatches=config.num_minibatches,
            update_epochs=config.update_epochs,
            update_proportion=config.proportion_of_exp_used_for_predictor_update,
            normalize_adv=config.normalize_adv,
            use_news=config.use_news,
        )
    )
    agent = RNDAtariAgent(train_state=train_state, rnd_target_params=rnd_target_params, obs_norm=obs_norm, rew_norm=rew_norm, config=agent_config)
    return agent, rng

@dataclass
class Config:
    # ppo (paper defaults)
    lr: float = 1e-4
    num_envs: int = 128
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma_int: float = 0.99
    gamma_ext: float = 0.99
    gae_lambda: float = 0.95
    cliprange: float = 0.1
    ent_coef: float = 0.001
    # vf_coef = 1.0 matches paper's effective weight 0.5: loss is
    # `0.5 * vf_coef * mean(square(...))`
    vf_coef: float = 1.0
    # Paper: max_grad_norm=0.0 == no clip
    max_grad_norm: float = 0.0
    int_coef: float = 1.0
    ext_coef: float = 2.0
    use_news: int = 0  # 0 -> no done mask on intrinsic stream
    # paper code default is 1.0; paper text reports 0.25 — we follow
    # the reported configuration to match algorithms/rnd.py
    proportion_of_exp_used_for_predictor_update: float = 0.25
    # rnd / atari
    frame_stack: int = 4
    rep_size: int = 512
    anneal_lr: bool = False
    normalize_adv: bool = False  # paper does NOT normalize advantages
    # training
    # envpool's XLA step feeds a fully-jitted on-device rollout+learner (see
    # run_xla). ~11k SPS at num_envs=128 on a 6-core / RTX 5080, paper-faithful.
    # (Single-GPU caps RND at ~11-12k: envpool's XLA step blocks the one CUDA
    #  stream, so the learner can't overlap env stepping. ~20k would need a 2nd GPU.)
    envpool_num_threads: int = 0  # 0 -> envpool auto (defaults to num_envs)
    seed: int = 0
    env: str = "MontezumaRevengeNoFrameskip-v4"
    total_timesteps: int = int(1e9)
    num_iterations_obs_norm_init: int = 50  # 128*50 random-agent steps
    update_ob_stats_from_random_agent: int = 1
    sticky_action_p: float = 0.25
    max_episode_steps: int = 4500
    wandb_project: str = "rnd-atari-jax"
    log_every_n_updates: int = 1

if __name__ == "__main__":
    """envpool reproduces the original RND env stack (atari_wrappers.py:200-224)
    natively: StickyActionEnv(p=0.25) -> MaxAndSkipEnv(skip=4) -> WarpFrame(84x84
    grayscale, INTER_AREA) -> ClipRewardEnv(sign) -> FrameStack(k=4), with
    visited_rooms recovered from info["ram"] inside the jitted rollout.
    """
    import time
    import envpool
    import tyro
    import wandb
    def _envpool_task_id(env_id):
        # "MontezumaRevengeNoFrameskip-v4" / "ALE/MontezumaRevenge-v5" -> "MontezumaRevenge-v5"
        name = env_id.split("/")[-1].split("NoFrameskip")[0].split("-")[0]
        return f"{name}-v5"
    def run_xla(cfg):
        """Fully-jitted RND on envpool's XLA interface.

        The rollout is a jax.lax.scan over envpool's XLA step (env-stepping is a
        CPU custom call, policy forward on GPU, zero per-step Python / host
        transfers). The learner (intrinsic reward, reward-forward-filter, dual
        GAE, obs+rew norm, PPO update) is one jitted call. It's synchronous and
        on-policy, so it stays paper-faithful (~11k SPS at num_envs=128).
        visited_rooms is recovered from info["ram"] inside the scan.
        """
        N, STEPS = cfg.num_envs, cfg.num_steps
        steps_per_update = STEPS * N
        total_updates = int(cfg.total_timesteps // steps_per_update)
        room_addr = 3 if "Montezuma" in cfg.env else (1 if "Pitfall" in cfg.env else None)
        env = envpool.make(
            _envpool_task_id(cfg.env),
            env_type="gymnasium",
            num_envs=N,
            seed=cfg.seed,
            stack_num=cfg.frame_stack,
            frame_skip=4,
            gray_scale=True,
            img_height=84,
            img_width=84,
            repeat_action_probability=cfg.sticky_action_p,
            episodic_life=False,
            reward_clip=True,
            use_inter_area_resize=True,
            max_episode_steps=cfg.max_episode_steps,
            num_threads=cfg.envpool_num_threads,
        )
        num_actions = int(env.action_space.n)
        handle, recv, send, step_env = env.xla()
        rng = jax.random.PRNGKey(cfg.seed)
        agent, rng = create_learner(cfg, rng, (84, 84, cfg.frame_stack), num_actions, num_total_updates=total_updates)
        wandb.init(project=cfg.wandb_project, config=vars(cfg))
        GE, GI, LAM, USE_NEWS = cfg.gamma_ext, cfg.gamma_int, cfg.gae_lambda, cfg.use_news
        apply_pol = agent.train_state.apply_fn
        def nhwc(o):
            return jnp.transpose(o, (0, 2, 3, 1))  # (N,4,84,84)->(N,84,84,4), free view on-device
        # ---- rollout: one jitted scan over STEPS envpool XLA steps ----
        def rollout(agent, handle, obs, rng, ep_ret, ep_len, rooms):
            params = agent.train_state.params["policy"]
            def body(carry, _):
                handle, obs, rng, ep_ret, ep_len, sret, slen, cnt, rooms = carry
                logits, ve, vi = apply_pol(params, obs)
                pi = distrax.Categorical(logits=logits)
                rng, sk = jax.random.split(rng)
                action = pi.sample(seed=sk)
                logp = pi.log_prob(action)
                handle, (nobs, reward, term, trunc, info) = step_env(handle, action.astype(jnp.int32))
                done = jnp.logical_or(term, trunc)
                # episode return uses the RAW (unclipped) reward from info -- the
                # true game score cleanRL/the paper plot. Training still uses the
                # clipped `reward`. episodic_life=False so done == true game-over.
                ep_ret = ep_ret + info["reward"]
                ep_len = ep_len + 1
                sret = sret + jnp.sum(jnp.where(done, ep_ret, 0.0))
                slen = slen + jnp.sum(jnp.where(done, ep_len, 0))
                cnt = cnt + jnp.sum(done.astype(jnp.int32))
                ep_ret = jnp.where(done, 0.0, ep_ret)
                ep_len = jnp.where(done, 0, ep_len)
                if room_addr is not None:
                    rooms = rooms.at[info["ram"][:, room_addr].astype(jnp.int32)].set(True)
                out = (obs, action, logp, ve, vi, reward, done.astype(jnp.float32))
                return (handle, nhwc(nobs), rng, ep_ret, ep_len, sret, slen, cnt, rooms), out
            z = (jnp.float32(0.0), jnp.int32(0), jnp.int32(0))
            init = (handle, obs, rng, ep_ret, ep_len, *z, rooms)
            (handle, obs, rng, ep_ret, ep_len, sret, slen, cnt, rooms), outs = jax.lax.scan(body, init, None, length=STEPS)
            obs_b, act_b, logp_b, ve_b, vi_b, rew_b, done_b = outs
            traj = Transition(done=done_b, action=act_b, value_ext=ve_b, value_int=vi_b, reward=rew_b, int_reward=rew_b, log_prob=logp_b, obs=obs_b)
            stats = dict(sret=sret, slen=slen, cnt=cnt)
            return handle, obs, rng, ep_ret, ep_len, rooms, traj, stats
        rollout_jit = jax.jit(rollout)
        # ---- on-device dual GAE (reverse scan over STEPS, arrays are (STEPS,N)) ----
        def dual_gae(rew_ext, norm_int, ve, vi, done, last_ve, last_vi):
            next_ve = jnp.concatenate([ve[1:], last_ve[None]], 0)
            next_vi = jnp.concatenate([vi[1:], last_vi[None]], 0)
            next_done = done  # done[t] == "obs_{t+1} is a boundary" == host buf_dones[t+1]
            def body(carry, x):
                gae_e, gae_i = carry
                re, ni, v_e, v_i, nv_e, nv_i, nd = x
                notdone = 1.0 - nd
                delta_e = re + GE * nv_e * notdone - v_e
                gae_e = delta_e + GE * LAM * notdone * gae_e
                nn_i = notdone if USE_NEWS else 1.0
                delta_i = ni + GI * nv_i * nn_i - v_i
                gae_i = delta_i + GI * LAM * nn_i * gae_i
                return (gae_e, gae_i), (gae_e, gae_i)
            _, (adv_e, adv_i) = jax.lax.scan(
                body, (jnp.zeros(N), jnp.zeros(N)), (rew_ext, norm_int, ve, vi, next_ve, next_vi, next_done), reverse=True
            )
            return adv_e, adv_i
        # ---- learner: intrinsic + rew-filter + norms + GAE + PPO update, one jit ----
        def learn(agent, traj, last_obs, rng):
            obs = traj.obs
            flat = obs.reshape((STEPS * N,) + obs.shape[2:])
            int_rews = agent.intrinsic_reward(flat).reshape(STEPS, N)  # uses pre-update obs_norm
            def rff(carry, r):
                rewems = carry * GI + r
                return rewems, rewems
            rewems_final, rffs = jax.lax.scan(rff, agent.rew_norm.rewems, int_rews)
            r_mean, r_var, r_cnt = welford_update(agent.rew_norm, rffs.reshape(-1))
            agent = agent.replace(rew_norm=RewNormState(rewems=rewems_final, mean=r_mean, var=r_var, count=r_cnt))
            norm_int = int_rews / jnp.sqrt(r_var + 1e-8)
            agent = agent.update_obs_norm(flat)  # post-intrinsic, like main()
            last_ve, last_vi = agent.value(last_obs)
            adv_e, adv_i = dual_gae(traj.reward, norm_int, traj.value_ext, traj.value_int, traj.done, last_ve, last_vi)
            tgt_e, tgt_i = adv_e + traj.value_ext, adv_i + traj.value_int
            adv = cfg.ext_coef * adv_e + cfg.int_coef * adv_i
            tr = traj._replace(int_reward=int_rews)
            agent, info, rng = agent.update(tr, adv, tgt_e, tgt_i, rng)
            info = dict(info)
            info.update(
                rewintmean_unnorm=int_rews.mean(),
                rewintmax_unnorm=int_rews.max(),
                rewintmean_norm=norm_int.mean(),
                vpredintmean=traj.value_int.mean(),
                vpredextmean=traj.value_ext.mean(),
                advmean=adv.mean(),
            )
            return agent, info, rng
        learn_jit = jax.jit(learn)
        o, _ = env.reset()
        obs = nhwc(jnp.asarray(o))
        ep_ret = jnp.zeros(N, jnp.float32)
        ep_len = jnp.zeros(N, jnp.int32)
        rooms = jnp.zeros(256, bool)
        # obs-norm warmup: uniform-random actions for 128*num_iterations steps
        @jax.jit
        def rand_rollout(handle, obs, rng):
            def body(carry, _):
                handle, obs, rng = carry
                rng, sk = jax.random.split(rng)
                a = jax.random.randint(sk, (N,), 0, num_actions).astype(jnp.int32)
                handle, (nobs, r, te, tr, info) = step_env(handle, a)
                return (handle, nhwc(nobs), rng), obs
            (handle, obs, rng), obs_b = jax.lax.scan(body, (handle, obs, rng), None, length=STEPS)
            return handle, obs, rng, obs_b
        if cfg.update_ob_stats_from_random_agent:
            print(f"Collecting obs-norm warmup: {STEPS * cfg.num_iterations_obs_norm_init} steps.")
            for _ in range(cfg.num_iterations_obs_norm_init):
                handle, obs, rng, obs_b = rand_rollout(handle, obs, rng)
                agent = agent.update_obs_norm(obs_b.reshape((STEPS * N,) + obs_b.shape[2:]))
        t_start = time.time()
        t_last = t_start
        win_sret = win_slen = win_cnt = 0.0
        rooms_seen = set()
        info = None
        for update_idx in range(1, total_updates + 1):
            handle, obs, rng, ep_ret, ep_len, rooms, traj, stats = rollout_jit(agent, handle, obs, rng, ep_ret, ep_len, rooms)
            rng, learn_rng = jax.random.split(rng)
            agent, info, rng = learn_jit(agent, traj, obs, learn_rng)
            win_sret += float(stats["sret"])
            win_slen += float(stats["slen"])
            win_cnt += float(stats["cnt"])
            if update_idx % cfg.log_every_n_updates == 0:
                if room_addr is not None:
                    rooms_seen = set(np.nonzero(np.asarray(rooms))[0].tolist())
                tcount = update_idx * steps_per_update
                now = time.time()
                log = {
                    "tcount": tcount,
                    "tps": tcount / max(now - t_start, 1e-6),
                    "sps": (cfg.log_every_n_updates * steps_per_update) / max(now - t_last, 1e-6),
                    "n_rooms": len(rooms_seen),
                    "rooms": sorted(rooms_seen),
                    "eprew_mean": win_sret / win_cnt if win_cnt else 0.0,
                    "eplen_mean": win_slen / win_cnt if win_cnt else 0.0,
                    **{k: float(v) for k, v in info.items()},
                }
                t_last = now
                win_sret = win_slen = win_cnt = 0.0
                print(f"[{update_idx}/{total_updates}] {log}")
                wandb.log(log, step=tcount)
        wandb.finish()
    run_xla(tyro.cli(Config))
