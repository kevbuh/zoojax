# single file version of https://wang-kevin3290.github.io/scaling-crl/

import os
import jax
import flax
import tyro
import time
import optax
import wandb
import pickle
import random
import functools
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp

import mujoco
import urllib.request
import posixpath
import xml.etree.ElementTree as ET

from pathlib import Path
from brax import envs
from brax import base
from brax import math
from brax import actuator
from etils import epath
from jax import flatten_util
from typing import NamedTuple, Any, Tuple
from dataclasses import dataclass
from brax.io import mjcf
from brax.io import html
from brax.envs.base import PipelineEnv, State
from brax.training.types import PRNGKey
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling

# The inlined env code below is transcribed verbatim from the envs/ package, which
# mixes the `jp` and `jnp` aliases for jax.numpy. Alias them so it pastes faithfully.
jp = jnp

# ==============================================================================
# Zero-dependency asset bootstrap
#
# The env XMLs (and, for the arm/manipulation envs, their referenced .obj/.stl
# meshes) are NOT embedded in this file. On first use they are downloaded from
# the public repo into a local cache dir, mirroring the repo's envs/assets/ tree
# so that mujoco's relative meshdir/texturedir resolution keeps working. Cached
# across runs; network is only needed the first time each asset is fetched.
#
# Override the cache location with $SCALING_CRL_ASSET_CACHE.
# (reacher/pusher load their XML from the installed brax package, not from here.)
# ==============================================================================

_REPO = "wang-kevin3290/scaling-crl"
_REPO_REF = "86c1f08c10d565eddabfcd00302c7b82c5934476"  # pinned to the commit this file was generated from
_REPO_RAW = f"https://raw.githubusercontent.com/{_REPO}/{_REPO_REF}/"
_ASSET_CACHE = Path(os.environ.get("SCALING_CRL_ASSET_CACHE", str(Path.home() / ".cache" / "scaling-crl-assets")))

def _fetch(repo_relpath):
    """Download a single repo file into the local cache (mirroring its path). Returns local Path."""
    local = _ASSET_CACHE / repo_relpath
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        url = _REPO_RAW + repo_relpath
        print(f"[assets] downloading {repo_relpath}", flush=True)
        # Download to a process-unique temp then atomically rename, so concurrent
        # training processes sharing one cache never clobber each other's temp file.
        tmp = local.with_name(f"{local.name}.{os.getpid()}.part")
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, local)
        finally:
            if tmp.exists():
                tmp.unlink()
    return local

def _ensure_xml_assets(local_xml, repo_relpath, _seen=None):
    """Parse an already-downloaded XML and fetch any meshes/textures/includes it references."""
    if _seen is None:
        _seen = set()
    if repo_relpath in _seen:
        return
    _seen.add(repo_relpath)
    xml_dir = posixpath.dirname(repo_relpath)
    try:
        root = ET.parse(str(local_xml)).getroot()
    except ET.ParseError:
        return
    compiler = root.find(".//compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    assetdir = compiler.get("assetdir", "") if compiler is not None else ""
    def _resolve(subdir, fname):
        return posixpath.normpath(posixpath.join(xml_dir, subdir, fname))
    for tag, subdir in [("mesh", meshdir or assetdir), ("skin", meshdir or assetdir), ("texture", texturedir or assetdir), ("hfield", assetdir)]:
        for el in root.iter(tag):
            fname = el.get("file")
            if fname:
                _fetch(_resolve(subdir, fname))
    # Nested <include> files are resolved relative to the XML dir; recurse into them.
    for el in root.iter("include"):
        fname = el.get("file")
        if fname:
            inc_rel = _resolve("", fname)
            inc_local = _fetch(inc_rel)
            _ensure_xml_assets(inc_local, inc_rel, _seen)

def _ensure_asset(repo_relpath):
    """Ensure a repo asset (and, for an XML, its referenced files) is cached. Returns local path str."""
    repo_relpath = repo_relpath.replace("\\", "/").lstrip("./")
    local = _fetch(repo_relpath)
    if local.suffix.lower() == ".xml":
        _ensure_xml_assets(local, repo_relpath)
    return str(local)

# ==============================================================================
# Environments (inlined from the envs/ package). XML assets are downloaded on
# demand via _ensure_asset(); the arm envs additionally pull in mesh files.
# ==============================================================================

class Ant(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        **kwargs,
    ):
        sys = mjcf.load(_ensure_asset("envs/assets/ant.xml"))
        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10
        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        if backend == "positional":
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jp.ones_like(sys.actuator.gear)))
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
        _, target = self._random_target(rng)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}
        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0
        obs = self._get_obs(pipeline_state)
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        dist = jp.linalg.norm(obs[:2] - obs[-2:])
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]
        target_pos = pipeline_state.x.pos[-1][:2]
        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]
        return jp.concatenate([qpos] + [qvel] + [target_pos])
    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = 10
        ang = jp.pi * 2.0 * jax.random.uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        return rng, jp.array([target_x, target_y])

class AntBall(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        **kwargs,
    ):
        sys = mjcf.load(_ensure_asset("envs/assets/ant_ball.xml"))
        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10
        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        if backend == "positional":
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jp.ones_like(sys.actuator.gear)))
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._object_idx = self.sys.link_names.index("object")
        self.state_dim = 31
        self.goal_indices = jp.array([28, 29])
        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
        _, target, obj = self._random_target(rng)
        q = q.at[-4:].set(jp.concatenate([obj, target]))
        qd = qd.at[-4:].set(0)
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}
        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0
        obs = self._get_obs(pipeline_state)
        dist = jp.linalg.norm(obs[-2:] - obs[-4:-2])
        reward = -dist + healthy_reward - ctrl_cost - contact_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)
        state.metrics.update(
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-4]
        qvel = pipeline_state.qd[:-4]
        target_pos = pipeline_state.x.pos[-1][:2]
        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]
        object_position = pipeline_state.x.pos[self._object_idx][:2]
        return jp.concatenate([qpos] + [qvel] + [object_position] + [target_pos])
    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target and object location."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = 5
        ang = jp.pi * 2.0 * jax.random.uniform(rng1)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        ang_obj = jp.pi * 2.0 * jax.random.uniform(rng2)
        obj_x_offset = jp.cos(ang_obj)
        obj_y_offset = jp.sin(ang)
        target_pos = jp.array([target_x, target_y])
        obj_pos = target_pos * 0.2 + jp.array([obj_x_offset, obj_y_offset])
        return rng, target_pos, obj_pos

class AntPush(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="mjx",
        **kwargs,
    ):
        sys = mjcf.load(_ensure_asset("envs/assets/ant_push.xml"))
        n_frames = 5
        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 4,
                    "opt.ls_iterations": 8,
                }
            )
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._object_idx = self.sys.link_names.index("movable")
        self.state_dim = 31
        self.goal_indices = jp.array([0, 1])
        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
        _, target = self._random_target(rng)
        q = q.at[-2:].set(jp.concatenate([target]))
        qd = qd.at[-4:].set(0)
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}
        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0
        obs = self._get_obs(pipeline_state)
        dist = jp.linalg.norm(obs[-2:] - obs[:2])
        reward = -dist + healthy_reward - ctrl_cost - contact_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)
        state.metrics.update(
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-5]
        qvel = pipeline_state.qd[:-5]
        target_pos = pipeline_state.x.pos[-1][:2]
        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]
        object_position = pipeline_state.x.pos[self._object_idx][:2]
        return jp.concatenate([qpos] + [qvel] + [object_position] + [target_pos])
    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        target_x = 16.0
        target_y = 8.0
        target_pos = jax.random.permutation(rng1, jp.array([target_x, target_y]))
        noise = 2.0 * jax.random.uniform(rng2, shape=(2,))
        target_pos += noise
        return rng, target_pos

class Reacher(PipelineEnv):
    def __init__(self, backend="generalized", **kwargs):
        # reacher.xml ships inside the installed brax package, not this repo.
        path = epath.resource_path("brax") / "envs/assets/reacher.xml"
        sys = mjcf.load(path)
        n_frames = 2
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            sys = sys.replace(actuator=sys.actuator.replace(gear=jp.array([25.0, 25.0])))
            n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self.state_dim = 10
        self.goal_indices = jp.array([4, 5, 6])
    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=-0.1, maxval=0.1)
        qd = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=-0.005, maxval=0.005)
        _, target = self._random_target(rng)
        q = q.at[2:].set(target)
        qd = qd.at[2:].set(0)
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {"reward_dist": zero, "reward_ctrl": zero, "success": zero, "dist": zero}
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state
    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}
        target_pos = pipeline_state.x.pos[2]
        tip_pos = pipeline_state.x.take(1).do(base.Transform.create(pos=jp.array([0.11, 0, 0]))).pos
        tip_to_target = target_pos - tip_pos
        dist = jp.linalg.norm(tip_to_target)
        reward_dist = -math.safe_norm(tip_to_target)
        reward = reward_dist
        state.metrics.update(reward_dist=reward_dist, success=jp.array(dist < 0.05, dtype=float), dist=dist)
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns egocentric observation of target and arm body."""
        theta = pipeline_state.q[:2]
        target_pos = pipeline_state.x.pos[2]
        tip_pos = pipeline_state.x.take(1).do(base.Transform.create(pos=jp.array([0.11, 0, 0]))).pos
        tip_vel = base.Transform.create(pos=jp.array([0.11, 0, 0])).do(pipeline_state.xd.take(1)).vel
        return jp.concatenate([jp.cos(theta), jp.sin(theta), tip_pos, tip_vel, target_pos])
    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = 0.2 * jax.random.uniform(rng1)
        ang = jp.pi * 2.0 * jax.random.uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        return rng, jp.array([target_x, target_y])

class Pusher(PipelineEnv):
    def __init__(self, backend="generalized", kind="easy", **kwargs):
        # pusher.xml ships inside the installed brax package, not this repo.
        path = epath.resource_path("brax") / "envs/assets/pusher.xml"
        sys = mjcf.load(path)
        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.001})
            sys = sys.replace(actuator=sys.actuator.replace(gear=jp.array([20.0] * sys.act_size())))
            n_frames = 50
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._tips_arm_idx = self.sys.link_names.index("r_wrist_flex_link")
        self._object_idx = self.sys.link_names.index("object")
        self._goal_idx = self.sys.link_names.index("goal")
        self.kind = kind
        self.state_dim = 20
        self.goal_indices = jp.array([10, 11, 12])
    def reset(self, rng: jax.Array) -> State:
        qpos = self.sys.init_q
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
        cylinder_pos = jp.concatenate(
            [jax.random.uniform(rng, (1,), minval=-0.3, maxval=-1e-6), jax.random.uniform(rng1, (1,), minval=-0.2, maxval=0.2)]
        )
        if self.kind == "hard":
            goal_pos = jp.concatenate(
                [jax.random.uniform(rng2, (1,), minval=-0.65, maxval=0.35), jax.random.uniform(rng3, (1,), minval=-0.55, maxval=0.45)]
            )
        elif self.kind == "easy":
            goal_pos = jp.concatenate(
                [jax.random.uniform(rng2, (1,), minval=-0.3, maxval=-1e-6) - 0.25, jax.random.uniform(rng3, (1,), minval=-0.2, maxval=0.2)]
            )
        norm = math.safe_norm(cylinder_pos - goal_pos)
        scale = jp.where(norm < 0.17, 0.17 / norm, 1.0)
        cylinder_pos *= scale
        qpos = qpos.at[-4:].set(jp.concatenate([cylinder_pos, goal_pos]))
        qvel = jax.random.uniform(rng4, (self.sys.qd_size(),), minval=-0.005, maxval=0.005)
        qvel = qvel.at[-4:].set(0.0)
        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {"reward_dist": zero, "reward_ctrl": zero, "reward_near": zero, "success": zero, "success_hard": zero}
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state
    def step(self, state: State, action: jax.Array) -> State:
        assert state.pipeline_state is not None
        x_i = state.pipeline_state.x.vmap().do(base.Transform.create(pos=self.sys.link.inertia.transform.pos))
        vec_1 = x_i.pos[self._object_idx] - x_i.pos[self._tips_arm_idx]
        vec_2 = x_i.pos[self._object_idx] - x_i.pos[self._goal_idx]
        obj_to_goal_dist = math.safe_norm(vec_2)
        reward_near = -math.safe_norm(vec_1)
        reward_dist = -obj_to_goal_dist
        reward_ctrl = -jp.square(action).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}
        obs = self._get_obs(pipeline_state)
        state.metrics.update(
            reward_near=reward_near,
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            success=jp.array(obj_to_goal_dist < 0.1, dtype=float),
            success_hard=jp.array(obj_to_goal_dist < 0.05, dtype=float),
        )
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observes pusher body position and velocities."""
        x_i = pipeline_state.x.vmap().do(base.Transform.create(pos=self.sys.link.inertia.transform.pos))
        return jp.concatenate(
            [pipeline_state.q[:7], x_i.pos[self._tips_arm_idx], x_i.pos[self._object_idx], pipeline_state.qd[:7], x_i.pos[self._goal_idx]]
        )

# Height of the humanoid torso target (fixed); shared by Humanoid + HumanoidMaze.
TARGET_Z_COORD = 1.25

class Humanoid(PipelineEnv):
    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=0.0,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        min_goal_dist=1.0,
        max_goal_dist=5.0,
        **kwargs,
    ):
        sys = mjcf.load(_ensure_asset("envs/assets/humanoid.xml"))
        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.0015})
            n_frames = 10
            gear = jp.array(
                [350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
            )  # pyformat: disable
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))
        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._target_ind = self.sys.link_names.index("target")
        self._min_goal_dist = min_goal_dist
        self._max_goal_dist = max_goal_dist
        self.state_dim = 268
        self.goal_indices = jp.array([0, 1, 2])
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)
        _, target = self._random_target(rng)
        qpos = qpos.at[-2:].set(target)
        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "dist": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "success": zero,
            "success_easy": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state
    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        obs = self._get_obs(pipeline_state, action)
        distance_to_target = jp.linalg.norm(obs[:3] - obs[-3:])
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        reward = -distance_to_target + healthy_reward - ctrl_cost
        success = jp.array(distance_to_target < 0.5, dtype=float)
        success_easy = jp.array(distance_to_target < 2.0, dtype=float)
        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.linalg.norm(com_after),
            dist=distance_to_target,
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            success=success,
            success_easy=success_easy,
        )
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""
        position = pipeline_state.q
        velocity = pipeline_state.qd
        if self._exclude_current_positions_from_observation:
            position = position[2:]
        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack([cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]])
        xd_i = base.Transform.create(pos=x_i.pos - pipeline_state.x.pos).vmap().do(pipeline_state.xd)
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])
        qfrc_actuator = actuator.to_tau(self.sys, action, pipeline_state.q, pipeline_state.qd)
        target_pos = pipeline_state.x.pos[-1][:2]
        return jp.concatenate([position, velocity, com_inertia.ravel(), com_velocity.ravel(), qfrc_actuator, target_pos, jp.array([TARGET_Z_COORD])])
    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        if self.backend in ["spring", "positional"]:
            inertia = inertia.replace(
                i=jax.vmap(jp.diag)(jax.vmap(jp.diagonal)(inertia.i) ** (1 - self.sys.spring_inertia_scale)),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        return com, inertia, mass_sum, x_i
    def _random_target(self, rng: jax.Array):
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = jax.random.uniform(rng1, minval=self._min_goal_dist, maxval=self._max_goal_dist)
        ang = jp.pi * 2.0 * jax.random.uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        return rng, jp.array([target_x, target_y])

# ==============================================================================
# Maze envs. Each family (AntMaze / HumanoidMaze / AntMazeGeneralization) defines
# its own maze-layout tables and make_maze() in the original files with colliding
# names; here they are namespaced with AM_/HM_/AMG_ prefixes.
# ==============================================================================

_MAZE_RESET = _MAZE_R = "r"
_MAZE_GOAL = _MAZE_G = "g"
_MAZE_HEIGHT = 0.5

def _maze_add_blocks(tree, maze_layout, maze_size_scaling):
    """Shared: append box geoms for every wall cell (== 1) into the XML worldbody."""
    worldbody = tree.find(".//worldbody")
    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            if maze_layout[i][j] == 1:
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (i * maze_size_scaling, j * maze_size_scaling, _MAZE_HEIGHT / 2 * maze_size_scaling),
                    size="%f %f %f" % (0.5 * maze_size_scaling, 0.5 * maze_size_scaling, _MAZE_HEIGHT / 2 * maze_size_scaling),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

# ----- AntMaze -----
_R, _G = _MAZE_R, _MAZE_G
AM_LAYOUTS = {
    "u_maze": [[1, 1, 1, 1, 1], [1, _R, _G, _G, 1], [1, 1, 1, _G, 1], [1, _G, _G, _G, 1], [1, 1, 1, 1, 1]],
    "u_maze_eval": [[1, 1, 1, 1, 1], [1, _R, 0, 0, 1], [1, 1, 1, 0, 1], [1, _G, _G, _G, 1], [1, 1, 1, 1, 1]],
    "u_maze_single_eval": [[1, 1, 1, 1, 1], [1, _R, 0, 0, 1], [1, 1, 1, 0, 1], [1, _G, 0, 0, 1], [1, 1, 1, 1, 1]],
    "u_maze_eval_1f2f3f4f5f": [[1, 1, 1, 1, 1], [1, _R, _G, _G, 1], [1, 1, 1, _G, 1], [1, 0, _G, _G, 1], [1, 1, 1, 1, 1]],
    "u_maze_eval_1f2f3f4f": [[1, 1, 1, 1, 1], [1, _R, _G, _G, 1], [1, 1, 1, _G, 1], [1, 0, 0, _G, 1], [1, 1, 1, 1, 1]],
    "u_maze_eval_1f2f3f": [[1, 1, 1, 1, 1], [1, _R, _G, _G, 1], [1, 1, 1, _G, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
    "u_maze_eval_5f6f": [[1, 1, 1, 1, 1], [1, _R, 0, 0, 1], [1, 1, 1, 0, 1], [1, _G, _G, 0, 1], [1, 1, 1, 1, 1]],
    "u2_maze": [[1, 1, 1, 1, 1, 1], [1, _R, _G, _G, _G, 1], [1, 1, 1, 1, _G, 1], [1, _G, _G, _G, _G, 1], [1, 1, 1, 1, 1, 1]],
    "u2_maze_eval": [[1, 1, 1, 1, 1, 1], [1, _R, 0, 0, 0, 1], [1, 1, 1, 1, 0, 1], [1, _G, _G, _G, _G, 1], [1, 1, 1, 1, 1, 1]],
    "u3_maze": [[1, 1, 1, 1, 1, 1, 1], [1, _R, _G, _G, _G, _G, 1], [1, 1, 1, 1, 1, _G, 1], [1, _G, _G, _G, _G, _G, 1], [1, 1, 1, 1, 1, 1, 1]],
    "u3_maze_eval": [[1, 1, 1, 1, 1, 1, 1], [1, _R, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 0, 1], [1, _G, _G, _G, _G, _G, 1], [1, 1, 1, 1, 1, 1, 1]],
    "u3_maze_single_eval": [[1, 1, 1, 1, 1, 1, 1], [1, _R, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 0, 1], [1, _G, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1]],
    "u4_maze": [[1, 1, 1, 1, 1], [1, _G, _G, _G, 1], [1, _R, 1, _G, 1], [1, 1, 1, _G, 1], [1, _G, 1, _G, 1], [1, _G, _G, _G, 1], [1, 1, 1, 1, 1]],
    "u4_maze_eval": [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, _R, 1, 0, 1], [1, 1, 1, 0, 1], [1, _G, 1, 0, 1], [1, _G, _G, _G, 1], [1, 1, 1, 1, 1]],
    "u5_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, _G, _G, _G, _G, _G, _G, 1],
        [1, _R, 1, 1, 1, 1, _G, 1],
        [1, 1, 1, 1, 1, 1, _G, 1],
        [1, _G, 1, 1, 1, 1, _G, 1],
        [1, _G, _G, _G, _G, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "u5_maze_eval": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, _R, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, _G, 1, 1, 1, 1, _G, 1],
        [1, _G, _G, _G, _G, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "u5_maze_single_eval": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, _R, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, _G, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "u6_maze": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, _G, _G, _G, _G, _G, 1],
        [1, _R, 1, 1, 1, _G, 1],
        [1, 1, 1, 1, 1, _G, 1],
        [1, _G, 1, 1, 1, _G, 1],
        [1, _G, _G, _G, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    "u6_maze_eval": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, _R, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 0, 1],
        [1, _G, 1, 1, 1, _G, 1],
        [1, _G, _G, _G, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    "u7_maze": [
        [1, 1, 1, 1, 1, 1],
        [1, _G, _G, _G, _G, 1],
        [1, _R, 1, 1, _G, 1],
        [1, 1, 1, 1, _G, 1],
        [1, _G, 1, 1, _G, 1],
        [1, _G, _G, _G, _G, 1],
        [1, 1, 1, 1, 1, 1],
    ],
    "u7_maze_eval": [
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, _R, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 1],
        [1, _G, 1, 1, _G, 1],
        [1, _G, _G, _G, _G, 1],
        [1, 1, 1, 1, 1, 1],
    ],
    "big_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, _R, _G, 1, 1, _G, _G, 1],
        [1, _G, _G, 1, _G, _G, _G, 1],
        [1, 1, _G, _G, _G, 1, 1, 1],
        [1, _G, _G, 1, _G, _G, _G, 1],
        [1, _G, 1, _G, _G, 1, _G, 1],
        [1, _G, _G, _G, 1, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "big_maze_eval": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, _R, 0, 1, 1, _G, _G, 1],
        [1, 0, 0, 1, 0, 0, _G, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, _G, 0, 1, _G, 1],
        [1, 0, _G, _G, 1, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "hardest_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, _R, _G, _G, _G, 1, _G, _G, _G, _G, _G, 1],
        [1, _G, 1, 1, _G, 1, _G, 1, _G, 1, _G, 1],
        [1, _G, _G, _G, _G, _G, _G, 1, _G, _G, _G, 1],
        [1, _G, 1, 1, 1, 1, _G, 1, 1, 1, _G, 1],
        [1, _G, _G, 1, _G, 1, _G, _G, _G, _G, _G, 1],
        [1, 1, _G, 1, _G, 1, _G, 1, _G, 1, 1, 1],
        [1, _G, _G, 1, _G, _G, _G, 1, _G, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
}
del _R, _G

def _antmaze_find_robot(structure, size_scaling):
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == _MAZE_RESET:
                return i * size_scaling, j * size_scaling

def _antmaze_find_goals(structure, size_scaling):
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == _MAZE_GOAL:
                goals.append([i * size_scaling, j * size_scaling])
    return jp.array(goals)

def _antmaze_make_maze(maze_layout_name, maze_size_scaling):
    if maze_layout_name not in AM_LAYOUTS:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")
    maze_layout = AM_LAYOUTS[maze_layout_name]
    xml_path = _ensure_asset("envs/assets/ant_maze.xml")
    robot_x, robot_y = _antmaze_find_robot(maze_layout, maze_size_scaling)
    possible_goals = _antmaze_find_goals(maze_layout, maze_size_scaling)
    tree = ET.parse(xml_path)
    _maze_add_blocks(tree, maze_layout, maze_size_scaling)
    torso = tree.find(".//numeric[@name='init_qpos']")
    data = torso.get("data")
    torso.set("data", f"{robot_x} {robot_y} " + data)
    xml_string = ET.tostring(tree.getroot())
    return xml_string, possible_goals

class AntMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        **kwargs,
    ):
        xml_string, possible_goals = _antmaze_make_maze(maze_layout_name, maze_size_scaling)
        sys = mjcf.loads(xml_string)
        self.possible_goals = possible_goals
        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10
        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        if backend == "positional":
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jp.ones_like(sys.actuator.gear)))
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
        _, target = self._random_target(rng)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}
        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0
        obs = self._get_obs(pipeline_state)
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        dist = jp.linalg.norm(obs[:2] - obs[-2:])
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)
        reward = -dist + healthy_reward - ctrl_cost - contact_cost
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]
        target_pos = pipeline_state.x.pos[-1][:2]
        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]
        return jp.concatenate([qpos] + [qvel] + [target_pos])
    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a random target location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return rng, jp.array(self.possible_goals[idx])[0]

# ----- HumanoidMaze -----
_R, _G = _MAZE_R, _MAZE_G
HM_LAYOUTS = {
    "u_maze": [[1, 1, 1, 1, 1], [1, _R, _G, _G, 1], [1, 1, 1, _G, 1], [1, _G, _G, _G, 1], [1, 1, 1, 1, 1]],
    "u_maze_eval": [[1, 1, 1, 1, 1], [1, _R, 0, 0, 1], [1, 1, 1, 0, 1], [1, _G, _G, _G, 1], [1, 1, 1, 1, 1]],
    "big_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, _R, _G, 1, 1, _G, _G, 1],
        [1, _G, _G, 1, _G, _G, _G, 1],
        [1, 1, _G, _G, _G, 1, 1, 1],
        [1, _G, _G, 1, _G, _G, _G, 1],
        [1, _G, 1, _G, _G, 1, _G, 1],
        [1, _G, _G, _G, 1, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "big_maze_eval": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, _R, 0, 1, 1, _G, _G, 1],
        [1, 0, 0, 1, 0, _G, _G, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, _G, 0, 1, _G, 1],
        [1, 0, _G, _G, 1, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "hardest_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, _R, _G, _G, _G, 1, _G, _G, _G, _G, _G, 1],
        [1, _G, 1, 1, _G, 1, _G, 1, _G, 1, _G, 1],
        [1, _G, _G, _G, _G, _G, _G, 1, _G, _G, _G, 1],
        [1, _G, 1, 1, 1, 1, _G, 1, 1, 1, _G, 1],
        [1, _G, _G, 1, _G, 1, _G, _G, _G, _G, _G, 1],
        [1, 1, _G, 1, _G, 1, _G, 1, _G, 1, 1, 1],
        [1, _G, _G, 1, _G, _G, _G, 1, _G, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
}
del _R, _G

def _hmaze_find_starts(structure, size_scaling):
    starts = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == _MAZE_RESET:
                starts.append([i * size_scaling, j * size_scaling])
    return jnp.array(starts)

def _hmaze_find_goals(structure, size_scaling):
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == _MAZE_GOAL:
                goals.append([i * size_scaling, j * size_scaling])
    return jnp.array(goals)

def _hmaze_make_maze(maze_layout_name, maze_size_scaling):
    if maze_layout_name not in HM_LAYOUTS:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")
    maze_layout = HM_LAYOUTS[maze_layout_name]
    xml_path = _ensure_asset("envs/assets/humanoid_maze.xml")
    possible_starts = _hmaze_find_starts(maze_layout, maze_size_scaling)
    possible_goals = _hmaze_find_goals(maze_layout, maze_size_scaling)
    tree = ET.parse(xml_path)
    _maze_add_blocks(tree, maze_layout, maze_size_scaling)
    xml_string = ET.tostring(tree.getroot())
    return xml_string, possible_starts, possible_goals

class HumanoidMaze(PipelineEnv):
    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=0.0,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=2.0,
        **kwargs,
    ):
        xml_string, possible_starts, possible_goals = _hmaze_make_maze(maze_layout_name, maze_size_scaling)
        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals
        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.0015})
            n_frames = 10
            gear = jnp.array(
                [350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
            )  # pyformat: disable
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))
        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._target_ind = self.sys.link_names.index("target")
        self.state_dim = 268
        self.goal_indices = jnp.array([0, 1, 2])
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(rng1, [self.sys.q_size()], minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, [self.sys.qd_size()], minval=low, maxval=hi)
        start = self._random_start(rng3)
        qpos = qpos.at[:2].set(start)
        target = self._random_target(rng)
        qpos = qpos.at[-2:].set(target)
        qvel = qvel.at[-2:].set(0)
        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jnp.zeros(self.sys.act_size()))
        reward, done, zero = jnp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "dist": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "success": zero,
            "success_easy": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state
    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jnp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]
        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        obs = self._get_obs(pipeline_state, action)
        distance_to_target = jnp.linalg.norm(obs[:3] - obs[-3:])
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        reward = -distance_to_target + healthy_reward - ctrl_cost
        success = jnp.array(distance_to_target < 0.5, dtype=float)
        success_easy = jnp.array(distance_to_target < 2.0, dtype=float)
        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jnp.linalg.norm(com_after),
            dist=distance_to_target,
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            success=success,
            success_easy=success_easy,
        )
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""
        position = pipeline_state.q
        velocity = pipeline_state.qd
        if self._exclude_current_positions_from_observation:
            position = position[2:]
        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jnp.hstack([cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]])
        xd_i = base.Transform.create(pos=x_i.pos - pipeline_state.x.pos).vmap().do(pipeline_state.xd)
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jnp.hstack([com_vel, com_ang])
        qfrc_actuator = actuator.to_tau(self.sys, action, pipeline_state.q, pipeline_state.qd)
        target_pos = pipeline_state.x.pos[-1][:2]
        return jnp.concatenate(
            [position, velocity, com_inertia.ravel(), com_velocity.ravel(), qfrc_actuator, target_pos, jnp.array([TARGET_Z_COORD])]
        )
    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        if self.backend in ["spring", "positional"]:
            inertia = inertia.replace(
                i=jax.vmap(jnp.diag)(jax.vmap(jnp.diagonal)(inertia.i) ** (1 - self.sys.spring_inertia_scale)),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jnp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = jnp.sum(jax.vmap(jnp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        return com, inertia, mass_sum, x_i
    def _random_target(self, rng: jax.Array) -> jax.Array:
        """Returns a random target location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return jnp.array(self.possible_goals[idx])[0]
    def _random_start(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return jnp.array(self.possible_starts[idx])[0]

# ----- AntMazeGeneralization -----
_R, _G = _MAZE_R, _MAZE_G
AMG_LAYOUTS = {
    "u_maze": [[1, 1, 1, 1, 1], [1, _R, 0, 0, 1], [1, 1, 1, 0, 1], [1, _G, 0, 0, 1], [1, 1, 1, 1, 1]],
    "u2_maze": [[1, 1, 1, 1, 1, 1], [1, _R, 0, 0, 0, 1], [1, 1, 1, 1, 0, 1], [1, _G, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]],
    "u3_maze": [[1, 1, 1, 1, 1, 1, 1], [1, _R, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 0, 1], [1, _G, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1]],
    "u4_maze": [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, _R, 1, 0, 1], [1, 1, 1, 0, 1], [1, _G, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
    "u5_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, _R, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, _G, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "big_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, _R, _G, 1, 1, _G, _G, 1],
        [1, _G, _G, 1, _G, _G, _G, 1],
        [1, 1, _G, _G, _G, 1, 1, 1],
        [1, _G, _G, 1, _G, _G, _G, 1],
        [1, _G, 1, _G, _G, 1, _G, 1],
        [1, _G, _G, _G, 1, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "hardest_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, _R, _G, _G, _G, 1, _G, _G, _G, _G, _G, 1],
        [1, _G, 1, 1, _G, 1, _G, 1, _G, 1, _G, 1],
        [1, _G, _G, _G, _G, _G, _G, 1, _G, _G, _G, 1],
        [1, _G, 1, 1, 1, 1, _G, 1, 1, 1, _G, 1],
        [1, _G, _G, 1, _G, 1, _G, _G, _G, _G, _G, 1],
        [1, 1, _G, 1, _G, 1, _G, 1, _G, 1, 1, 1],
        [1, _G, _G, 1, _G, _G, _G, 1, _G, _G, _G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
}
del _R, _G

def _amg_get_forward_path(maze_layout):
    start, end = None, None
    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            if maze_layout[i][j] == _MAZE_RESET:
                start = (i, j)
            elif maze_layout[i][j] == _MAZE_GOAL:
                end = (i, j)
    return _amg_dfs(maze_layout, start, end)

def _amg_dfs(maze_layout, start, end):
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    prev_x, prev_y = None, None
    curr_x, curr_y = start
    path = []
    while not (curr_x == end[0] and curr_y == end[1]):
        path.append((curr_x, curr_y))
        for direction in range(4):
            next_x, next_y = curr_x + dx[direction], curr_y + dy[direction]
            assert not (next_x < 0 or next_x >= len(maze_layout) or next_y < 0 or next_y >= len(maze_layout[0]))
            if maze_layout[next_x][next_y] == 1:
                continue
            if next_x == prev_x and next_y == prev_y:
                continue
            prev_x, prev_y = curr_x, curr_y
            curr_x, curr_y = next_x, next_y
            break
    path.append(end)
    return path

def _amg_get_start_goal(maze_layout, generalization_config, rng):
    sg_pairs = []
    forward_path = _amg_get_forward_path(maze_layout)
    num_valid_pairs = sum([len(forward_path) - i for i in range(1, 6) if f"{i}f" in generalization_config])
    num_distances = len(generalization_config.split("f")[:-1])
    weights = []
    for config in generalization_config.split("f")[:-1]:
        config = int(config)
        pairs = []
        for i in range(len(forward_path) - config):
            pairs.append((forward_path[i], forward_path[i + config]))
            weight = num_valid_pairs / num_distances / (len(forward_path) - config)
            weights.append(weight)
        sg_pairs.extend(pairs)
    print(f"num_valid_pairs: {num_valid_pairs}, sg_pairs: {sg_pairs}, weights: {weights}", flush=True)
    sg_pairs = jp.array(sg_pairs)
    weights = jp.array(weights)
    idx = jax.random.choice(rng, len(sg_pairs), p=weights)
    random_pair = jp.array(sg_pairs[idx])
    return random_pair

def _amg_get_maze_layout(maze_layout_name):
    if maze_layout_name not in AMG_LAYOUTS:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")
    return AMG_LAYOUTS[maze_layout_name]

def _amg_make_maze(maze_layout, maze_size_scaling):
    xml_path = _ensure_asset("envs/assets/ant_maze.xml")
    tree = ET.parse(xml_path)
    _maze_add_blocks(tree, maze_layout, maze_size_scaling)
    torso = tree.find(".//numeric[@name='init_qpos']")
    data = torso.get("data")
    torso.set("data", f"{0} {0} " + data)
    xml_string = ET.tostring(tree.getroot())
    return xml_string

class AntMazeGeneralization(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        generalization_config="1f",
        **kwargs,
    ):
        self.maze_layout = _amg_get_maze_layout(maze_layout_name)
        self.maze_size_scaling = maze_size_scaling
        self.generalization_config = generalization_config
        xml_string = _amg_make_maze(self.maze_layout, self.maze_size_scaling)
        sys = mjcf.loads(xml_string)
        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10
        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        if backend == "positional":
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jp.ones_like(sys.actuator.gear)))
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        start, goal = _amg_get_start_goal(self.maze_layout, self.generalization_config, rng3)
        print(f"start: {start}, goal: {goal}", flush=True)
        start_pos = jp.array([start[0] * self.maze_size_scaling, start[1] * self.maze_size_scaling])
        goal_pos = jp.array([goal[0] * self.maze_size_scaling, goal[1] * self.maze_size_scaling])
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
        q = q.at[:2].set(start_pos)
        q = q.at[-2:].set(goal_pos)
        qd = qd.at[-2:].set(0)
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}
        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0
        obs = self._get_obs(pipeline_state)
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        dist = jp.linalg.norm(obs[:2] - obs[-2:])
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)
        reward = -dist + healthy_reward - ctrl_cost - contact_cost
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]
        target_pos = pipeline_state.x.pos[-1][:2]
        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]
        return jp.concatenate([qpos] + [qvel] + [target_pos])

# ==============================================================================
# Arm / manipulation envs. The panda XMLs reference dozens of external .obj/.stl
# mesh files (via meshdir="franka_emika_panda/assets"); _ensure_asset() downloads
# the XML and all referenced meshes into the mirrored cache before mjcf.load().
# ==============================================================================

class ArmEnvs(PipelineEnv):
    def __init__(self, backend="mjx", **kwargs):
        self._set_environment_attributes()
        sys = mjcf.load(_ensure_asset(self._get_xml_path()))
        sys = sys.tree_replace({"opt.timestep": 0.002, "opt.iterations": 6, "opt.ls_iterations": 12})
        self.n_frames = 25
        kwargs["n_frames"] = kwargs.get("n_frames", self.n_frames)
        if backend != "mjx":
            raise Exception("Use the mjx backend for stability/reasonable speed.")
        super().__init__(sys=sys, backend=backend, **kwargs)
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, subkey = jax.random.split(rng)
        q, qd = self._get_initial_state(subkey)
        pipeline_state = self.pipeline_init(q, qd)
        timestep = 0.0
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        goal = self._get_initial_goal(pipeline_state, subkey1)
        pipeline_state = self._update_goal_visualization(pipeline_state, goal)
        info = {"seed": 0, "goal": goal, "timestep": 0.0, "postexplore_timestep": jax.random.uniform(subkey2)}
        obs = self._get_obs(pipeline_state, goal, timestep)
        reward, done, zero = jnp.zeros(3)
        metrics = {"success": zero, "success_easy": zero, "success_hard": zero}
        return State(pipeline_state, obs, reward, done, metrics, info)
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        if "EEF" in self.env_name:
            action = self._convert_action_to_actuator_input_EEF(pipeline_state0, action)
        else:
            arm_angles = self._get_arm_angles(pipeline_state0)
            action = self._convert_action_to_actuator_input_joint_angle(action, arm_angles, delta_control=False)
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jnp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        timestep = state.info["timestep"] + 1 / self.episode_length
        obs = self._get_obs(pipeline_state, state.info["goal"], timestep)
        success, success_easy, success_hard = self._compute_goal_completion(obs, state.info["goal"])
        state.metrics.update(success=success, success_easy=success_easy, success_hard=success_hard)
        reward = success
        done = 0.0
        info = {**state.info, "timestep": timestep, "seed": seed}
        new_state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=info)
        if self.env_name == "arm_grasp":
            cube_pos = obs[:3]
            left_finger_goal_pos = cube_pos + jnp.array([0.0375, 0, 0])
            right_finger_goal_pos = cube_pos + jnp.array([-0.0375, 0, 0])
            adjusted_goal = state.info["goal"].at[:6].set(jnp.concatenate([left_finger_goal_pos] + [right_finger_goal_pos]))
            new_state = self.update_goal(new_state, adjusted_goal)
        return new_state
    def update_goal(self, state: State, goal: jax.Array) -> State:
        info = {**state.info, "goal": goal}
        pipeline_state = self._update_goal_visualization(state.pipeline_state, goal)
        return state.replace(pipeline_state=pipeline_state, info=info)
    def _convert_action_to_actuator_input_joint_angle(self, action: jax.Array, arm_angles: jax.Array, delta_control=False) -> jax.Array:
        arm_action = jnp.array([action[0], action[1], 0, action[2], 0, action[3], 0])
        min_value = jnp.array([0.3491, 0, 0, -3.0718, 0, 2.3562, 1.4487])
        max_value = jnp.array([2.7925, 1.48353, 0, -0.0698, 0, 3.7525, 1.4487])
        offset = (min_value + max_value) / 2
        multiplier = (max_value - min_value) / 2
        if delta_control:
            normalized_arm_angles = jnp.where(multiplier > 0, (arm_angles - offset) / multiplier, 0)
            delta_range = 0.5
            arm_action = normalized_arm_angles + arm_action * delta_range
            arm_action = jnp.clip(arm_action, -1, 1)
        arm_action = offset + arm_action * multiplier
        if self.env_name not in ("arm_reach"):
            gripper_action = jnp.where(action[-1] > 0, jnp.array([0, 0], dtype=float), jnp.array([255, 255], dtype=float))
            converted_action = jnp.concatenate([arm_action] + [gripper_action])
        else:
            converted_action = arm_action
        return converted_action
    def _convert_action_to_actuator_input_EEF(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        eef_index = 2
        current_position = pipeline_state.x.pos[eef_index]
        delta_range = 0.2
        arm_action = current_position + delta_range * jnp.clip(action[:3], -1, 1)
        gripper_action = jnp.where(action[-1] > 0, jnp.array([0, 0], dtype=float), jnp.array([255, 255], dtype=float))
        converted_action = jnp.concatenate([arm_action] + [gripper_action])
        return converted_action
    def _get_xml_path(self):
        raise NotImplementedError
    def _set_environment_attributes(self):
        raise NotImplementedError
    def _get_initial_state(self, rng):
        raise NotImplementedError
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        raise NotImplementedError
    def _compute_goal_completion(self, obs, goal):
        raise NotImplementedError
    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        raise NotImplementedError
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        raise NotImplementedError
    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        raise NotImplementedError

class ArmReach(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_reach.xml"
    @property
    def action_size(self) -> int:
        return 4
    def _set_environment_attributes(self):
        self.env_name = "arm_reach"
        self.episode_length = 50
        self.goal_indices = jnp.array([7, 8, 9])
        self.completion_goal_indices = jnp.array([7, 8, 9])
        self.state_dim = 13
        self.arm_noise_scale = 0
        self.goal_noise_scale = 0.2
    def _get_initial_state(self, rng):
        target_q = self.sys.init_q[:7]
        arm_q_default = jnp.array([1.571, 0.742, 0, -1.571, 0, 3.054, 1.449])
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(rng, [self.sys.q_size() - 7], minval=-1)
        q = jnp.concatenate([target_q] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        goal = jnp.array([0, 0.5, 0.3]) + self.goal_noise_scale * jax.random.uniform(rng, [3], minval=-1)
        return goal
    def _compute_goal_completion(self, obs, goal):
        eef_pos = obs[self.completion_goal_indices]
        goal_eef_pos = goal[:3]
        dist = jnp.linalg.norm(eef_pos - goal_eef_pos)
        success = jnp.array(dist < 0.1, dtype=float)
        success_easy = jnp.array(dist < 0.3, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard
    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[:3].set(goal)
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        q_subset = pipeline_state.q[7:14]
        eef_index = 7
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        return jnp.concatenate([q_subset] + [eef_x_pos] + [eef_xd_vel] + [goal])
    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(7, 14)
        return pipeline_state.q[q_indices]

class ArmBinpickEasy(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_binpick_easy.xml"
    @property
    def action_size(self) -> int:
        return 5
    def _set_environment_attributes(self):
        self.env_name = "arm_binpick_easy"
        self.episode_length = 80
        self.goal_indices = jnp.array([0, 1, 2])
        self.completion_goal_indices = jnp.array([0, 1, 2])
        self.state_dim = 17
        self.arm_noise_scale = 0
        self.cube_noise_scale = 0.09
        self.goal_noise_scale = 0.09
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]
        arm_q_default = jnp.array([1.571, 0.742, 0, -1.571, 0, 3.054, 1.449, 0.04, 0.04])
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 14], minval=-1)
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [target_q] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array([0.17, 0.6, 0.03]) + jnp.array([self.goal_noise_scale, self.goal_noise_scale, 0]) * jax.random.uniform(
            subkey, [3], minval=-1
        )
        return cube_goal_pos
    def _compute_goal_completion(self, obs, goal):
        current_cube_pos = obs[self.completion_goal_indices]
        goal_pos = goal[:3]
        dist = jnp.linalg.norm(current_cube_pos - goal_pos)
        success = jnp.array(dist < 0.1, dtype=float)
        success_easy = jnp.array(dist < 0.3, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard
    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[7:10].set(goal[:3])
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        q_indices = jnp.array([0, 1, 2, 14, 15, 16, 17, 18, 19, 20])
        q_subset = pipeline_state.q[q_indices]
        eef_index = 8
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        left_finger_index = 9
        left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
        right_finger_index = 10
        right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
        finger_distance = jnp.linalg.norm(right_finger_x_pos - left_finger_x_pos)[None]
        return jnp.concatenate([q_subset] + [eef_x_pos] + [eef_xd_vel] + [finger_distance] + [goal])
    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(14, 21)
        return pipeline_state.q[q_indices]

class ArmBinpickHard(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_binpick_hard.xml"
    @property
    def action_size(self) -> int:
        return 5
    def _set_environment_attributes(self):
        self.env_name = "arm_binpick_hard"
        self.episode_length = 80
        self.goal_indices = jnp.array([0, 1, 2])
        self.completion_goal_indices = jnp.array([0, 1, 2])
        self.state_dim = 17
        self.arm_noise_scale = 0
        self.cube_noise_scale = 0.15
        self.goal_noise_scale = 0.15
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]
        arm_q_default = jnp.array([1.571, 0.742, 0, -1.571, 0, 3.054, 1.449, 0.04, 0.04])
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 14], minval=-1)
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [target_q] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array([0.3, 0.6, 0.03]) + jnp.array([self.goal_noise_scale, self.goal_noise_scale, 0]) * jax.random.uniform(
            subkey, [3], minval=-1
        )
        return cube_goal_pos
    def _compute_goal_completion(self, obs, goal):
        current_cube_pos = obs[self.completion_goal_indices]
        goal_pos = goal[:3]
        dist = jnp.linalg.norm(current_cube_pos - goal_pos)
        success = jnp.array(dist < 0.1, dtype=float)
        success_easy = jnp.array(dist < 0.3, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard
    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[7:10].set(goal[:3])
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        q_indices = jnp.array([0, 1, 2, 14, 15, 16, 17, 18, 19, 20])
        q_subset = pipeline_state.q[q_indices]
        eef_index = 8
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        left_finger_index = 9
        left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
        right_finger_index = 10
        right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
        finger_distance = jnp.linalg.norm(right_finger_x_pos - left_finger_x_pos)[None]
        return jnp.concatenate([q_subset] + [eef_x_pos] + [eef_xd_vel] + [finger_distance] + [goal])
    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(14, 21)
        return pipeline_state.q[q_indices]

class ArmBinpickEasyEEF(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_binpick_easy_EEF.xml"
    @property
    def action_size(self) -> int:
        return 4
    def _set_environment_attributes(self):
        self.env_name = "arm_binpick_easy_EEF"
        self.episode_length = 150
        self.goal_indices = jnp.array([0, 1, 2])
        self.completion_goal_indices = jnp.array([0, 1, 2])
        self.state_dim = 11
        self.goal_dist = 0.1
        self.eef_noise_scale = 0
        self.cube_noise_scale = 0.07
        self.goal_noise_scale = 0.005
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]
        eef_q_default = jnp.array([0, 0.6, 0.2, 0.04, 0.04])
        eef_q = eef_q_default + self.eef_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 14], minval=-1)
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [target_q] + [eef_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array([0.17, 0.6, 0.03]) + jnp.array([self.goal_noise_scale, self.goal_noise_scale, 0]) * jax.random.uniform(
            subkey, [3], minval=-1
        )
        return cube_goal_pos
    def _compute_goal_completion(self, obs, goal):
        current_cube_pos = obs[self.completion_goal_indices]
        goal_pos = goal[:3]
        dist = jnp.linalg.norm(current_cube_pos - goal_pos)
        success = jnp.array(dist < self.goal_dist, dtype=float)
        success_easy = jnp.array(dist < 0.3, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard
    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[7:10].set(goal[:3])
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        q_indices = jnp.array([0, 1, 2])
        q_subset = pipeline_state.q[q_indices]
        eef_index = 2
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        left_finger_index = 3
        left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
        right_finger_index = 4
        right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
        finger_distance = jnp.linalg.norm(right_finger_x_pos - left_finger_x_pos, keepdims=True)
        gripper_force = (pipeline_state.qfrc_actuator[:-2]).mean(keepdims=True) * 0.1
        return jnp.concatenate([q_subset] + [eef_x_pos] + [eef_xd_vel] + [finger_distance] + [gripper_force] + [goal])

class ArmGrasp(ArmEnvs):
    def __init__(self, cube_noise_scale=0.3, **kwargs):
        super().__init__(**kwargs)
        self.cube_noise_scale = cube_noise_scale
    def _get_xml_path(self):
        return "envs/assets/panda_grasp.xml"
    @property
    def action_size(self) -> int:
        return 5
    def _set_environment_attributes(self):
        self.env_name = "arm_grasp"
        self.episode_length = 50
        self.goal_indices = jnp.array([16, 17, 18, 19, 20, 21, 22])
        self.completion_goal_indices = jnp.array([16, 17, 18, 19, 20, 21, 22])
        self.state_dim = 23
        self.arm_noise_scale = 0
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]
        arm_q_default = jnp.array([1.571, 0.742, 0, -1.571, 0, 3.054, 1.449, 0.04, 0.04, 0, 0])
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 14], minval=-1)
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [target_q] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        cube_pos = pipeline_state.q[:3]
        left_fingertip_goal_pos = cube_pos + jnp.array([0.0375, 0, 0])
        right_fingertip_goal_pos = cube_pos + jnp.array([-0.0375, 0, 0])
        gripper_openness_goal = jnp.array([0.075])
        goal = jnp.concatenate([left_fingertip_goal_pos] + [right_fingertip_goal_pos] + [gripper_openness_goal])
        return goal
    def _compute_goal_completion(self, obs, goal):
        cube_pos = obs[:3]
        left_fingertip_pos = obs[16:19]
        right_fingertip_pos = obs[19:22]
        fingertip_midpoint = (left_fingertip_pos + right_fingertip_pos) / 2
        cube_to_fingertip_midpoint_dist = jnp.linalg.norm(cube_pos - fingertip_midpoint)
        gripper_openness = obs[22]
        goal_gripper_openness = goal[9]
        gripper_openness_difference = jnp.linalg.norm(gripper_openness - goal_gripper_openness)
        success = jnp.array(jnp.all(jnp.array([cube_to_fingertip_midpoint_dist < 0.05, gripper_openness_difference < 0.02])), dtype=float)
        success_easy = jnp.array(jnp.all(jnp.array([cube_to_fingertip_midpoint_dist < 0.15, gripper_openness_difference < 0.05])), dtype=float)
        success_hard = jnp.array(jnp.all(jnp.array([cube_to_fingertip_midpoint_dist < 0.02, gripper_openness_difference < 0.005])), dtype=float)
        return success, success_easy, success_hard
    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[7:10].set(goal[:3])
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        q_indices = jnp.array([0, 1, 2, 14, 15, 16, 17, 18, 19, 20])
        q_subset = pipeline_state.q[q_indices]
        eef_index = 8
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        left_fingertip_index = 10
        left_fingertip_x_pos = pipeline_state.x.pos[left_fingertip_index]
        right_fingertip_index = 12
        right_fingertip_x_pos = pipeline_state.x.pos[right_fingertip_index]
        fingertip_distance = jnp.linalg.norm(right_fingertip_x_pos - left_fingertip_x_pos)[None]
        return jnp.concatenate(
            [q_subset] + [eef_x_pos] + [eef_xd_vel] + [left_fingertip_x_pos] + [right_fingertip_x_pos] + [fingertip_distance] + [goal]
        )
    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(14, 21)
        return pipeline_state.q[q_indices]

class ArmPushEasy(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_push_easy.xml"
    @property
    def action_size(self) -> int:
        return 5
    def _set_environment_attributes(self):
        self.env_name = "arm_push_easy"
        self.episode_length = 50
        self.goal_indices = jnp.array([0, 1, 2])
        self.completion_goal_indices = jnp.array([0, 1, 2])
        self.state_dim = 17
        self.arm_noise_scale = 0
        self.cube_noise_scale = 0.1
        self.goal_noise_scale = 0.1
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]
        arm_q_default = jnp.array([1.571, 0.742, 0, -1.571, 0, 3.054, 1.449, 0.04, 0.04])
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 14], minval=-1)
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [target_q] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array([0.1, 0.6, 0.03]) + jnp.array([self.goal_noise_scale, self.goal_noise_scale, 0]) * jax.random.uniform(
            subkey, [3], minval=-1
        )
        return cube_goal_pos
    def _compute_goal_completion(self, obs, goal):
        current_cube_pos = obs[self.completion_goal_indices]
        goal_pos = goal[:3]
        dist = jnp.linalg.norm(current_cube_pos - goal_pos)
        success = jnp.array(dist < 0.1, dtype=float)
        success_easy = jnp.array(dist < 0.3, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard
    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[7:10].set(goal[:3])
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        q_indices = jnp.array([0, 1, 2, 14, 15, 16, 17, 18, 19, 20])
        q_subset = pipeline_state.q[q_indices]
        eef_index = 8
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        left_finger_index = 9
        left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
        right_finger_index = 10
        right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
        finger_distance = jnp.linalg.norm(right_finger_x_pos - left_finger_x_pos)[None]
        return jnp.concatenate([q_subset] + [eef_x_pos] + [eef_xd_vel] + [finger_distance] + [goal])
    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(14, 21)
        return pipeline_state.q[q_indices]

class ArmPushHard(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_push_hard.xml"
    @property
    def action_size(self) -> int:
        return 5
    def _set_environment_attributes(self):
        self.env_name = "arm_push_hard"
        self.episode_length = 80
        self.goal_indices = jnp.array([0, 1, 2])
        self.completion_goal_indices = jnp.array([0, 1, 2])
        self.state_dim = 17
        self.arm_noise_scale = 0
        self.cube_noise_scale = 0.25
        self.goal_noise_scale = 0.25
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]
        arm_q_default = jnp.array([1.571, 0.742, 0, -1.571, 0, 3.054, 1.449, 0.04, 0.04])
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 14], minval=-1)
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [target_q] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array([0.25, 0.65, 0.03]) + jnp.array([self.goal_noise_scale, self.goal_noise_scale, 0]) * jax.random.uniform(
            subkey, [3], minval=-1
        )
        return cube_goal_pos
    def _compute_goal_completion(self, obs, goal):
        current_cube_pos = obs[self.completion_goal_indices]
        goal_pos = goal[:3]
        dist = jnp.linalg.norm(current_cube_pos - goal_pos)
        success = jnp.array(dist < 0.1, dtype=float)
        success_easy = jnp.array(dist < 0.3, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard
    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[7:10].set(goal[:3])
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        q_indices = jnp.array([0, 1, 2, 14, 15, 16, 17, 18, 19, 20])
        q_subset = pipeline_state.q[q_indices]
        eef_index = 8
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        left_finger_index = 9
        left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
        right_finger_index = 10
        right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
        finger_distance = jnp.linalg.norm(right_finger_x_pos - left_finger_x_pos)[None]
        return jnp.concatenate([q_subset] + [eef_x_pos] + [eef_xd_vel] + [finger_distance] + [goal])
    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(14, 21)
        return pipeline_state.q[q_indices]

# ============================================================================
# Replay buffer, evaluator, args, networks (inlined from buffer.py / evaluator.py)
# ============================================================================

@dataclass
class Args:
    exp_name: str = "train"
    seed: int = 1000
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "clean_JaxGCRL_test"
    wandb_entity: str = "kevinbuhler"
    wandb_mode: str = "online"
    wandb_dir: str = "."
    wandb_group: str = "."
    capture_vis: bool = True
    vis_length: int = 1000
    checkpoint: bool = True
    # environment specific arguments
    env_id: str = "humanoid"  # "ant_big_maze" "humanoid_u_maze" "arm_binpick_hard"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0
    # Algorithm specific arguments
    total_env_steps: int = 100000000  # 50000000
    num_epochs: int = 100  # 50
    num_envs: int = 512
    eval_env_id: str = ""
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4
    actor_skip_connections: int = 0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every N layers)
    critic_skip_connections: int = 0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every N layers)
    num_episodes_per_env: int = 1  # recommended to keep at 1
    training_steps_multiplier: int = 1  # recommended to keep at 1
    use_all_batches: int = 0  # recommended to keep at 0
    num_sgd_batches_per_training_step: int = 800
    eval_actor: int = 0  # recommended to keep at 0
    # if 0, use deterministic actor for evaluation
    # if 1, use stochastic actor for evaluation
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    expl_actor: int = 1  # recommended to keep at 1
    # if 0, use deterministic actor for exploration/collecting data
    # if 1, use stochastic actor for exploration/collecting data
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    entropy_param: float = 0.5
    disable_entropy: int = 0
    use_relu: int = 0
    num_render: int = 10
    save_buffer: int = 0
    # to be filled in runtime
    env_steps_per_actor_step: int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps: int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps: int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch: int = 0
    """the number of training steps per epoch(computed in runtime)"""

lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros

def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x

class SA_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish
        x = jnp.concatenate([s, a], axis=-1)
        # Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        # Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        # Final layer
        x = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x

class G_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    @nn.compact
    def __call__(self, g: jnp.ndarray):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish
        x = g
        # Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        # Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        # Final layer
        x = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x

class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        # Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        # Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        # Final layer
        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.tanh(log_std)
        # From SpinUp / Denis Yarats.
        # rescales tanh output (−1..1) into the range [LOG_STD_MIN, LOG_STD_MAX] = [−5, 2].
        # preffered over clipping since it keeps gradients everywhere.
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState

class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()

# ==============================================================================
# Replay buffer (inlined from buffer.py)
# ==============================================================================

@flax.struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""
    data: jnp.ndarray
    insert_position: jnp.ndarray
    sample_position: jnp.ndarray
    key: PRNGKey

class TrajectoryUniformSamplingQueue:
    """
    Base class for limited-size FIFO reply buffers.

    Implements an `insert()` method which behaves like a limited-size queue.
    I.e. it adds samples to the end of the queue and, if necessary, removes the
    oldest samples form the queue in order to keep the maximum size within the
    specified limit.

    Derived classes must implement the `sample()` method.
    """
    def __init__(self, max_replay_size: int, dummy_data_sample, sample_batch_size: int, num_envs: int, episode_length: int):
        self._flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))
        data_size = len(dummy_flatten)
        print(f"data_size: {data_size}", flush=True)
        self._data_shape = (max_replay_size, num_envs, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self._size = 0
        self.num_envs = num_envs
        self.episode_length = episode_length
    def init(self, key):
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )
    def insert(self, buffer_state, samples):
        """Insert data into the replay buffer."""
        self.check_can_insert(buffer_state, samples, 1)
        return self.insert_internal(buffer_state, samples)
    def check_can_insert(self, buffer_state, samples, shards):
        """Checks whether insert operation can be performed."""
        assert isinstance(shards, int), "This method should not be JITed."
        insert_size = jax.tree_util.tree_flatten(samples)[0][0].shape[0] // shards
        if self._data_shape[0] < insert_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay"
                f" size. num_samples: {insert_size}, max replay size"
                f" {self._data_shape[0]}"
            )
        self._size = min(self._data_shape[0], self._size + insert_size)
    def check_can_sample(self, buffer_state, shards):
        """Checks whether sampling can be performed. Do not JIT this method."""
        pass
    def insert_internal(self, buffer_state, samples):
        """Insert data in the replay buffer.

        Args:
          buffer_state: Buffer state
          samples: Sample to insert with a leading batch size.

        Returns:
          New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(f"buffer_state.data.shape ({buffer_state.data.shape}) " f"doesn't match the expected value ({self._data_shape})")
        update = self._flatten_fn(samples)  # Updates has shape (unroll_len, num_envs, self._data_shape[-1])
        data = buffer_state.data  # shape = (max_replay_size, num_envs, data_size)
        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll
        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (
            len(data) + 1
        )  # so whenever roll happens, position becomes len(data), else it is increased by len(update), what is the use of doing % (len(data) + 1)??
        sample_position = jnp.maximum(
            0, buffer_state.sample_position + roll
        )  # what is the use of this line? sample_position always remains 0 as roll can never be positive
        return buffer_state.replace(data=data, insert_position=position, sample_position=sample_position)
    def sample(self, buffer_state):
        """Sample a batch of data."""
        self.check_can_sample(buffer_state, 1)
        return self.sample_internal(buffer_state)
    def sample_internal(self, buffer_state):
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # Note: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs
        # Sampling envs idxs
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)
        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix
        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")
        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))
        matrix = create_matrix(
            shape, self.episode_length, buffer_state.sample_position, buffer_state.insert_position - self.episode_length, sample_key
        )
        """
        The function create_batch will be called for every envs_idxs of buffer_state.data and every row of matrix.
        Because every row of matrix has consecutive indices of self.episode_length, for every
        envs_idx of envs_idxs, we will sample a random self.episode_length length sequence from
        buffer_state.data[:, envs_idx, :]. But I don't think the code ensures that this sequence
        won't be across episodes?

        flatten_crl_fn takes care of this
        """
        print(f"buffer_state.data[:, envs_idxs, :].shape: {buffer_state.data[:, envs_idxs, :].shape}", flush=True)
        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        return buffer_state.replace(key=key), transitions
    @staticmethod
    @functools.partial(jax.jit, static_argnames=("buffer_config"))
    def flatten_crl_fn(buffer_config, transition, sample_key):
        gamma, obs_dim, goal_start_idx, goal_end_idx = buffer_config
        # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(
            arrangement[:, None] < arrangement[None], dtype=jnp.float32
        )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
        discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
        probs = is_future_mask * discount
        # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
        #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
        #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
        #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
        #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
        #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
        # assuming seq_len = 5
        # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))
        single_trajectories = jnp.concatenate([transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0)
        # array of seq_len x seq_len where a row is an array of seeds that correspond to the episode index from which that time-step was collected
        # timesteps collected from the same episode will have the same seed. All rows of the single_trajectories are same.
        probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        # ith row of probs will be non zero only for time indices that
        # 1) are greater than i
        # 2) have the same seed as the ith time index
        goal_index = jax.random.categorical(sample_key, jnp.log(probs))
        future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)  # the last goal_index cannot be considered as there is no future.
        future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
        goal = future_state[:, goal_start_idx:goal_end_idx]
        future_state = future_state[:, :obs_dim]
        state = transition.observation[:-1, :obs_dim]  # all states are considered
        new_obs = jnp.concatenate([state, goal], axis=1)
        # BASICALLY HERE, for each state in the 1000 time-steps, we are creating a new observation by
        # appending the goal to the state (where the goal is extracted from the future state, which
        # is sampled with geometric of gamma of the same trajectory)
        extras = {
            "policy_extras": {},
            "state_extras": {
                "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
                "seed": jnp.squeeze(transition.extras["state_extras"]["seed"][:-1]),
            },
            "state": state,
            "future_state": future_state,
            "future_action": future_action,
        }
        return transition._replace(
            observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
            action=jnp.squeeze(transition.action[:-1]),
            reward=jnp.squeeze(transition.reward[:-1]),
            discount=jnp.squeeze(transition.discount[:-1]),
            extras=extras,
        )
    def size(self, buffer_state: ReplayBufferState) -> int:
        return buffer_state.insert_position - buffer_state.sample_position

# ==============================================================================
# Evaluator (inlined from evaluator.py)
# ==============================================================================

def generate_unroll(actor_step, training_state, env, env_state, unroll_length, extra_fields=()):
    """Collect trajectories of given unroll_length."""
    @jax.jit
    def f(carry, unused_t):
        state = carry
        nstate, transition = actor_step(training_state, env, state, extra_fields=extra_fields)
        return nstate, transition
    final_state, data = jax.lax.scan(f, env_state, (), length=unroll_length)
    return final_state, data

class CrlEvaluator:
    def __init__(self, actor_step, eval_env, num_eval_envs, episode_length, key):
        self._key = key
        self._eval_walltime = 0.0
        eval_env = envs.training.EvalWrapper(eval_env)
        def generate_eval_unroll(training_state, key):
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(actor_step, training_state, eval_env, eval_first_state, unroll_length=episode_length)[0]
        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs
    def run_evaluation(self, training_state, training_metrics, aggregate_episodes=True):
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)
        t = time.time()
        eval_state = self._generate_eval_unroll(training_state, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            # (np.std, "_std"),
            # (np.max, "_max"),
            # (np.min, "_min"),
        ]
        print("Available keys in episode_metrics:", eval_metrics.episode_metrics.keys())
        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(eval_metrics.episode_metrics[name]) if aggregate_episodes else eval_metrics.episode_metrics[name]
                    )
                    for name in ["reward", "success", "success_easy", "success_hard", "dist", "distance_from_origin"]
                    if name in eval_metrics.episode_metrics  # THIS WAS ADDED BY ME (for arm tasks, may not be)
                }
            )
        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(eval_metrics.episode_metrics["success"] > 0.0)
        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}
        return metrics

def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)

def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))

if __name__ == "__main__":
    args = tyro.cli(Args)
    # Print every arg
    print("Arguments:", flush=True)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}", flush=True)
    print("\n", flush=True)
    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    print(f"env_steps_per_actor_step: {args.env_steps_per_actor_step}", flush=True)
    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    print(f"num_prefill_env_steps: {args.num_prefill_env_steps}", flush=True)
    args.num_prefill_actor_steps = np.ceil(args.min_replay_size / args.unroll_length)
    print(f"num_prefill_actor_steps: {args.num_prefill_actor_steps}", flush=True)
    args.num_training_steps_per_epoch = (args.total_env_steps - args.num_prefill_env_steps) // (args.num_epochs * args.env_steps_per_actor_step)
    print(f"num_training_steps_per_epoch: {args.num_training_steps_per_epoch}", flush=True)
    run_name = f"{args.env_id}{'_' + args.eval_env_id if args.eval_env_id else ''}_{args.batch_size}_{args.total_env_steps}_nenvs:{args.num_envs}_criticwidth:{args.critic_network_width}_actorwidth:{args.actor_network_width}_criticdepth:{args.critic_depth}_actordepth:{args.actor_depth}_actorskip:{args.actor_skip_connections}_criticskip:{args.critic_skip_connections}_{args.seed}"
    print(f"run_name: {run_name}", flush=True)
    if args.track:
        if args.wandb_group == ".":
            args.wandb_group = None
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        if args.wandb_mode == "offline":
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()
    if args.checkpoint:
        from pathlib import Path
        from datetime import datetime
        short_run_name = f"runs/{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_path = Path(args.wandb_dir) / Path(short_run_name)
        os.mkdir(path=save_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)
    def make_env(env_id=args.env_id):
        # Dispatches to the env classes inlined at the top of this file. Assets are
        # downloaded on first use; no local envs/ package or asset files are needed.
        print(f"making env with env_id: {env_id}", flush=True)
        if env_id == "reacher":
            env = Reacher(backend="spring")
            args.obs_dim = 10
            args.goal_start_idx = 4
            args.goal_end_idx = 7
        elif env_id == "pusher":
            env = Pusher(backend="spring")
            args.obs_dim = 20
            args.goal_start_idx = 10
            args.goal_end_idx = 13
        elif env_id == "ant":
            env = Ant(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
            args.obs_dim = 29
            args.goal_start_idx = 0
            args.goal_end_idx = 2
        elif "ant" in env_id and "maze" in env_id:  # needed the add the ant check to differentiate with humanoid maze
            if "gen" not in env_id:
                env = AntMaze(
                    backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True, maze_layout_name=env_id[4:]
                )
                args.obs_dim = 29
                args.goal_start_idx = 0
                args.goal_end_idx = 2
            else:
                gen_idx = env_id.find("gen")
                maze_layout_name = env_id[4 : gen_idx - 1]
                generalization_config = env_id[gen_idx + 4 :]
                print(f"maze_layout_name: {maze_layout_name}, generalization_config: {generalization_config}", flush=True)
                env = AntMazeGeneralization(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=True,
                    maze_layout_name=maze_layout_name,
                    generalization_config=generalization_config,
                )
                args.obs_dim = 29
                args.goal_start_idx = 0
                args.goal_end_idx = 2
        elif env_id == "ant_ball":
            env = AntBall(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
            args.obs_dim = 31
            args.goal_start_idx = 28
            args.goal_end_idx = 30
        elif env_id == "ant_push":
            env = AntPush(backend="mjx")
            args.obs_dim = 31
            args.goal_start_idx = 0
            args.goal_end_idx = 2
        elif env_id == "humanoid":
            env = Humanoid(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
            args.obs_dim = 268
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        elif "humanoid" in env_id and "maze" in env_id:
            env = HumanoidMaze(backend="spring", maze_layout_name=env_id[9:])
            args.obs_dim = 268
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        elif env_id == "arm_reach":
            env = ArmReach(backend="mjx")
            args.obs_dim = 13
            args.goal_start_idx = 7
            args.goal_end_idx = 10
        elif env_id == "arm_binpick_easy":
            env = ArmBinpickEasy(backend="mjx")
            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        elif env_id == "arm_binpick_hard":
            env = ArmBinpickHard(backend="mjx")
            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        elif env_id == "arm_binpick_easy_EEF":
            env = ArmBinpickEasyEEF(backend="mjx")
            args.obs_dim = 11
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        elif "arm_grasp" in env_id:  # either arm_grasp or arm_grasp_0.5, etc
            cube_noise_scale = float(env_id[10:]) if len(env_id) > 9 else 0.3
            env = ArmGrasp(cube_noise_scale=cube_noise_scale, backend="mjx")
            args.obs_dim = 23
            args.goal_start_idx = 16
            args.goal_end_idx = 23
        elif env_id == "arm_push_easy":
            env = ArmPushEasy(backend="mjx")
            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        elif env_id == "arm_push_hard":
            env = ArmPushHard(backend="mjx")
            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        else:
            raise NotImplementedError
        return env
    env = make_env()
    env = envs.training.wrap(env, episode_length=args.episode_length)
    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)
    print(f"obs_size: {obs_size}, action_size: {action_size}", flush=True)
    if not args.eval_env_id:
        args.eval_env_id = args.env_id
    # make eval env
    eval_env = make_env(args.eval_env_id)
    eval_env = envs.training.wrap(eval_env, episode_length=args.episode_length)
    eval_env_keys = jax.random.split(eval_env_key, args.num_envs)
    eval_env_state = jax.jit(eval_env.reset)(eval_env_keys)
    eval_env.step = jax.jit(eval_env.step)
    # Network setup
    # Actor
    actor = Actor(
        action_size=action_size,
        network_width=args.actor_network_width,
        network_depth=args.actor_depth,
        skip_connections=args.actor_skip_connections,
        use_relu=args.use_relu,
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply, params=actor.init(actor_key, np.ones([1, obs_size])), tx=optax.adam(learning_rate=args.actor_lr)
    )
    # Critic
    sa_encoder = SA_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
    )
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
    g_encoder = G_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
    )
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    critic_state = TrainState.create(
        apply_fn=None, params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params}, tx=optax.adam(learning_rate=args.critic_lr)
    )
    # Entropy coefficient
    target_entropy = -args.entropy_param * action_size  # action_size = 8 for ant, 17 for humanoid, etc
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(apply_fn=None, params={"log_alpha": log_alpha}, tx=optax.adam(learning_rate=args.alpha_lr))
    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()), gradient_steps=jnp.zeros(()), actor_state=actor_state, critic_state=critic_state, alpha_state=alpha_state
    )
    # Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(
        observation=dummy_obs, action=dummy_action, reward=0.0, discount=0.0, extras={"state_extras": {"truncation": 0.0, "seed": 0.0}}
    )
    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer
    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=args.batch_size,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)
    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh(means)
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs, action=actions, reward=nstate.reward, discount=1 - nstate.done, extras={"state_extras": state_extras}
        )
    def actor_step(training_state, env, env_state, key, extra_fields):
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs, action=actions, reward=nstate.reward, discount=1 - nstate.done, extras={"state_extras": state_extras}
        )
    def multi_sample_actor_step(training_state, env, env_state, key, K, extra_fields):
        # Get K sets of actions from the actor
        keys = jax.random.split(key, K)
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = jnp.stack([nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype)) for k in keys])
        state = env_state.obs[:, : args.obs_dim]
        goal = env_state.obs[:, args.obs_dim :]
        sa_reprs = jax.vmap(lambda a: sa_encoder.apply(training_state.critic_state.params["sa_encoder"], state, a))(actions)
        g_repr = g_encoder.apply(training_state.critic_state.params["g_encoder"], goal)
        q_values = -jnp.sqrt(jnp.sum((sa_reprs - g_repr) ** 2, axis=-1))
        best_action_idx = jnp.argmax(q_values, axis=0)
        best_actions = jnp.take_along_axis(actions, best_action_idx[None, :, None], axis=0)[0]
        # Step environment with best actions
        nstate = env.step(env_state, best_actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs, action=best_actions, reward=nstate.reward, discount=1 - nstate.done, extras={"state_extras": state_extras}
        )
    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):  # conducts a single actor step in environment
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            if args.expl_actor == 1:
                env_state, transition = actor_step(training_state, env, env_state, current_key, extra_fields=("truncation", "seed"))
            elif args.expl_actor == 0:
                env_state, transition = deterministic_actor_step(training_state, env, env_state, extra_fields=("truncation", "seed"))
            else:
                env_state, transition = multi_sample_actor_step(
                    training_state, env, env_state, current_key, args.expl_actor, extra_fields=("truncation", "seed")
                )
            return (env_state, next_key), transition
        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state
    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(training_state, env_state, buffer_state, key)
            training_state = training_state.replace(env_steps=training_state.env_steps + args.env_steps_per_actor_step)
            return (training_state, env_state, buffer_state, new_key), ()
        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_prefill_actor_steps)[0]
    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        actor_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:actor_batch_size], transitions)
        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            obs = transitions.observation  # expected_shape = batch_size, obs_size + goal_size
            state = obs[:, : args.obs_dim]
            future_state = transitions.extras["future_state"]
            goal = future_state[:, args.goal_start_idx : args.goal_end_idx]
            observation = jnp.concatenate([state, goal], axis=1)
            means, log_stds = actor.apply(actor_params, observation)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)  # dimension = B
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
            g_repr = g_encoder.apply(g_encoder_params, goal)
            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
            if args.disable_entropy:
                actor_loss = -jnp.mean(qf_pi)
            else:
                actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - (qf_pi))
            return actor_loss, log_prob
        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
            return jnp.mean(alpha_loss)
        (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params, training_state.critic_state.params, training_state.alpha_state.params["log_alpha"], transitions, key
        )
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)
        alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)
        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)
        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "alph_aloss": alphaloss,
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }
        return training_state, metrics
    @jax.jit
    def update_critic(transitions, training_state, key):
        critic_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:critic_batch_size], transitions)
        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            obs = transitions.observation[:, : args.obs_dim]
            action = transitions.action
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
            g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, args.obs_dim :])
            # InfoNCE
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))  # shape = BxB
            critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
            # logsumexp regularisation
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)
            I, correct, logits_pos, logits_neg = jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)
            return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)
        (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(critic_loss, has_aux=True)(
            training_state.critic_state.params, transitions, key
        )
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)
        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
        }
        return training_state, metrics
    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)
        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)
        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)
        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        return (training_state, key), metrics
    @jax.jit
    def training_step(training_state, env_state, buffer_state, key, t):
        experience_key1, experience_key2, sampling_key, training_key, sgd_batches_key = jax.random.split(key, 5)
        # update buffer
        env_state, buffer_state = get_experience(training_state, env_state, buffer_state, experience_key1)
        training_state = training_state.replace(env_steps=training_state.env_steps + args.env_steps_per_actor_step)
        transitions_list = []
        for _ in range(args.num_episodes_per_env):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)
        # Concatenate all sampled transitions
        transitions = jax.tree_util.tree_map(lambda *arrays: jnp.concatenate(arrays, axis=0), *transitions_list)
        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        )
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions)
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        # I added this code, so as to ensure len(transitions.observation) is divisible by batch_size
        num_full_batches = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[: num_full_batches * args.batch_size], transitions)
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]), transitions)
        if args.use_all_batches == 0:
            num_total_batches = transitions.observation.shape[0]
            selected_indices = jax.random.permutation(sgd_batches_key, num_total_batches)[: args.num_sgd_batches_per_training_step]
            transitions = jax.tree_util.tree_map(lambda x: x[selected_indices], transitions)
        # take actor-step worth of training-step
        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)
        return (training_state, env_state, buffer_state), metrics
    @jax.jit
    def training_epoch(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs), metrics = training_step(ts, es, bs, train_key, t)
            return (ts, es, bs, k), metrics
        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f, (training_state, env_state, buffer_state, key), jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier)
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(training_state, env_state, buffer_state, prefill_key)
    if args.eval_actor == 0:
        """Setting up evaluator"""
        evaluator = CrlEvaluator(
            deterministic_actor_step, eval_env, num_eval_envs=args.num_eval_envs, episode_length=args.episode_length, key=eval_env_key
        )
    elif args.eval_actor == 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, extra_fields: actor_step(training_state, env, env_state, eval_actor_key, extra_fields),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    elif args.eval_actor > 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            # Replace deterministic_actor_step with a partial function of multi_sample_actor_step
            lambda training_state, env, env_state, extra_fields: multi_sample_actor_step(
                training_state, env, env_state, eval_actor_key, args.eval_actor, extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    training_walltime = 0
    print("starting training....", flush=True)
    start_time = time.time()
    for ne in range(args.num_epochs):
        t = time.time()
        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, epoch_key)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        metrics = evaluator.run_evaluation(training_state, metrics)
        print(f"epoch {ne} out of {args.num_epochs} complete. metrics: {metrics}", flush=True)
        if args.checkpoint:
            if ne < 5 or ne >= args.num_epochs - 5 or ne % 10 == 0:
                # Save current policy and critic params.
                params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
                path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
        if args.track:
            wandb.log(metrics, step=ne)
            if args.wandb_mode == "offline":
                trigger_sync()
        hours_passed = (time.time() - start_time) / 3600
        print(f"Time elapsed: {hours_passed:.3f} hours", flush=True)
    if args.checkpoint:
        # Save current policy and critic params.
        params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
        path = f"{save_path}/final.pkl"
        save_params(path, params)
    # After training is complete, render the final policy
    if args.capture_vis:
        def render_policy(training_state, save_path):
            """Renders the policy and saves it as an HTML file."""
            @jax.jit
            def policy_step(env_state, actor_params):
                means, _ = actor.apply(actor_params, env_state.obs)
                actions = nn.tanh(means)
                next_state = env.step(env_state, actions)
                return next_state, env_state
            rollout_states = []
            for i in range(args.num_render):
                env = make_env(args.eval_env_id)
                rng = jax.random.PRNGKey(seed=i + 1)
                env_state = jax.jit(env.reset)(rng)
                for _ in range(args.vis_length):
                    env_state, current_state = policy_step(env_state, training_state.actor_state.params)
                    rollout_states.append(current_state.pipeline_state)
            # Render and save
            html_string = html.render(env.sys, rollout_states)
            render_path = f"{save_path}/vis.html"
            with open(render_path, "w") as f:
                f.write(html_string)
            wandb.log({"vis": wandb.Html(html_string)})
        print("Rendering final policy...", flush=True)
        try:
            render_policy(training_state, save_path)
        except Exception as e:
            print(f"Error rendering final policy: {e}", flush=True)
    # After training is complete, save the Args
    if args.checkpoint:
        with open(f"{save_path}/args.pkl", "wb") as f:
            pickle.dump(args, f)
        print(f"Saved args to {save_path}/args.pkl", flush=True)
    # After training is complete, save the replay buffer (if save_buffer is 1, this takes a lot of memory)
    if args.checkpoint:
        if args.save_buffer:
            print("Saving final buffer_state and buffer data (everything needed to recreate replay_buffer)...", flush=True)
            try:
                buffer_path = f"{save_path}/final_buffer.pkl"
                buffer_data = {
                    "buffer_state": buffer_state,
                    "max_replay_size": args.max_replay_size,
                    "batch_size": args.batch_size,
                    "num_envs": args.num_envs,
                    "episode_length": args.episode_length,
                }
                with open(buffer_path, "wb") as f:
                    pickle.dump(buffer_data, f)
                print(f"Saved replay_buffer to {buffer_path}", flush=True)
            except Exception as e:
                print(f"Error saving final replay buffer: {e}", flush=True)
