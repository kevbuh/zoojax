# https://github.com/hujie-frank/SENet
# Squeeze-and-Excitation Networks (SENet) — single-file JAX/Flax port.
# Paper: https://arxiv.org/abs/1709.01507 (Hu, Shen, Sun; CVPR 2018)
# Ported from the original Caffe reference in misc/SENet (SE-ResNet-50/101/152).
#
# The port is faithful to the Caffe prototxts:
#   - stem: 7x7 s2 conv (64) -> BN -> ReLU -> 3x3 s2 maxpool
#   - bottleneck: 1x1 reduce -> 3x3 -> 1x1 increase, each conv followed by BN
#     (Caffe BatchNorm + Scale), with spatial stride placed on the 1x1 *reduce*
#     conv (ResNet-v1 placement, matching conv{3,4,5}_1_1x1_reduce stride: 2).
#   - SE block: global avg pool (squeeze) -> 1x1 down (C/r) -> ReLU -> 1x1 up (C)
#     -> sigmoid (excitation) -> channel-wise rescale. r = 16 for all models.
#   - the Caffe "Axpy" layer fuses (scale * branch + shortcut); here it is the
#     plain `se(y) + residual` followed by ReLU.
#
# Run the embedded tests:  python src/squeeze_excitation.py --test
#                     or:  pytest src/squeeze_excitation.py

import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence

import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

# MSRA / He initialization for the conv weights, as used in the paper
# ("we initialize the weights as in [He et al.]"). Caffe's "msra" filler is a
# fan-in, ReLU-gain (2.0) Gaussian.
MSRA = nn.initializers.variance_scaling(2.0, "fan_in", "normal")


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: recalibrate channels via a global descriptor.

    Squeeze with global average pooling, then a bottleneck MLP (down -> ReLU ->
    up -> sigmoid) produces a per-channel gate in (0, 1) that rescales the input.
    The two Dense layers are equivalent to the 1x1 convs on a 1x1 feature map in
    the Caffe reference (both carry a bias term).
    """

    channels: int
    reduction: int = 16

    @nn.compact
    def __call__(self, x):
        # x: (N, H, W, C). Squeeze spatial dims -> (N, C).
        s = jnp.mean(x, axis=(1, 2))
        s = nn.Dense(max(1, self.channels // self.reduction), name="fc_down")(s)
        s = nn.relu(s)
        s = nn.Dense(self.channels, name="fc_up")(s)
        s = nn.sigmoid(s)
        # Excitation: broadcast the gate over the spatial dims and rescale.
        return x * s[:, None, None, :]


class SEBottleneck(nn.Module):
    """SE-ResNet bottleneck block (Caffe conv{stage}_{idx})."""

    width: int          # channels of the 1x1-reduce / 3x3 convs
    out_channels: int   # channels of the 1x1-increase conv (4 * width)
    stride: int = 1
    reduction: int = 16
    proj: bool = False  # use a 1x1 conv shortcut (first block of every stage)

    @nn.compact
    def __call__(self, x, train: bool):
        norm = partial(
            nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5
        )
        conv = partial(nn.Conv, use_bias=False, kernel_init=MSRA)

        # Stride lives on the 1x1 reduce conv, matching the Caffe prototxt.
        y = conv(self.width, (1, 1), strides=(self.stride, self.stride),
                 name="conv_reduce")(x)
        y = norm(name="bn_reduce")(y)
        y = nn.relu(y)

        y = conv(self.width, (3, 3), padding=((1, 1), (1, 1)), name="conv_3x3")(y)
        y = norm(name="bn_3x3")(y)
        y = nn.relu(y)

        y = conv(self.out_channels, (1, 1), name="conv_increase")(y)
        y = norm(name="bn_increase")(y)

        y = SEBlock(self.out_channels, self.reduction, name="se")(y)

        residual = x
        if self.proj:
            residual = conv(self.out_channels, (1, 1),
                            strides=(self.stride, self.stride), name="conv_proj")(x)
            residual = norm(name="bn_proj")(residual)

        # Axpy + ReLU.
        return nn.relu(y + residual)


class SEResNet(nn.Module):
    """SE-ResNet backbone + classifier (NHWC inputs)."""

    stage_blocks: Sequence[int] = (3, 4, 6, 3)  # SE-ResNet-50
    num_classes: int = 1000
    reduction: int = 16
    stage_widths: Sequence[int] = (64, 128, 256, 512)

    @nn.compact
    def __call__(self, x, train: bool = False):
        norm = partial(
            nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5
        )

        # Stem.
        y = nn.Conv(64, (7, 7), strides=(2, 2), padding=((3, 3), (3, 3)),
                    use_bias=False, kernel_init=MSRA, name="stem_conv")(x)
        y = norm(name="stem_bn")(y)
        y = nn.relu(y)
        y = nn.max_pool(y, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Stages. The first block of each stage uses a projection shortcut;
        # stages 3-5 also downsample (stride 2) in their first block.
        for i, (nb, w) in enumerate(zip(self.stage_blocks, self.stage_widths)):
            out = w * 4
            for b in range(nb):
                stride = 2 if (b == 0 and i > 0) else 1
                y = SEBottleneck(
                    width=w, out_channels=out, stride=stride,
                    reduction=self.reduction, proj=(b == 0),
                    name=f"conv{i + 2}_{b + 1}",
                )(y, train)

        # Global average pool -> classifier.
        y = jnp.mean(y, axis=(1, 2))
        y = nn.Dense(self.num_classes, name="classifier")(y)
        return y


# Standard SE-ResNet depth configurations.
def se_resnet50(**kw):
    return SEResNet(stage_blocks=(3, 4, 6, 3), **kw)


def se_resnet101(**kw):
    return SEResNet(stage_blocks=(3, 4, 23, 3), **kw)


def se_resnet152(**kw):
    return SEResNet(stage_blocks=(3, 8, 36, 3), **kw)


ARCHS = {"se_resnet50": se_resnet50,
         "se_resnet101": se_resnet101,
         "se_resnet152": se_resnet152}


# --------------------------------------------------------------------------- #
# Faithful ImageNet training loop.                                            #
#                                                                             #
# The SENet repo ships only deploy prototxts (no solver.prototxt / training   #
# script), so the recipe below reproduces what the authors describe:          #
#   - README: minibatch 256, initial LR 0.1, 8x Titan X, "more epochs",       #
#     augmentation = random resized crop (8%-100% area, aspect 3/4-4/3) +     #
#     horizontal mirror + rotation +/-10 deg + pixel jitter +/-20, and BGR    #
#     mean subtraction (104, 117, 123) from the prototxt header comment.      #
#   - paper:  SGD with momentum 0.9, weight decay 1e-4, LR divided by 10      #
#     every 30 epochs, MSRA weight init (applied above).                      #
# Weight decay follows the standard ResNet convention of decaying only conv/  #
# fc kernels (not BN scale/shift or biases), implemented with an optax mask.  #
# --------------------------------------------------------------------------- #

IMAGENET_MEAN_BGR = (104.0, 117.0, 123.0)  # prototxt: "# mean_value: 104, 117, 123"


@dataclass
class Args:
    data_dir: str = ""          # ImageNet root (ImageFolder layout); empty -> synthetic smoke run
    arch: Literal["se_resnet50", "se_resnet101", "se_resnet152"] = "se_resnet50"
    num_classes: int = 1000
    epochs: int = 100           # README: "more epoches"; paper steps the LR every 30
    batch_size: int = 256       # README
    base_lr: float = 0.1        # README
    momentum: float = 0.9       # paper
    weight_decay: float = 1e-4  # paper
    lr_step_epochs: int = 30    # paper: divide LR by 10 every 30 epochs
    lr_gamma: float = 0.1       # paper
    crop_size: int = 224
    seed: int = 0
    smoke_steps: int = 6        # synthetic-only: gradient steps to run when data_dir == ""


class TrainState(train_state.TrainState):
    batch_stats: dict


def _decay_mask(params):
    """True for conv/fc kernels, False for BN scale/shift and biases."""
    return jax.tree_util.tree_map_with_path(
        lambda path, _: path[-1].key == "kernel", params
    )


def make_lr_schedule(base_lr, steps_per_epoch, step_epochs, gamma, total_epochs):
    """Step decay: multiply LR by `gamma` every `step_epochs` epochs."""
    boundaries = {}
    e = step_epochs
    while e < total_epochs:
        boundaries[e * steps_per_epoch] = gamma
        e += step_epochs
    return optax.piecewise_constant_schedule(base_lr, boundaries)


def make_optimizer(args, steps_per_epoch):
    schedule = make_lr_schedule(
        args.base_lr, steps_per_epoch, args.lr_step_epochs, args.lr_gamma, args.epochs
    )
    # Caffe SGD: g <- g + wd * w (decay folded into the gradient), then heavy-ball
    # momentum. optax.add_decayed_weights before optax.sgd matches that ordering.
    return optax.chain(
        optax.add_decayed_weights(args.weight_decay, mask=_decay_mask),
        optax.sgd(learning_rate=schedule, momentum=args.momentum, nesterov=False),
    )


def create_train_state(rng, model, args, steps_per_epoch):
    dummy = jnp.zeros((1, args.crop_size, args.crop_size, 3))
    variables = model.init(rng, dummy, train=False)
    tx = make_optimizer(args, steps_per_epoch)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=tx,
    )


@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            images, train=True, mutable=["batch_stats"],
        )
        # Softmax cross-entropy (Caffe SoftmaxWithLoss); weight decay is in the
        # optimizer, not the loss, matching Caffe's solver.
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, (logits, updates)

    (loss, (logits, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return state, loss, acc


@jax.jit
def eval_step(state, images, labels):
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        images, train=False,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, acc


def imagenet_transforms(crop_size, train):
    """torchvision transforms matching the README augmentation settings.

    RandomResizedCrop's torchvision defaults (scale 0.08-1.0, ratio 3/4-4/3)
    coincide exactly with the README's "Random Crop 8%~100%" and "Aspect Ratio
    3/4~4/3", so we use them directly; rotation, mirror and pixel jitter are the
    other documented augmentations. Returned tensors are 0-255 BGR with the
    ImageNet mean subtracted (Caffe convention), as a CHW float32 numpy array.
    """
    from torchvision import transforms  # lazy: only needed for real data

    mean_bgr = np.array(IMAGENET_MEAN_BGR, dtype=np.float32)

    def to_caffe(img):
        x = np.asarray(img, dtype=np.float32)            # HWC, RGB, 0-255
        x = x[:, :, ::-1]                                # RGB -> BGR
        if train:                                        # pixel jitter +/-20
            x = x + np.random.uniform(-20, 20, size=x.shape).astype(np.float32)
        x = x - mean_bgr
        return np.ascontiguousarray(x)                   # HWC BGR, mean-subtracted

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(crop_size),     # 8%-100% area, 3/4-4/3 aspect
            transforms.RandomRotation(10),               # +/-10 degrees
            transforms.RandomHorizontalFlip(),           # random mirror
            transforms.Lambda(to_caffe),
        ])
    return transforms.Compose([
        transforms.Resize(256),                          # shorter side = 256
        transforms.CenterCrop(crop_size),                # center 224 crop (eval protocol)
        transforms.Lambda(to_caffe),
    ])


def make_imagenet_loader(root, batch_size, crop_size, train):
    """ImageFolder-backed loader yielding (NHWC float32 BGR, int labels)."""
    import torch
    from torchvision import datasets

    ds = datasets.ImageFolder(root, transform=imagenet_transforms(crop_size, train))

    def collate(samples):
        imgs = np.stack([s[0] for s in samples])         # NHWC BGR float32
        labels = np.array([s[1] for s in samples], dtype=np.int32)
        return imgs, labels

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=4,
        drop_last=train, collate_fn=collate,
    )


def synthetic_batch(key, args):
    """A single fixed random batch (used when no data_dir is given)."""
    n = min(args.batch_size, 8)
    ik, lk = jax.random.split(key)
    images = jax.random.normal(ik, (n, args.crop_size, args.crop_size, 3))
    labels = jax.random.randint(lk, (n,), 0, args.num_classes)
    return images, labels


def train(args):
    print(f"# faithful SENet training: {args.arch}, batch={args.batch_size}, "
          f"lr={args.base_lr} (/{int(1/args.lr_gamma)} every {args.lr_step_epochs} ep), "
          f"mom={args.momentum}, wd={args.weight_decay}, epochs={args.epochs}")
    rng = jax.random.PRNGKey(args.seed)
    model = ARCHS[args.arch](num_classes=args.num_classes)

    if args.data_dir:
        train_loader = make_imagenet_loader(
            f"{args.data_dir}/train", args.batch_size, args.crop_size, train=True)
        steps_per_epoch = len(train_loader)
        rng, init_rng = jax.random.split(rng)
        state = create_train_state(init_rng, model, args, steps_per_epoch)
        n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
        print(f"# {n_params:,} parameters, {steps_per_epoch} steps/epoch")
        for epoch in range(args.epochs):
            t0 = time.time()
            for images, labels in train_loader:
                state, loss, acc = train_step(state, jnp.asarray(images), jnp.asarray(labels))
            print(f"epoch {epoch:3d}  loss {float(loss):.4f}  acc {float(acc):.3f}  "
                  f"({time.time() - t0:.0f}s)")
        return state

    # No data: overfit one synthetic batch to verify the loop is wired correctly.
    rng, init_rng, data_rng = jax.random.split(rng, 3)
    state = create_train_state(init_rng, model, args, steps_per_epoch=args.smoke_steps)
    images, labels = synthetic_batch(data_rng, args)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"# {n_params:,} parameters; no data_dir -> synthetic smoke run "
          f"({args.smoke_steps} steps, batch={images.shape[0]})")
    for step in range(args.smoke_steps):
        state, loss, acc = train_step(state, images, labels)
        print(f"step {step}  loss {float(loss):.4f}  acc {float(acc):.3f}")
    return state


# --------------------------------------------------------------------------- #
# Embedded tests (run: python src/squeeze_excitation.py --test  or  pytest).  #
# --------------------------------------------------------------------------- #

def _tiny(num_classes=10):
    # One block per stage keeps init/apply fast on CPU while exercising every
    # code path (stem, all four stages, projection + strided shortcuts, SE, head).
    return SEResNet(stage_blocks=(1, 1, 1, 1), num_classes=num_classes)


def _init(model, key, shape):
    x = jnp.zeros(shape)
    variables = model.init(key, x, train=False)
    return variables, x


def test_seblock_gate_in_unit_interval():
    """The SE excitation gate must lie in (0, 1) and scale, not shift, channels."""
    key = jax.random.PRNGKey(0)
    se = SEBlock(channels=8, reduction=4)
    x = jax.random.normal(key, (4, 5, 5, 8))
    params = se.init(key, x)
    y = se.apply(params, x)
    # Recover the per-channel gate from any spatial location where x != 0.
    gate = y[0, 0, 0] / x[0, 0, 0]
    assert jnp.all(gate > 0.0) and jnp.all(gate < 1.0)
    assert y.shape == x.shape


def test_forward_shape_and_finiteness():
    """A full forward pass returns class logits of the right shape."""
    key = jax.random.PRNGKey(0)
    model = _tiny(num_classes=10)
    variables, x = _init(model, key, (2, 32, 32, 3))
    logits = model.apply(variables, x, train=False)
    assert logits.shape == (2, 10)
    assert jnp.all(jnp.isfinite(logits))


def test_spatial_downsampling_factor():
    """Stem (/4) + three strided stages (/8) => total spatial reduction of 32x."""
    key = jax.random.PRNGKey(1)
    # For a 64x64 input the final feature map is 64/32 = 2x2 before global pool;
    # a successful apply over the strided stages confirms shape compatibility.
    model = _tiny(num_classes=5)
    variables, x = _init(model, key, (1, 64, 64, 3))
    logits = model.apply(variables, x, train=False)
    assert logits.shape == (1, 5)


def test_batchnorm_stats_update_in_train_mode():
    """train=True must mutate batch_stats; inference must not."""
    key = jax.random.PRNGKey(2)
    model = _tiny(num_classes=4)
    variables, x = _init(model, key, (2, 32, 32, 3))
    params, batch_stats = variables["params"], variables["batch_stats"]

    # Inference path: deterministic, batch_stats untouched.
    y1 = model.apply({"params": params, "batch_stats": batch_stats}, x, train=False)
    y2 = model.apply({"params": params, "batch_stats": batch_stats}, x, train=False)
    assert jnp.allclose(y1, y2)

    # Training path: returns mutated batch_stats.
    _, new_state = model.apply(
        {"params": params, "batch_stats": batch_stats},
        x, train=True, mutable=["batch_stats"],
    )
    leaves_old = jax.tree_util.tree_leaves(batch_stats)
    leaves_new = jax.tree_util.tree_leaves(new_state["batch_stats"])
    assert any(not jnp.allclose(a, b) for a, b in zip(leaves_old, leaves_new))


def test_depth_configs_block_counts():
    """The named factories produce the canonical SE-ResNet depths."""
    assert tuple(se_resnet50().stage_blocks) == (3, 4, 6, 3)
    assert tuple(se_resnet101().stage_blocks) == (3, 4, 23, 3)
    assert tuple(se_resnet152().stage_blocks) == (3, 8, 36, 3)


def test_lr_schedule_step_decay():
    """LR must start at base and drop by 10x at every 30-epoch boundary."""
    spe = 5
    sched = make_lr_schedule(0.1, steps_per_epoch=spe, step_epochs=30, gamma=0.1,
                             total_epochs=100)
    assert jnp.allclose(sched(0), 0.1)
    assert jnp.allclose(sched(30 * spe), 0.1 * 0.1)       # epoch 30
    assert jnp.allclose(sched(60 * spe), 0.1 * 0.1 * 0.1)  # epoch 60
    assert jnp.allclose(sched(90 * spe), 0.1 ** 4)         # epoch 90


def test_weight_decay_mask_selects_only_kernels():
    """Weight decay applies to conv/fc kernels, never BN params or biases."""
    key = jax.random.PRNGKey(0)
    model = _tiny(num_classes=4)
    variables = model.init(key, jnp.zeros((1, 32, 32, 3)), train=False)
    mask = _decay_mask(variables["params"])
    flat = jax.tree_util.tree_leaves_with_path(mask)
    decayed = [bool(v) for p, v in flat if p[-1].key == "kernel"]
    not_decayed = [bool(v) for p, v in flat if p[-1].key != "kernel"]
    assert all(decayed) and len(decayed) > 0      # every kernel is decayed
    assert not any(not_decayed)                   # nothing else is (scale/bias/etc.)


def test_train_step_overfits_one_batch():
    """A few faithful SGD steps must drive the loss down on a fixed batch."""
    args = Args(num_classes=5, crop_size=32, smoke_steps=20, base_lr=0.1)
    model = _tiny(num_classes=args.num_classes)
    rng = jax.random.PRNGKey(0)
    rng, init_rng, data_rng = jax.random.split(rng, 3)
    state = create_train_state(init_rng, model, args, steps_per_epoch=args.smoke_steps)
    images, labels = synthetic_batch(data_rng, args)
    first = None
    for _ in range(args.smoke_steps):
        state, loss, _ = train_step(state, images, labels)
        if first is None:
            first = float(loss)
    assert float(loss) < first        # loss decreased
    assert jnp.isfinite(loss)


def test_optimizer_excludes_bn_from_decay_in_practice():
    """With zero gradient, only masked (kernel) params shrink under weight decay."""
    args = Args(num_classes=4, crop_size=32, weight_decay=0.5, base_lr=1.0)
    model = _tiny(num_classes=args.num_classes)
    rng = jax.random.PRNGKey(1)
    state = create_train_state(rng, model, args, steps_per_epoch=1)
    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    new_state = state.apply_gradients(grads=zero_grads, batch_stats=state.batch_stats)
    flat_old = dict(jax.tree_util.tree_leaves_with_path(state.params))
    flat_new = dict(jax.tree_util.tree_leaves_with_path(new_state.params))
    for path, old in flat_old.items():
        new = flat_new[path]
        if path[-1].key == "kernel":
            assert not jnp.allclose(old, new) or jnp.allclose(old, 0.0)  # decayed
        else:
            assert jnp.allclose(old, new)                                # untouched


def _run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_tests()
    else:
        import tyro
        train(tyro.cli(Args))
