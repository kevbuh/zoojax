# https://arxiv.org/abs/1612.08810

from __future__ import annotations

import argparse
import datetime
import logging
import math
import os
import queue
import random
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig()
logger = logging.getLogger("predictron")
logger.setLevel(logging.INFO)

WEIGHT_DECAY = 0.0001
NORM_BN_DECAY = 0.997
NORM_BN_EPSILON = 1e-5
PREACT_BN_DECAY = 0.999
PREACT_BN_EPSILON = 1e-3
NORM_BN_MOMENTUM = 1.0 - NORM_BN_DECAY
PREACT_BN_MOMENTUM = 1.0 - PREACT_BN_DECAY

class Colour:
    NORMAL = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    @classmethod
    def highlight(cls, input_, success):
        colour = Colour.GREEN if success else Colour.RED
        return colour + input_ + Colour.NORMAL

class MazeGenerator:
    def __init__(self, height=20, width=None, density=0.3):
        if not width:
            width = height
        self.height = height
        self.width = width
        self.len = height * width
        # Create the right number of walls to be shuffled for each new maze
        non_corner_size = height * width - 2
        population_count = int(non_corner_size * density)
        empty_squares = non_corner_size - population_count
        self.walls = ["1"] * population_count + ["0"] * empty_squares
        # Starting point is the bottom right corner
        self.bottom_right_corner = int("0" * (self.len - 1) + "1", base=2)
        # Edges for use in flood search
        self.not_left_edge, self.not_right_edge, self.not_top_edge, self.not_bottom_edge = self._edges()
    def _edges(self):
        full_columns = "1" * (self.width - 1)
        not_left = int(("0" + full_columns) * self.height, base=2)
        not_right = int((full_columns + "0") * self.height, base=2)
        empty_row = "0" * self.width
        full_row = "1" * self.width
        full_rows = full_row * (self.height - 1)
        not_top = int(empty_row + full_rows, base=2)
        not_bottom = int(full_rows + empty_row, base=2)
        return not_left, not_right, not_top, not_bottom
    def maze_to_binary(self, maze):
        binary = bin(maze)[2:]
        return "0" * (self.len - len(binary)) + binary
    def print_maze(self, maze, labels):
        rows = []
        for i in range(self.height):
            row = maze[i]
            row = "".join([str(row[i][0]) for i in range(self.width)])
            row = row.replace("0", ".").replace("1", "#")
            row = row[:i] + Colour.highlight(row[i], labels[i]) + row[i + 1 :]
            rows.append(row)
        print("\n".join(rows))
    def generate(self):
        random.shuffle(self.walls)
        return int("0" + "".join(self.walls) + "0", base=2)
    def connected_squares(self, maze, start=None):
        empty_squares = ~maze
        current = None
        next = start or self.bottom_right_corner
        while current != next:
            current = next
            left = current << 1 & self.not_right_edge
            right = current >> 1 & self.not_left_edge
            up = current << self.width & self.not_bottom_edge
            down = current >> self.width & self.not_top_edge
            next = (current | left | right | up | down) & empty_squares
        return current
    def connected_diagonals(self, maze):
        assert self.height == self.width
        connected = self.maze_to_binary(self.connected_squares(maze))
        return [int(connected[(self.height + 1) * i]) for i in range(self.height)]
    def generate_labelled_mazes(self, batch_size):
        mazes = []
        labels = []
        for _ in range(batch_size):
            maze = self.generate()
            connected_diagonals = self.connected_diagonals(maze)
            mazes.append(self.maze_to_input(maze))
            labels.append(connected_diagonals)
        mazes = np.array(mazes).astype(np.float32)  # [batch, H, W, 1]
        labels = np.array(labels).astype(np.float32)  # [batch, W]
        return mazes, labels
    def generate_mazes(self, batch_size):
        return [self.maze_to_input(self.generate()) for _ in range(batch_size)]
    def maze_to_input(self, maze):
        maze = self.maze_to_binary(maze)
        maze = [[[int(maze[i + j])] for j in range(self.width)] for i in range(0, self.height * self.width, self.width)]
        return maze

def _make_norm_bn(num_features):
    return nn.BatchNorm2d(num_features, eps=NORM_BN_EPSILON, momentum=NORM_BN_MOMENTUM, affine=True)

def _make_preact_bn(num_features, dim):
    cls = nn.BatchNorm2d if dim == 2 else nn.BatchNorm1d
    bn = cls(num_features, eps=PREACT_BN_EPSILON, momentum=PREACT_BN_MOMENTUM, affine=True)
    nn.init.ones_(bn.weight)
    bn.weight.requires_grad_(False)
    return bn

def _variance_scaling_init_(weight):
    fan_in = weight.shape[1] * weight.shape[2] * weight.shape[3]
    std = math.sqrt(1.3 * 2.0 / fan_in)
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        # slim conv2d weights_initializer = variance_scaling_initializer().
        _variance_scaling_init_(self.conv.weight)
        self.norm_bn = _make_norm_bn(out_ch)
        self.preact_bn = _make_preact_bn(out_ch, dim=2)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm_bn(x)
        x = self.preact_bn(x)
        return F.relu(x)

class FCHead(nn.Module):
    def __init__(self, in_dim, maze_size, out_activation=None, regularize=True):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, 32)
        self.preact_bn = _make_preact_bn(32, dim=1)
        self.fc1 = nn.Linear(32, maze_size)
        self.out_activation = out_activation
        self.regularize = regularize
        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.zeros_(self.fc0.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.preact_bn(x))
        x = self.fc1(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

class PredictronCore(nn.Module):
    def __init__(self, maze_size):
        super().__init__()
        flat_dim = 32 * maze_size * maze_size
        self.value_head = FCHead(flat_dim, maze_size, out_activation=None, regularize=False)
        self.conv1 = ConvBlock(32, 32)
        self.reward_head = FCHead(flat_dim, maze_size, out_activation=None, regularize=True)
        self.gamma_head = FCHead(flat_dim, maze_size, out_activation=torch.sigmoid, regularize=True)
        self.lambda_head = FCHead(flat_dim, maze_size, out_activation=torch.sigmoid, regularize=True)
        self.conv2 = ConvBlock(32, 32)
        self.conv3 = ConvBlock(32, 32)
    def forward(self, state):
        b = state.shape[0]
        value = self.value_head(state.permute(0, 2, 3, 1).reshape(b, -1))
        net = self.conv1(state)
        net_flat = net.permute(0, 2, 3, 1).reshape(b, -1)
        reward = self.reward_head(net_flat)
        gamma = self.gamma_head(net_flat)
        lambda_ = self.lambda_head(net_flat)
        net = self.conv2(net)
        net = self.conv3(net)
        return net, reward, gamma, lambda_, value

class Predictron(nn.Module):
    def __init__(self, maze_size=20, max_depth=16):
        super().__init__()
        self.maze_size = maze_size
        self.max_depth = max_depth
        self.state_conv1 = ConvBlock(1, 32)
        self.state_conv2 = ConvBlock(32, 32)
        self.core = PredictronCore(maze_size)
    def forward(self, x):
        device = x.device
        b = x.shape[0]
        ms = self.maze_size
        state = self.state_conv1(x)
        state = self.state_conv2(state)
        rewards_arr, gammas_arr, lambdas_arr, values_arr = [], [], [], []
        for _ in range(self.max_depth):
            state, reward, gamma, lambda_, value = self.core(state)
            rewards_arr.append(reward)
            gammas_arr.append(gamma)
            lambdas_arr.append(lambda_)
            values_arr.append(value)
        _, _, _, _, value = self.core(state)
        values_arr.append(value)
        # rewards: [B, K, ms] -> prepend zeros -> [B, K + 1, ms]
        rewards = torch.stack(rewards_arr, dim=1)
        rewards = torch.cat([torch.zeros(b, 1, ms, device=device), rewards], dim=1)
        # gammas: [B, K, ms] -> prepend ones -> [B, K + 1, ms]
        gammas = torch.stack(gammas_arr, dim=1)
        gammas = torch.cat([torch.ones(b, 1, ms, device=device), gammas], dim=1)
        # lambdas: [B, K, ms]
        lambdas = torch.stack(lambdas_arr, dim=1)
        # values: [B, K + 1, ms]
        values = torch.stack(values_arr, dim=1)
        g_preturns = self._preturns(rewards, gammas, values)
        g_lambda_preturns = self._lambda_preturns(rewards, gammas, lambdas, values)
        return g_preturns, g_lambda_preturns
    # EQ 2: preturns
    def _preturns(self, rewards, gammas, values):
        g_preturns = []
        for k in range(self.max_depth, -1, -1):
            g_k = values[:, k, :]
            for kk in range(k, 0, -1):
                g_k = rewards[:, kk, :] + gammas[:, kk, :] * g_k
            g_preturns.append(g_k)
        # reverse to make 0...K from K...0
        g_preturns = g_preturns[::-1]
        return torch.stack(g_preturns, dim=1)  # [B, K + 1, ms]
    # EQ 4: lambda-preturns
    def _lambda_preturns(self, rewards, gammas, lambdas, values):
        g_k = values[:, -1, :]
        for k in range(self.max_depth - 1, -1, -1):
            g_k = (1 - lambdas[:, k, :]) * values[:, k, :] + lambdas[:, k, :] * (rewards[:, k + 1, :] + gammas[:, k + 1, :] * g_k)
        return g_k  # [B, ms]
    def regularization_loss(self):
        total = None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                sq = torch.sum(m.weight**2)
            elif isinstance(m, FCHead) and m.regularize:
                sq = torch.sum(m.fc0.weight**2) + torch.sum(m.fc1.weight**2)
            else:
                continue
            total = sq if total is None else total + sq
        if total is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return WEIGHT_DECAY * 0.5 * total

# Loss -- Eqn (5) preturn loss + Eqn (7) lambda-preturn loss + L2 decay
def predictron_loss(model, g_preturns, g_lambda_preturns, targets):
    # Eqn (5): tile targets to [B, K + 1, ms] and compare with preturns
    targets_tiled = targets.unsqueeze(1).expand(-1, g_preturns.shape[1], -1)
    loss_preturns = F.mse_loss(g_preturns, targets_tiled)
    # Eqn (7): lambda-preturn loss
    loss_lambda_preturns = F.mse_loss(g_lambda_preturns, targets)
    total_loss = 2.0 * loss_preturns + 2.0 * loss_lambda_preturns + model.regularization_loss()
    return total_loss, loss_preturns, loss_lambda_preturns

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    model = Predictron(maze_size=args.maze_size, max_depth=args.max_depth)
    model.to(device)
    model.train()
    logger.info("Trainable variables:")
    logger.info("*" * 30)
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.info("%s %s", name, tuple(p.shape))
    logger.info("*" * 30)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    # background maze-generation threads
    maze_queue = queue.Queue(100)
    def maze_generator():
        maze_gen = MazeGenerator(height=args.maze_size, width=args.maze_size, density=args.maze_density)
        while True:
            maze_ims, maze_labels = maze_gen.generate_labelled_mazes(args.batch_size)
            maze_queue.put((maze_ims, maze_labels))
    for _ in range(args.num_threads):
        t = threading.Thread(target=maze_generator, daemon=True)
        t.start()
    train_dir = os.path.join(args.train_dir, "max_steps_{}".format(args.max_depth))
    os.makedirs(train_dir, exist_ok=True)
    for step in range(args.max_steps):
        start_time = time.time()
        maze_ims_np, maze_labels_np = maze_queue.get()
        maze_ims = torch.from_numpy(np.transpose(maze_ims_np, (0, 3, 1, 2))).to(device)
        maze_labels = torch.from_numpy(maze_labels_np).to(device)
        g_preturns, g_lambda_preturns = model(maze_ims)
        total_loss, loss_preturns, loss_lambda_preturns = predictron_loss(model, g_preturns, g_lambda_preturns, maze_labels)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        duration = time.time() - start_time
        loss_value = total_loss.item()
        assert not np.isnan(loss_value), "Model diverged with loss = NaN"
        if step % 10 == 0:
            examples_per_sec = args.batch_size / duration
            sec_per_batch = duration
            format_str = "%s: step %d, loss = %.4f, loss_preturns = %.4f, loss_lambda_preturns = %.4f (%.1f examples/sec; %.3f sec/batch)"
            logger.info(
                format_str
                % (datetime.datetime.now(), step, loss_value, loss_preturns.item(), loss_lambda_preturns.item(), examples_per_sec, sec_per_batch)
            )
        if step % 1000 == 0 or (step + 1) == args.max_steps:
            checkpoint_path = os.path.join(train_dir, "model.ckpt-{}.pt".format(step))
            torch.save({"step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, checkpoint_path)

def parse_args():
    p = argparse.ArgumentParser(description="PyTorch Predictron (maze task).")
    p.add_argument("--train_dir", type=str, default="./ckpts/predictron_train", help="dir to save checkpoints")
    p.add_argument("--max_steps", type=int, default=10000000, help="num of batches")
    p.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    p.add_argument("--batch_size", type=int, default=128, help="batch size")
    p.add_argument("--maze_size", type=int, default=20, help="size of maze (square)")
    p.add_argument("--maze_density", type=float, default=0.3, help="maze density")
    p.add_argument("--max_depth", type=int, default=16, help="maximum model depth")
    p.add_argument("--max_grad_norm", type=float, default=10.0, help="clip grad norm into this value")
    p.add_argument("--num_threads", type=int, default=10, help="num of threads used to generate mazes")
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
