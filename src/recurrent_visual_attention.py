# https://arxiv.org/pdf/1406.6247

import argparse
import json
import os
import pickle
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

def str2bool(v):
    return v.lower() in ("true", "1")

def get_config():
    parser = argparse.ArgumentParser(description="RAM")
    g = parser.add_argument_group("Glimpse Network Params")
    g.add_argument("--patch_size", type=int, default=8)
    g.add_argument("--glimpse_scale", type=int, default=1)
    g.add_argument("--num_patches", type=int, default=1)
    g.add_argument("--loc_hidden", type=int, default=128)
    g.add_argument("--glimpse_hidden", type=int, default=128)
    c = parser.add_argument_group("Core Network Params")
    c.add_argument("--num_glimpses", type=int, default=6)
    c.add_argument("--hidden_size", type=int, default=256)
    r = parser.add_argument_group("Reinforce Params")
    r.add_argument("--std", type=float, default=0.05)
    r.add_argument("--M", type=int, default=1)
    d = parser.add_argument_group("Data Params")
    d.add_argument("--valid_size", type=float, default=0.1)
    d.add_argument("--batch_size", type=int, default=128)
    d.add_argument("--num_workers", type=int, default=4)
    d.add_argument("--shuffle", type=str2bool, default=True)
    d.add_argument("--show_sample", type=str2bool, default=False)
    t = parser.add_argument_group("Training Params")
    t.add_argument("--is_train", type=str2bool, default=True)
    t.add_argument("--momentum", type=float, default=0.5)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--init_lr", type=float, default=3e-4)
    t.add_argument("--lr_patience", type=int, default=20)
    t.add_argument("--train_patience", type=int, default=50)
    m = parser.add_argument_group("Misc.")
    m.add_argument("--use_gpu", type=str2bool, default=True)
    m.add_argument("--best", type=str2bool, default=True)
    m.add_argument("--random_seed", type=int, default=1)
    m.add_argument("--data_dir", type=str, default="./data")
    m.add_argument("--ckpt_dir", type=str, default="./ckpt")
    m.add_argument("--logs_dir", type=str, default="./logs/")
    m.add_argument("--use_tensorboard", type=str2bool, default=False)
    m.add_argument("--resume", type=str2bool, default=False)
    m.add_argument("--print_freq", type=int, default=10)
    m.add_argument("--plot_freq", type=int, default=1)
    return parser.parse_known_args()

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def prepare_dirs(config):
    for path in [config.data_dir, config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def save_config(config):
    model_name = "ram_{}_{}x{}_{}".format(config.num_glimpses, config.patch_size, config.patch_size, config.glimpse_scale)
    filename = model_name + "_params.json"
    param_path = os.path.join(config.ckpt_dir, filename)
    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))
    with open(param_path, "w") as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def get_train_valid_loader(data_dir, batch_size, random_seed, valid_size=0.1, shuffle=True, show_sample=False, num_workers=4, pin_memory=False):
    assert 0 <= valid_size <= 1, "[!] valid_size should be in the range [0, 1]."
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans)
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx), num_workers=num_workers, pin_memory=pin_memory
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx), num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, valid_loader

def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trans)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

class Retina:
    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s
    def foveate(self, x, l):
        phi = []
        size = self.g
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)
        phi = torch.cat(phi, 1)
        phi = phi.view(phi.shape[0], -1)
        return phi
    def extract_patch(self, x, l, size):
        B, C, H, W = x.shape
        start = self.denormalize(H, l)
        end = start + size
        x = F.pad(x, (size // 2, size // 2, size // 2, size // 2))
        patch = []
        for i in range(B):
            patch.append(x[i, :, start[i, 1] : end[i, 1], start[i, 0] : end[i, 0]])
        return torch.stack(patch)
    def denormalize(self, T, coords):
        return (0.5 * ((coords + 1.0) * T)).long()

class GlimpseNetwork(nn.Module):
    def __init__(self, h_g, h_l, g, k, s, c):
        super().__init__()
        self.retina = Retina(g, k, s)
        D_in = k * g * g * c
        self.fc1 = nn.Linear(D_in, h_g)
        self.fc2 = nn.Linear(2, h_l)
        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)
    def forward(self, x, l_t_prev):
        phi = self.retina.foveate(x, l_t_prev)
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))
        what = self.fc3(phi_out)
        where = self.fc4(l_out)
        return F.relu(what + where)

class CoreNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
    def forward(self, g_t, h_t_prev):
        return F.relu(self.i2h(g_t) + self.h2h(h_t_prev))

class ActionNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, h_t):
        return F.log_softmax(self.fc(h_t), dim=1)

class LocationNetwork(nn.Module):
    def __init__(self, input_size, output_size, std):
        super().__init__()
        self.std = std
        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)
    def forward(self, h_t):
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))
        l_t = Normal(mu, self.std).rsample().detach()
        log_pi = torch.sum(Normal(mu, self.std).log_prob(l_t), dim=1)
        l_t = torch.clamp(l_t, -1, 1)
        return log_pi, l_t

class BaselineNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, h_t):
        return self.fc(h_t.detach())

class RecurrentAttention(nn.Module):
    def __init__(self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes):
        super().__init__()
        self.std = std
        self.sensor = GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.rnn = CoreNetwork(hidden_size, hidden_size)
        self.locator = LocationNetwork(hidden_size, 2, std)
        self.classifier = ActionNetwork(hidden_size, num_classes)
        self.baseliner = BaselineNetwork(hidden_size, 1)
    def forward(self, x, l_t_prev, h_t_prev, last=False):
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)
        log_pi, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()
        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, b_t, log_probas, log_pi
        return h_t, l_t, b_t, log_pi

class Trainer:
    def __init__(self, config, data_loader):
        self.config = config
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size
        self.std = config.std
        self.M = config.M
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 10
        self.num_channels = 1
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "ram_{}_{}x{}_{}".format(config.num_glimpses, config.patch_size, config.patch_size, config.glimpse_scale)
        self.plot_dir = "./plots/" + self.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.init_lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=self.lr_patience)
    def reset(self):
        h_t = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float, device=self.device, requires_grad=True)
        l_t = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True
        return h_t, l_t
    def train(self):
        if self.resume:
            self.load_checkpoint(best=False)
        print("\n[*] Train on {} samples, validate on {} samples".format(self.num_train, self.num_valid))
        for epoch in range(self.start_epoch, self.epochs):
            print("\nEpoch: {}/{} - LR: {:.6f}".format(epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]))
            train_loss, train_acc = self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validate(epoch)
            self.scheduler.step(-valid_acc)
            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            print((msg1 + msg2).format(train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc))
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                },
                is_best,
            )
    def train_one_epoch(self, epoch):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                plot = (epoch % self.plot_freq == 0) and (i == 0)
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()
                imgs = [x[0:9]]
                locs = []
                log_pi = []
                baselines = []
                for t in range(self.num_glimpses - 1):
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)
                    locs.append(l_t[0:9])
                    baselines.append(b_t)
                    log_pi.append(p)
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                log_pi.append(p)
                baselines.append(b_t)
                locs.append(l_t[0:9])
                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)
                predicted = torch.max(log_probas, 1)[1]
                R = (predicted.detach() == y).float()
                R = R.unsqueeze(1).repeat(1, self.num_glimpses)
                loss_action = F.nll_loss(log_probas, y)
                loss_baseline = F.mse_loss(baselines, R)
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)
                loss = loss_action + loss_baseline + loss_reinforce * 0.01
                correct = (predicted == y).float()
                acc = 100 * (correct.sum() / len(y))
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])
                loss.backward()
                self.optimizer.step()
                toc = time.time()
                batch_time.update(toc - tic)
                pbar.set_description("{:.1f}s - loss: {:.3f} - acc: {:.3f}".format((toc - tic), loss.item(), acc.item()))
                pbar.update(self.batch_size)
                if plot:
                    imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    locs = [l.cpu().data.numpy() for l in locs]
                    pickle.dump(imgs, open(self.plot_dir + "g_{}.p".format(epoch + 1), "wb"))
                    pickle.dump(locs, open(self.plot_dir + "l_{}.p".format(epoch + 1), "wb"))
            return losses.avg, accs.avg
    @torch.no_grad()
    def validate(self, epoch):
        losses = AverageMeter()
        accs = AverageMeter()
        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.repeat(self.M, 1, 1, 1)
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)
                baselines.append(b_t)
                log_pi.append(p)
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
            log_pi.append(p)
            baselines.append(b_t)
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)
            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)
            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)
            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)
            loss = loss_action + loss_baseline + loss_reinforce * 0.01
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])
        return losses.avg, accs.avg
    @torch.no_grad()
    def test(self):
        correct = 0
        self.load_checkpoint(best=self.best)
        for i, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.repeat(self.M, 1, 1, 1)
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()
            for t in range(self.num_glimpses - 1):
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)
            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        perc = (100.0 * correct) / self.num_test
        error = 100 - perc
        print("[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(correct, self.num_test, perc, error))
    def save_checkpoint(self, state, is_best):
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            best_name = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, best_name))
    def load_checkpoint(self, best=False):
        print("[*] Loading model from {}".format(self.ckpt_dir))
        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        if best:
            print("[*] Loaded {} checkpoint @ epoch {} with best valid acc of {:.3f}".format(filename, ckpt["epoch"], ckpt["best_valid_acc"]))
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))

def main(config):
    prepare_dirs(config)
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}
    if config.is_train:
        dloader = get_train_valid_loader(
            config.data_dir, config.batch_size, config.random_seed, config.valid_size, config.shuffle, config.show_sample, **kwargs
        )
    else:
        dloader = get_test_loader(config.data_dir, config.batch_size, **kwargs)
    trainer = Trainer(config, dloader)
    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
