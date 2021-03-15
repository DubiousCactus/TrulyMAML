#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
A few dataset wrapper classes
"""

import numpy as np
import torchvision
import torch
import math

from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, SubsetRandomSampler
from tqdm import tqdm
from PIL import Image

from const import device


class SineWave(Dataset):
    def __init__(self, samples=20):
        self.x = torch.linspace(-5.0, 5.0, samples, device=device)
        phase, magnitude = np.random.uniform(0, math.pi), np.random.uniform(0.1, 5.0)
        # phase, magnitude = 0, 1
        # self.sin = lambda x: magnitude * torch.sin(x + phase)
        self.y = magnitude * torch.sin(self.x + phase).to(device)

    def shuffle(self):
        indices = torch.randperm(self.x.size()[0])
        self.x = self.x[indices]
        self.y = self.y[indices]


    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        return self

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # assert self.sin(self.x[idx]) == self.y[idx], "Sample pairs are wrong!"
        return self.x[idx].unsqueeze(dim=0), self.y[idx].unsqueeze(dim=0)


# TODO: Implement Dataset and have .next() for next batch?
class SineWaveDataset:
    '''
    A dataset of random sinusoid tasks with meta-train & meta-test splits.
    '''
    def __init__(self, tasks_num: int, samples_per_task: int, K: int, N:int, randomize: bool):
        print(f"[*] Generating {tasks_num} sinewaves of random phases and magnitudes...")
        self.tasks = []
        assert N <= samples_per_task-K, "N too big!"
        for n in tqdm(range(tasks_num)):
            sine_wave = SineWave(samples=samples_per_task)
            # sine_wave.shuffle() Shuffling induces terrible performance and slow
            # converging for regression!
            meta_train_loader = DataLoader(
                    sine_wave,
                    batch_size=10,
                    # num_workers=8,
                    sampler=SubsetRandomSampler(range(K)) if randomize else list(range(K)),
                    pin_memory=False)
            meta_test_loader = DataLoader(
                    sine_wave,
                    batch_size=10,
                    # num_workers=8,
                    sampler=(SubsetRandomSampler(range(K, len(sine_wave))) if
                        randomize else list(range(K, K+N))),
                    pin_memory=False)
            self.tasks.append((meta_train_loader, meta_test_loader))


class OmniglotDataset:
    def __init__(self, batch_size: int, img_size: int, background: bool = False):
        print("[*] Loading Omniglot...")
        self.dataset = torchvision.datasets.Omniglot(root='./datasets/',
                download=True, background=background,
                transform=torchvision.transforms.Compose([
                    lambda x: x.convert('L'),
                    lambda x: x.resize((img_size, img_size)),
                    lambda x: np.reshape(x, (img_size, img_size, 1)),
                    # lambda x: np.transpose(x, [2, 0, 1]),
                    lambda x: x/255.]))
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        # TODO: pin memory or load on GPU?
        tmp = dict()
        self.x = []
        for x, y in self.dataset:
            if y not in tmp:
                tmp[y] = []
            tmp[y].append(x)
        for y, x in tmp.items():
            self.x.append(np.array(x))
        self.x = np.array(self.x).astype(np.float)
        del tmp
        print(self.x.shape)
