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
import os

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


class SineWaveDataset:
    '''
    A dataset of random sinusoid tasks with meta-train & meta-test splits.
    '''
    def __init__(self, tasks_num: int, samples_per_task: int, k_shot: int, k_query: int, randomize: bool):
        print(f"[*] Generating {tasks_num} sinewaves of random phases and magnitudes...")
        self.tasks = []
        assert k_query <= samples_per_task-k_shot, "k_query too big!"
        for _ in tqdm(range(tasks_num)):
            sine_wave = SineWave(samples=samples_per_task)
            # sine_wave.shuffle() Shuffling induces terrible performance and slow
            # converging for regression!
            meta_train_loader = DataLoader(
                    sine_wave,
                    batch_size=10,
                    # num_workers=8,
                    sampler=SubsetRandomSampler(range(k_shot)) if randomize else list(range(k_shot)),
                    pin_memory=False)
            meta_test_loader = DataLoader(
                    sine_wave,
                    batch_size=10,
                    # num_workers=8,
                    sampler=(SubsetRandomSampler(range(k_shot, len(sine_wave))) if
                        randomize else list(range(k_shot, k_shot+k_query))),
                    pin_memory=False)
            self.tasks.append((meta_train_loader, meta_test_loader))


class OmniglotDataset:
    def __init__(self, batch_size: int, img_size: int, k_shot: int, k_query: int, n_way: int, background: bool = False):
        assert k_shot + k_query <= 20, "Not enough samples per class for such k-shot and k-query values!"
        self.idx = 0
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_way = n_way
        path = os.path.join('datasets', 'omniglot.npy')
        if os.path.exists(path):
            print("[*] Loading Omniglot from a saved file...")
            self.x = np.load(path)
        else:
            print("[*] Loading and preparing Omniglot...")
            self.x = self._load(background, img_size, batch_size)
            np.save(path, self.x)

    def _load(self, background, img_size, batch_size):
        dataset = torchvision.datasets.Omniglot(root='./datasets/',
                download=True, background=background,
                transform=torchvision.transforms.Compose([
                    lambda x: x.convert('L'),
                    lambda x: x.resize((img_size, img_size)),
                    lambda x: np.reshape(x, (img_size, img_size, 1)),
                    lambda x: np.transpose(x, [2, 0, 1]),
                    lambda x: x/255.]))
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # TODO: pin memory or load on GPU?
        tmp = dict()
        data = []
        t = tqdm(total=len(dataset))
        for x, y in dataset:
            if y not in tmp:
                tmp[y] = []
            tmp[y].append(x)
            t.update()
        for y, x in tmp.items():
            data.append(np.array(x))
            t.update()
        data = np.array(data).astype(np.float32)
        del tmp
        t.close()
        return data

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        batch = []
        for j in range(self.n_way):
            # Build the support set with K shots
            # TODO: Randomly index the class/index?
            cls = self.idx + j
            support = DataLoader(
                    list(zip(self.x[cls][:self.k_shot], [j]*self.k_shot)),
                    batch_size=self.k_shot,
                    shuffle=True)
            query = DataLoader(
                    list(zip(self.x[cls][self.k_shot:self.k_shot+self.k_query],
                        [j]*self.k_query)),
                    batch_size=self.k_query,
                    shuffle=True)
            batch.append((support, query))
        self.idx += self.n_way
        return batch
