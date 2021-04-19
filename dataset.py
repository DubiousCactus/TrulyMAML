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
import random
import torch
import math
import os

from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, SubsetRandomSampler
from torchmeta.toy import Harmonic, SinusoidAndLine
from tqdm import tqdm
from PIL import Image

from const import device


# TODO: Some heavy refactoring needed!

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
    def __init__(self, tasks_num: int, samples_per_task: int, k_shot: int, k_query: int,
            meta_batch_size: int):
        print(f"[*] Generating {tasks_num} sinewaves of random phases and magnitudes...")
        self.tasks = []
        self.meta_batch_size = meta_batch_size
        assert k_query <= samples_per_task-k_shot, "k_query too big!"
        for _ in tqdm(range(tasks_num)):
            sine_wave = SineWave(samples=samples_per_task)
            # sine_wave.shuffle() Shuffling induces terrible performance and slow
            # converging for regression!
            meta_train_loader = DataLoader(
                    sine_wave,
                    batch_size=10,
                    # num_workers=8,
                    sampler=list(range(k_shot)),
                    pin_memory=False)
            meta_test_loader = DataLoader(
                    sine_wave,
                    batch_size=10,
                    # num_workers=8,
                    sampler=list(range(k_shot, k_shot+k_query)),
                    pin_memory=False)
            self.tasks.append((meta_train_loader, meta_test_loader))

    def __iter__(self):
        for t in self.tasks:
            yield t

    def __getitem__(self, idx):
        return self.tasks[idx]

    def __len__(self):
        return len(self.tasks)

    def __next__(self):
        return random.sample(self.tasks, self.meta_batch_size)


class OmniglotDataset:
    def __init__(self, batch_size: int, img_size: int, k_shot: int, k_query: int, n_way: int,
            evaluation: bool = False):
        assert (k_shot + k_query) <= 20, "Not enough samples per class for such k-shot and k-query values!"
        self.idx = 0
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_way = n_way
        self.eval = evaluation
        path = os.path.join('datasets', 'omniglot.npy')
        if os.path.exists(path) and device == "cpu":
            print("[*] Loading Omniglot from a saved file...")
            self.dataset = np.load(path)
        else:
            print("[*] Loading and preparing Omniglot...")
            self.dataset = self._load(not evaluation, img_size, batch_size)
            if device == "cpu":
                np.save(path, self.dataset)

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
            data.append(np.array(x, dtype=np.float32))
            t.update()
        data = torch.tensor(data, device=device)
        del tmp
        t.close()
        return data

    def __iter__(self):
        self.idx = 0
        return self

    @property
    def total_batches(self):
        return self.dataset.shape[0] // self.n_way + 1

    def __next__(self):
        '''
        Build a batch of N (for N-way classification) tasks, where each task is a random class.
        '''
        batch = []
        classes = (np.random.choice(self.dataset.shape[0], self.n_way, False) if not self.eval
                else list(range(self.idx, min(self.dataset.shape[0], self.idx+self.n_way))))
        for i, class_ in enumerate(classes):
            samples = np.random.choice(self.dataset.shape[1], self.k_shot+self.k_query, False)
            support = DataLoader(
                    list(zip(self.dataset[class_][samples[:self.k_shot]], torch.tensor([i]*self.k_shot, device=device))),
                    batch_size=self.k_shot,
                    shuffle=True,
                    pin_memory=False)
            query = DataLoader(
                    list(zip(self.dataset[class_][samples[self.k_shot:]], torch.tensor([i]*self.k_query, device=device))),
                    batch_size=self.k_query,
                    shuffle=True,
                    pin_memory=False)
            batch.append((support, query))
        self.idx += self.n_way
        return batch

    def __len__(self):
        return self.total_batches


class HarmonicDataset:
    def __init__(self, tasks_num: int, samples_per_task: int, k_shot: int, k_query: int,
            meta_batch_size: int):
        print(f"[*] Generating {tasks_num} harmonic functions of random phases and magnitudes...")
        self.tasks = []
        self.meta_batch_size = meta_batch_size
        assert k_query <= samples_per_task-k_shot, "k_query too big!"
        dataset = Harmonic(samples_per_task, tasks_num, transform=lambda x: x.astype(np.float32),
                target_transform=lambda x: x.astype(np.float32))
        for t in tqdm(dataset):
            meta_train_loader = DataLoader(
                    t,
                    batch_size=k_shot,
                    # num_workers=8,
                    sampler=list(range(k_shot)),
                    pin_memory=False)
            meta_test_loader = DataLoader(
                    t,
                    batch_size=k_query,
                    # num_workers=8,
                    sampler=list(range(k_shot, k_shot+k_query)),
                    pin_memory=False)
            self.tasks.append((meta_train_loader, meta_test_loader))

    def __iter__(self):
        for t in self.tasks:
            yield t

    def __getitem__(self, idx):
        return self.tasks[idx]

    def __len__(self):
        return len(self.tasks)

    def __next__(self):
        return random.sample(self.tasks, self.meta_batch_size)


class SinusoidAndLineDataset:
    def __init__(self, tasks_num: int, samples_per_task: int, k_shot: int, k_query: int,
            meta_batch_size: int):
        print(f"[*] Generating {tasks_num} sinusoid and line functions of random phases and magnitudes or random slopes and intercepts...")
        self.tasks = []
        self.meta_batch_size = meta_batch_size
        assert k_query <= samples_per_task-k_shot, "k_query too big!"
        dataset = SinusoidAndLine(samples_per_task, tasks_num, transform=lambda x: x.astype(np.float32),
                target_transform=lambda x: x.astype(np.float32))
        for t in tqdm(dataset):
            meta_train_loader = DataLoader(
                    t,
                    batch_size=k_shot,
                    # num_workers=8,
                    sampler=list(range(k_shot)),
                    pin_memory=False)
            meta_test_loader = DataLoader(
                    t,
                    batch_size=k_query,
                    # num_workers=8,
                    sampler=list(range(k_shot, k_shot+k_query)),
                    pin_memory=False)
            self.tasks.append((meta_train_loader, meta_test_loader))

    def __iter__(self):
        for t in self.tasks:
            yield t

    def __getitem__(self, idx):
        return self.tasks[idx]

    def __len__(self):
        return len(self.tasks)

    def __next__(self):
        return random.sample(self.tasks, self.meta_batch_size)
