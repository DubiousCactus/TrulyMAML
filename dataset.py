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
from queue import SimpleQueue
from pytictoc import TicToc
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


class RegressionDataset:
    def __init__(self, tasks_num: int, samples_per_task: int, k_shot: int, k_query: int,
            meta_batch_size: int):
        self.tasks = []
        self.meta_batch_size = meta_batch_size
        assert k_query <= samples_per_task-k_shot, "k_query too big!"

    def __iter__(self):
        for i in range(0, len(self.tasks), self.meta_batch_size):
            yield self.tasks[i:i+self.meta_batch_size]

    def __getitem__(self, idx):
        return self.tasks[idx]

    def __len__(self):
        return len(self.tasks)

    def __next__(self):
        return random.sample(self.tasks, self.meta_batch_size)


'''
/!\ Using a batch size equal to the number of shots / queries seems to be much faster (of course)
than an arbitrary number, especially for large query sets. The convergence speed seems similar.
More data is needed to draw conclusions.
'''


class SineWaveDataset(RegressionDataset):
    '''
    A dataset of random sinusoid tasks with meta-train & meta-test splits.
    '''
    def __init__(self, tasks_num: int, samples_per_task: int, k_shot: int, k_query: int,
            meta_batch_size: int):
        super().__init__(tasks_num, samples_per_task, k_shot, k_query, meta_batch_size)
        print(f"[*] Generating {tasks_num} sinewaves of random phases and magnitudes...")
        for _ in tqdm(range(tasks_num)):
            sine_wave = SineWave(samples=samples_per_task)
            # sine_wave.shuffle() Shuffling induces terrible performance and slow
            # converging for regression!
            meta_train_loader = DataLoader(
                    sine_wave,
                    batch_size=k_shot,
                    # num_workers=8,
                    sampler=list(range(k_shot)),
                    pin_memory=False)
            meta_test_loader = DataLoader(
                    sine_wave,
                    batch_size=k_query,
                    # num_workers=8,
                    sampler=list(range(k_shot, k_shot+k_query)),
                    pin_memory=False)
            self.tasks.append((meta_train_loader, meta_test_loader))

class HarmonicDataset(RegressionDataset):
    def __init__(self, tasks_num: int, samples_per_task: int, k_shot: int, k_query: int,
            meta_batch_size: int):
        super().__init__(tasks_num, samples_per_task, k_shot, k_query, meta_batch_size)
        print(f"[*] Generating {tasks_num} harmonic functions of random phases and magnitudes...")
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


class SinusoidAndLineDataset(RegressionDataset):
    def __init__(self, tasks_num: int, samples_per_task: int, k_shot: int, k_query: int,
            meta_batch_size: int):
        super().__init__(tasks_num, samples_per_task, k_shot, k_query, meta_batch_size)
        print(f"[*] Generating {tasks_num} sinusoid and line functions of random phases and magnitudes or random slopes and intercepts...")
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


class OmniglotDataset:
    def __init__(self, meta_batch_size: int, img_size: int, k_shot: int, k_query: int, n_way: int,
            evaluation: bool = False):
        assert (k_shot + k_query) <= 20, "Not enough samples per class for such k-shot and k-query values!"
        self.idx = 0
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_way = n_way
        self.eval = evaluation
        self.img_size = img_size
        self.meta_batch_size = meta_batch_size
        path = os.path.join('datasets', 'omniglot.npy')
        if os.path.exists(path) and device == "cpu":
            print("[*] Loading Omniglot from a saved file...")
            self.dataset = np.load(path)
        else:
            print("[*] Loading and preparing Omniglot...")
            self.dataset = self._load(not evaluation, img_size)
            if device == "cpu":
                np.save(path, self.dataset)
        self._cache = self._load_in_cache()

    def _load(self, background, img_size):
        dataset = torchvision.datasets.Omniglot(root='./datasets/',
                download=True, background=background,
                transform=torchvision.transforms.Compose([
                    lambda x: x.convert('L'),
                    lambda x: x.resize((img_size, img_size)),
                    lambda x: np.reshape(x, (img_size, img_size, 1)),
                    lambda x: np.transpose(x, [2, 0, 1]),
                    lambda x: x/255.]))
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

    def _load_in_cache(self, size=100):
        print("[*] Loading data cache...")
        cache = SimpleQueue()
        for _ in tqdm(range(size)):
            batch = []
            spt_sz, qry_sz = self.n_way * self.k_shot, self.n_way * self.k_query
            for i in range(self.meta_batch_size):
                classes = (np.random.choice(self.dataset.shape[0], self.n_way, False) if not self.eval
                        else list(range(self.idx, min(self.dataset.shape[0], self.idx+self.n_way))))
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                for j, class_ in enumerate(classes):
                    samples = (np.random.choice(self.dataset.shape[1], self.k_shot+self.k_query, False) if not self.eval
                            else list(range(self.k_shot+self.k_query)))
                    x_spt.append(self.dataset[class_][samples[:self.k_shot]])
                    y_spt.append(torch.tensor([j]*self.k_shot, device=device))
                    x_qry.append(self.dataset[class_][samples[self.k_shot:]])
                    y_qry.append(torch.tensor([j]*self.k_query, device=device))

                # Shuffle the batch
                perm = torch.randperm(spt_sz)
                x_spt = torch.stack(x_spt, dim=0).reshape(spt_sz, 1, self.img_size, self.img_size)[perm]
                y_spt = torch.stack(y_spt, dim=0).reshape(spt_sz)[perm]
                perm = torch.randperm(qry_sz)
                x_qry = torch.stack(x_qry, dim=0).reshape(qry_sz, 1, self.img_size, self.img_size)[perm]
                y_qry = torch.stack(y_qry, dim=0).reshape(qry_sz)[perm]

                spt_loader = DataLoader(
                        list(zip(x_spt, y_spt)),
                        batch_size=self.k_shot,
                        shuffle=False,
                        pin_memory=False)
                qry_loader = DataLoader(
                        list(zip(x_qry, y_qry)),
                        batch_size=self.k_query,
                        shuffle=False,
                        pin_memory=False)
                batch.append((spt_loader, qry_loader))
            if self.eval:
                self.idx += self.n_way
            cache.put(batch) # Should not need exception handling
        return cache

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
        if self._cache.empty():
            self._cache = self._load_in_cache()
        batch = self._cache.get()
        return batch

    def __len__(self):
        return self.total_batches

