#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 cactus <cactus@archcactus>
#
# Distributed under terms of the MIT license.

"""

"""

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import random
import torch
import math

from learner import DummiePolyLearner, MLP
from maml import MAML

from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, SubsetRandomSampler
from pytictoc import TicToc
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class SineWaveDataset(Dataset):
    def __init__(self, samples=20):
        self.x = torch.linspace(-5.0, 5.0, samples, device=device)
        phase, magnitude = np.random.uniform(0, math.pi), np.random.uniform(0.1, 5.0)
        # phase, magnitude = 0, 1
        self.sin = lambda x: magnitude * torch.sin(x + phase)
        self.y = self.sin(self.x).to(device)
        # self.samples = torch.stack((x, y)).T.to(device)

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


def train(training_dataset, learner):
    print("[*] Training...")
    model = MAML(learner)
    model.to(device)
    model.fit(training_dataset, 1, 50000)
    # model.eval(test)
    print("[*] Done!")


def conventional_train(dataset, learner):
    print("[*] Training with a conventional optimizer...")
    # Make the training / eval splits
    model = learner
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss(reduction='sum')
    train_dataset, eval_dataset = dataset[0][0], dataset[0][1]
    print("[*] Evaluating with random initialization...")
    total_loss = 0
    for i, (x, y) in enumerate(eval_dataset):
        y_pred = model(x.to(device))
        loss = criterion(y_pred, y.to(device))
        print(f"-> Batch {i}: {loss}")
        total_loss += loss
    avg_loss = total_loss.item() / len(eval_dataset)

    print(f"[*] Average evaluation loss: {avg_loss}")

    t = TicToc()
    model.train()
    t.tic()
    for i in range(2000):
        loss = 0
        optimizer.zero_grad()
        for x, y in train_dataset:
            y_pred = model(x)
            loss += criterion(y_pred, y)
        if i % 100 == 99:
            print(i, loss.item()/len(train_dataset))
            t.toc()
            t.tic()
        loss.backward()
        optimizer.step()

    print("[*] Evaluating...")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(eval_dataset):
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y.to(device))
            total_loss += loss
    avg_loss = total_loss.item() / len(eval_dataset)
    print(f"[*] Average evaluation loss: {avg_loss}")

def prepare_omniglot():
    print("[*] Loading Omniglot...")
    omniglot = torchvision.datasets.Omniglot(root='./datasets/',
            download=True)
    omniglot_loader = DataLoader(omniglot, batch_size=4,
            shuffle=True, num_workers=6)
    return omniglot


def prepare_sinewave_dataset(tasks_num: int, samples_per_task: int, K: int) -> List[Tuple[DataLoader]]:
    print(f"[*] Generating {tasks_num} sinwaves of random phases and magnitudes...")
    tasks = []
    for n in tqdm(range(tasks_num)):
        sine_wave = SineWaveDataset(samples=samples_per_task)
        sine_wave.shuffle()
        meta_train_loader = DataLoader(
                sine_wave,
                batch_size=10,
                # num_workers=8,
                sampler=SubsetRandomSampler(range(K)),
                pin_memory=False)
        meta_test_loader = DataLoader(
                sine_wave,
                batch_size=1,
                # num_workers=8,
                sampler=SubsetRandomSampler(range(K, len(sine_wave))),
                pin_memory=False)
        tasks.append((meta_train_loader, meta_test_loader))
    return tasks


def main():
    learner = MLP(device)
    learner.to(device)
    dataset = prepare_sinewave_dataset(500, 20, 10)
    train(dataset, learner)
    # conventional_train(dataset, learner)


if __name__ == "__main__":
    main()
