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

from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from pytictoc import TicToc
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class SineWaveDataset(Dataset):
    def __init__(self, samples=20):
        self.x = torch.linspace(-5.0, 5.0, samples)
        phase, magnitude = np.random.uniform(0, math.pi), np.random.uniform(0.1, 5.0)
        # phase, magnitude = 0, 1
        self.sin = lambda x: magnitude * torch.sin(x + phase)
        self.y = self.sin(self.x)
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


class SineWaveTasksDataset(Dataset):
    def __init__(self, tasks, sampled_per_task):
        tasks = []
        for n in range(tasks):
            x = torch.linspace(-5.0, 5.0, samples, device=device)
            phase, magnitude = np.random.uniform(0, math.pi), np.random.uniform(0.1, 5.0)
            y = magnitude * torch.sin(x + phase).to(device)
            samples = torch.stack((x, y)).T.to(device)
            tasks.append(samples)
        self.tasks = TensorDataset(tasks)

    def suffle(self):
        self.tasks = self.tasks[torch.randperm(self.tasks.size()[0]),:]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


def train(training_dataset, learner):
    print("[*] Training...")
    model = MAML(learner)
    model.to(device)
    model.fit(training_dataset, 25, 50000)
    # model.eval(test)
    # TODO: Maybe implement MAML's training within MAML itself
   #  criterion = torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    # # Training loop
    # for t in range(2000):
        # # TODO: Sample a batch of tasks (how many??)
        # optimizer.zero_grad()
        # y_pred = model(x) # x should be the batch of tasks
        # loss = criterion(y_pred, y)
        # if t % 100 == 99:
            # print(t, loss.item())

        # loss.backward()
        # optimizer.step()
    print("[*] Done!")


def conventional_train(train_dataset, eval_dataset, learner):
    print("[*] Training with a conventional optimizer...")
    # Make the training / eval splits
    model = learner
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss(reduction='sum')
    print("[*] Evaluating with random initialization...")
    total_loss = 0
    for i, (x, y) in enumerate(eval_dataset):
        y_pred = model(x.to(device))
        loss = criterion(y_pred, y.to(device))
        print(f"-> Batch {i}: {loss}")
        total_loss += loss
    avg_loss = total_loss.item() / len(eval_dataset)

    print(f"[*] Average evaluation loss: {avg_loss}")

    # t = TicToc()
    model.train()
    # t.tic()
    for i in range(2000):
        loss = 0
        optimizer.zero_grad()
        for x, y in train_dataset:
            y_pred = model(x.to(device))
            loss += criterion(y_pred, y.to(device))
        if i % 100 == 99:
            print(i, loss.item()/len(train_dataset))
            # t.toc()
            # t.tic()
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
                TensorDataset(sine_wave[:K]),
                batch_size=16,
                shuffle=True, # Shuffle at each epoch
                pin_memory=True)
        meta_test_loader = DataLoader(
                TensorDataset(sine_wave[K:]),
                batch_size=1,
                shuffle=False,
                pin_memory=True)
        tasks.append((meta_train_loader, meta_test_loader))
    return tasks


def main():
    learner = MLP(device)
    learner.to(device)
    dataset = prepare_sinewave_dataset(100, 25, 10)
    train(dataset, learner)
    # train_dataloader = DataLoader(
            # dataset,
            # batch_size=32,
            # shuffle=True, pin_memory=True)
    # dataset = prepare_sinewave(1)
    # eval_dataloader = DataLoader(
            # dataset,
            # batch_size=16, pin_memory=True)
    # conventional_train(train_dataloader, eval_dataloader, learner)


if __name__ == "__main__":
    main()
