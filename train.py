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

from typing import List
from tqdm import tqdm
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


class SineWaveDataset(torch.utils.data.Dataset):
    def __init__(self, samples=5000):
        x = torch.linspace(-5.0, 5.0, samples, device=device)
        phase, magnitude = np.random.uniform(0, math.pi), np.random.uniform(0.1, 5.0)
        # phase, magnitude = 0, 1
        self.sin = lambda x: magnitude * torch.sin(x + phase)
        y = self.sin(x).to(device)
        self.samples = torch.stack((x, y)).T.to(device)

    def shuffle(self):
        self.samples = self.samples[torch.randperm(self.samples.size()[0]),:]

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        assert self.sin(self.samples[idx][0][0]) == self.samples[idx][0][1], "Sample pairs are wrong!"
        return self.samples[idx]



def train(dataset, learner):
    print("[*] Training...")
    # Make the training / eval splits
    t_size = int(0.7*len(dataset))
    train, test = dataset[:t_size], dataset[t_size:]

    model = MAML(learner)
    model.to(device)
    model.fit(train, 150)
    model.eval(test)
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


def conventional_train(dataset, learner):
    print("[*] Training with a conventional optimizer...")
    # Make the training / eval splits
    task = dataset[0]
    batch_size = 10
    t_size = int(0.7*len(task))
    task.shuffle()
    train, test = task[:t_size], task[t_size:]
    model = learner
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss(reduction='sum')
    print("[*] Evaluating with random initialization...")
    total_loss = 0
    for x, y in test:
        y_pred = model(x.unsqueeze(dim=0))[0]
        loss = criterion(y_pred, y)
        total_loss += loss
    avg_loss = total_loss.item() / len(test)

    print(f"[*] Average evaluation loss: {avg_loss}")
    for i in range(2000):
        indices = torch.randperm(len(train))[:batch_size]
        batch = train[indices]
        loss = 0
        optimizer.zero_grad()
        for x, y in batch:
            y_pred = model(x.unsqueeze(dim=0))[0]
            loss += criterion(y_pred, y)
        if i % 100 == 99:
            print(i, loss.item()/batch_size)
        loss.backward()
        optimizer.step()

    print("[*] Evaluating...")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in test:
            y_pred = model(x.unsqueeze(dim=0))[0]
            loss = criterion(y_pred, y)
            total_loss += loss
    avg_loss = total_loss.item() / len(test)
    print(f"[*] Average evaluation loss: {avg_loss}")

def prepare_omniglot():
    print("[*] Loading Omniglot...")
    omniglot = torchvision.datasets.Omniglot(root='./datasets/',
            download=True)
    omniglot_loader = torch.utils.data.DataLoader(omniglot, batch_size=4,
            shuffle=True, num_workers=6)
    return omniglot


def prepare_sinewave(task_number: int) -> List[torch.tensor]:
    print(f"[*] Generating {task_number} sinwaves of random phases and magnitudes...")
    tasks = []
    for n in tqdm(range(task_number)):
        random_sine = SineWaveDataset()
        random_sine.shuffle()
        tasks.append(random_sine)
    # print("[*] Plotting the sinwave...")
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # plt.show()
    return tasks


def main():
    learner = MLP()
    learner.to(device)
    train(prepare_sinewave(50), learner)
    # conventional_train(prepare_sinewave(1), learner)


if __name__ == "__main__":
    main()
