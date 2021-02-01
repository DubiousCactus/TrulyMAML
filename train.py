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
import torch
import math

from maml import MAML, DummiePolyLearner

from typing import List
from tqdm import tqdm
from PIL import Image


class SineWaveDataset(torch.utils.data.Dataset):
    def __init__(self, samples=2000):
        x = torch.linspace(-math.pi, math.pi, samples)
        phase, magnitude = np.random.uniform(-math.pi, math.pi), np.random.rand()
        y = magnitude * torch.sin(x + phase)
        self.samples = torch.stack((x, y)).T

    def shuffle(self):
        self.samples = self.samples[torch.randperm(self.samples.size()[0]),:]

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        return self.samples[idx]



def train(dataset, K=5):
    print("[*] Training...")
    # Make the training / eval splits
    t_size = int(0.7*len(dataset))
    train, test = dataset[:t_size], dataset[t_size:]

    model = MAML(DummiePolyLearner())
    model.fit(train)
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


def prepare_omniglot():
    print("[*] Loading Omniglot...")
    omniglot = torchvision.datasets.Omniglot(root='./omniglot/',
            download=True)
    omniglot_loader = torch.utils.data.DataLoader(omniglot, batch_size=4,
            shuffle=True, num_workers=6)
    return omniglot


def prepare_sinewave(task_number: int) -> List[torch.tensor]:
    print(f"[*] Generating {task_number} sinwaves of random phases and magnitudes...")
    tasks = []
    for n in tqdm(range(task_number)):
        tasks.append(SineWaveDataset())
    # print("[*] Plotting the sinwave...")
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # plt.show()
    return tasks


def main():
    train(prepare_sinewave(100))


if __name__ == "__main__":
    main()
