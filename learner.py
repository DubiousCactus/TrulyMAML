#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 cactus <cactus@archcactus>
#
# Distributed under terms of the MIT license.

"""
Dummy learner examples. Hopefully drop in your own!
"""


import torch.nn.functional as F
import torch.nn as nn
import torch


class DummiePolyLearner(nn.Module):
    """For the sin wave 3rd deg polynomial regression"""
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 1)

    def forward(self, x):
        # Prepare the input tensor (x, x^2, x^3).
        p = torch.tensor([1, 2, 3])
        xx = x.pow(p)
        return self.net(xx)


class MLP(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(1, 40),
                nn.ReLU(),
                nn.Linear(40, 40),
                nn.ReLU(),
                nn.Linear(40, 1)).to(device)


    def forward(self, x):
        return self.net(x)


class ConvNetClassifier(nn.Module):
    def __init__(self, device, input_channels: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(input_channels, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64, n_classes),
                nn.Softmax()).to(device)

    def forward(self, x):
        return self.net(x)


class MLPClassifier(nn.Module):
    def __init__(self, device, input_shape: tuple, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(reduce(operator.mul, input_shape, 1), 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64, n_classes),
                nn.Softmax()).to(device)

    def forward(self, x):
        return self.net(x)
