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
    def __init__(self):
        super().__init__()
#         self.net = nn.Sequential(
                # nn.Linear(1, 40),
                # nn.ReLU(),
                # nn.Linear(40, 40),
                # nn.ReLU(),
                # nn.Linear(40, 1))
        self.lin1 = nn.Linear(1, 40)
        # self.lin2 = nn.Linear(40, 40)
        self.lin3 = nn.Linear(40, 1)


    def forward(self, x):
        # print(x)
        x = self.lin1(x)
        # x = self.lin2(F.relu(x))
        return self.lin3(F.relu(x))
