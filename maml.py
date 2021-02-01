#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 cactus <cactus@archcactus>
#
# Distributed under terms of the MIT license.

"""
MAML module
"""

import numpy as np
import random
import torch
import math

from typing import List


class DummiePolyLearner(torch.nn.Module):
    """For the sin wave 3rd deg polynomial regression"""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(3, 1)

    def forward(self, x):
        # Prepare the input tensor (x, x^2, x^3).
        p = torch.tensor([1, 2, 3])
        xx = x.unsqueeze(-1).pow(p)
        return self.net(xx)


class MAML(torch.nn.Module):
    def __init__(self, learner: torch.nn.Module,
            meta_lr=1e-6, inner_lr=1e-6, K=5, steps=1):
        super().__init__()
        self.meta_lr = meta_lr # This term is beta in the paper
        # TODO: Make the inner learning rate optionally learnable
        self.inner_lr = inner_lr # This term is alpha in the paper
        # TODO: Make meta-training and meta-testing splits?
        self.dataset = None
        self.learner = learner
        self.K = K
        self.inner_steps = steps
        self.inner_optim = torch.optim.SGD(self.learner.parameters(),
                lr=self.meta_lr)
        self.inner_loss = torch.nn.MSELoss(reduction='sum')
        self.meta_loss = torch.nn.MSELoss(reduction='sum')
        # TODO: Make this a tensor of parameters of the size of the learner's
        # parameters
        # self.theta = torch.nn.Parameter(torch.randn(()))

    def forward(self, task):
        # Step 1. Meta-train with K samples from a task T
        task.shuffle()
        m_train, m_test = task[:self.K], task[self.K:]
        inner_grads = {}
        # TODO: vectorize this loop if possible
        for s in range(self.inner_steps):
            # TODO: Refactor this if possible. The gradients should be
            # accumulated on each tensor, meaning that inner_grads{} would
            # be written only after the for loop. Wait actually the two loops
            # might be nested the wrong way...
            self.learner.zero_grad()
            for i, (x, y) in enumerate(m_train):
                self.inner_optim.zero_grad()
                y_pred = self.learner(x)[0]
                print("y_pred: ", y_pred, "y: ", y)
                loss = self.inner_loss(y_pred, y)
                print(f"Loss={loss}")
                # Actually do not improve the model yet! Keep the
                # gradient on the side. But do use those gradients to
                # temporarily update the model during meta-testing for the
                # loss of the meta-objective!
                grad = torch.autograd.grad(loss,
                        self.learner.parameters())
                print(f"Grad={grad}")
                if i not in inner_grads:
                    # First innit the gradient for that task if not set
                    inner_grads[i] = grad
                else:
                    # TODO: This doesn't work! The gradient is the same and the
                    # tensors are not added together!
                    print(f"Incrementing inner grad of task {i} from {inner_grads[i]} with +grad = {grad}")
                    inner_grads[i] += grad
                    print(f"To {inner_grads[i]}")
        # Step 2. Meta-test with X (N-K?) samples from task T
        # First, save the parameters
        initial_param = self.learner.net.state_dict()
        print(initial_param)
        meta_loss = 0
        # TODO: vectorize this loop if possible
        for i, (x, y) in enumerate(m_test):
            '''For each task in m_test'''
            self.inner_optim.zero_grad()
            with torch.no_grad():
                # TODO: Parse all initial parameters and replace them one by
                # one
                # for param in initial_param:
                    # self.learner.net.
                for k, v in initial_param.items():
                    print(f"self.learner.{k}=", getattr(self.learner, k))
                    setattr(self.learner, k, v)
                    print(f"self.learner.{k}=", getattr(self.learner, k))
                # Update the parameters with the accumulated gradients on that task
                # self.learner.weight = initial_param['weight']
                # print("Initial weights: ", self.learner.weight, "Inner grads: ", inner_grads[i])
                # for k, v in inner_grads[i]:
                    # eval(self.learner.k) -= self.meta_lr * v
                # self.learner.weight -= self.meta_lr * inner_grads[i]
                y_pred = self.learner(x)
                # TODO: Accumulate the loss over all tasks in the meta-testing set
                meta_loss += self.inner_loss(y_pred, y)

        # Step 3. Now that the meta-loss is computed
        meta_loss.backward()
        self.inner_optim.step()
        # TODO: What the heck do we return?
        return torch.zeros([1,100])

    def fit(self, dataset: List[tuple]):
        # t = np.random.choice(dataset)
        t = dataset[0]
        self.forward(t)

