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


class MAML(torch.nn.Module):
    def __init__(self, learner: torch.nn.Module,
            meta_lr=1e-6, inner_lr=1e-6, K=1, steps=1):
        super().__init__()
        self.meta_lr = meta_lr # This term is beta in the paper
        # TODO: Make the inner learning rate optionally learnable
        self.inner_lr = inner_lr # This term is alpha in the paper
        # TODO: Make meta-training and meta-testing splits?
        self.dataset = None
        self.learner = learner
        self.K = K
        # TODO: Fix the inner steps mechanism: the gradient does not improve
        # because the model isn't updated!
        self.inner_steps = steps
        self.inner_optim = torch.optim.SGD(self.learner.parameters(),
                lr=self.meta_lr)
        self.inner_loss = torch.nn.MSELoss(reduction='sum')
        self.meta_loss = torch.nn.MSELoss(reduction='sum')

    def forward(self, tasks_batch):
        # For each task in the batch
        initial_param = self.learner.net.state_dict()
        meta_loss = 0
        for i, task in enumerate(tasks_batch):
            # Step 1. Meta-train with K samples from a task T
            task.shuffle()
            m_train, m_test = task[:self.K], task[self.K:]
            inner_grads = {}
            # TODO: vectorize this loop if possible
            for s in range(self.inner_steps):
                self.learner.zero_grad()
                for x, y in m_train:
                    self.inner_optim.zero_grad()
                    y_pred = self.learner(x)[0]
                    loss = self.inner_loss(y_pred, y)
                    # print(f"Meta-training Loss={loss}")
                    # Actually do not improve the model yet! Keep the
                    # gradient on the side. But do use those gradients to
                    # temporarily update the model during meta-testing for the
                    # loss of the meta-objective!
                    grad = torch.autograd.grad(loss,
                            self.learner.parameters())
                    if i not in inner_grads:
                        inner_grads[i] = list(grad)
                    else:
                        for j, tensor in enumerate(inner_grads[i]):
                            tensor += grad[j]
            # Step 2. Meta-test with X (N-K?) samples from task T
            # Update the parameters with the accumulated gradients on that task
            # TODO: Do we update the parameters continuously for the whole
            # batch or reset at each task?
            for j, (k, v) in enumerate(initial_param.items()):
                setattr(self.learner.net, k, torch.nn.Parameter(v - (self.inner_lr * inner_grads[i][j])))
            # TODO: vectorize this loop if possible
            for x, y in m_test:
                '''For each task in m_test'''
                # self.inner_optim.zero_grad()
                y_pred = self.learner(x)
                # Accumulate the loss over all tasks in the meta-testing set
                meta_loss += self.meta_loss(y_pred, y)
            # Restore initial parameters for learner
            for k, v in initial_param.items():
                setattr(self.learner.net, k, torch.nn.Parameter(v))

        # Step 3. Now that the meta-loss is computed
        print(f"Meta-testing Cummulative Loss={meta_loss}")
        # For now, let's try a manual parameter update
        meta_gradient = torch.autograd.grad(meta_loss,
                self.learner.parameters(), allow_unused=True)
        print(f"Meta-gradient: {meta_gradient}")
        assert None not in meta_gradient, "Empty meta-gradient!"
        for j, (k, v) in enumerate(initial_param.items()):
            setattr(self.learner.net, k, torch.nn.Parameter(v - (self.meta_lr * meta_gradient[j])))
        # meta_loss.backward()
        # self.inner_optim.step()
        # TODO: What the heck do we return?

    def fit(self, dataset: List[tuple], iterations: int):
        # t = np.random.choice(dataset)
        # Sample a batch of tasks
        for i in range(iterations):
            batch = dataset[:6]
            self.forward(batch)
        # TODO: Meta-testing here?
        # TODO: Computation of the meta-objective here?
        # TODO: Meta-optimization here?

