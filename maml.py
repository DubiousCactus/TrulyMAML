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
import higher
import torch
import math

from pytictoc import TicToc
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class MAML(torch.nn.Module):
    def __init__(self, learner: torch.nn.Module,
            meta_lr=1e-3, inner_lr=1e-3, K=10, steps=1):
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
        self.meta_opt = torch.optim.Adam(self.learner.parameters(),
                lr=self.meta_lr)
        self.inner_opt = torch.optim.SGD(self.learner.parameters(),
                lr=self.inner_lr)
        self.inner_loss = torch.nn.MSELoss(reduction='sum')
        self.meta_loss = torch.nn.MSELoss(reduction='sum')

    def forward(self, tasks_batch):
        # TODO: m_train should never intersect with m_test!
        # For each task in the batch
        initial_param = dict(list(self.learner.named_parameters()))
        initial_param_raw = self.learner.parameters()
        meta_loss = 0
        passes = 0
        self.learner.zero_grad()
        for i, task in enumerate(tasks_batch):
            # Step 1. Meta-train with K samples from a task T
            task.shuffle()
            m_train, m_test = task[:self.K], task[self.K:]
            adapted_parameters = initial_param.copy()
            # # TODO: vectorize this loop if possible
            for s in range(self.inner_steps):
                self.learner.zero_grad()
                step_loss = 0
                for x, y in m_train:
                    y_pred = self.learner(x.unsqueeze(dim=0))[0]
                    step_loss += self.inner_loss(y_pred, y)
                grad = torch.autograd.grad(step_loss,
                        self.learner.parameters())
                with torch.no_grad():
                    for j, (name, tensor) in enumerate(adapted_parameters.items()):
                        tensor += grad[j]
            # Step 2. Meta-test with X (N-K?) samples from task T
            with torch.no_grad():
                for name, param in self.learner.named_parameters():
                    param -= self.inner_lr * adapted_parameters[name]
#             with torch.no_grad():
                # for j, (k, v) in enumerate(initial_param):
                    # tree = k.split(".")
                    # prev = []
                    # attr = self.learner
                    # for t in tree:
                        # for p in prev:
                            # attr = getattr(attr, p)
                        # prev.append(t)
#                     setattr(attr, tree[-1], torch.nn.Parameter(v - (self.inner_lr * inner_grads[i][j])))
            # TODO: vectorize this loop if possible
            for x, y in m_test:
                y_pred = self.learner(x.unsqueeze(dim=0))[0] # Use the adapted parameters for that task
                # Accumulate the loss over all tasks in the meta-testing set
                meta_loss += self.meta_loss(y_pred, y)
                passes += 1
                # self.learner.zero_grad()
                # loss = self.meta_loss(y_pred, y)
                # loss.backward()

            # Restore initial parameters for learner
            with torch.no_grad():
                for name, param in self.learner.named_parameters():
                    param = initial_param[name]
            # TODO: Find a workaround for this problem:
            #       When the parameters are replaced, the gradient can't be
            #       computed with respect to the computed loss with the
            #       previous gradient values!
  #           with torch.no_grad():
                # for k, v in initial_param.items():
  #                   setattr(self.learner.net, k, torch.nn.Parameter(v))
#             with torch.no_grad():
                # for j, (k, v) in enumerate(initial_param):
                    # tree = k.split(".")
                    # prev = []
                    # attr = self.learner
                    # for t in tree:
                        # for p in prev:
                            # attr = getattr(attr, p)
                        # prev.append(t)
#                     setattr(attr, tree[-1], torch.nn.Parameter(v))

        # Step 3. Now that the meta-loss is computed
        avg_loss = meta_loss.item() / passes
        print(f"Meta-testing Average Loss={avg_loss}")
        # For now, let's try a manual parameter update
        meta_gradient = torch.autograd.grad(meta_loss,
                initial_param_raw)
        # assert None not in meta_gradient, "Empty meta-gradient!"
#         with torch.no_grad():
            # for j, param in enumerate(self.learner.parameters()):
#                 param -= self.meta_lr * meta_gradient[j]
#         for j, (k, v) in enumerate(initial_param.items()):
#             setattr(self.learner.net, k, torch.nn.Parameter(v - (self.meta_lr * meta_gradient[j])))
        # meta_loss.backward()
        # self.meta_optim.step()
        # TODO: What the heck do we return?


    def forward2(self, tasks_batch, return_loss=False):
        # m_train should never intersect with m_test! So only shuffle the task
        # at creation!
        # For each task in the batch
        meta_losses = []
        self.meta_opt.zero_grad()
        # t = TicToc()
        # t.tic()
        for i, task in enumerate(tasks_batch):
            with higher.innerloop_ctx(
                    self.learner, self.inner_opt, copy_initial_weights=False
                    ) as (f_learner, diff_opt):
                meta_loss = 0
                m_train, m_test = task[0], task[1]
                for s in range(self.inner_steps):
                    step_loss = 0
                    for x, y in m_train:
                        # m_train is an iterator returning batches
                        y_pred = f_learner(x.to(device))
                        step_loss += self.inner_loss(y_pred, y)
                    diff_opt.step(step_loss)

                for x, y in m_test:
                    y_pred = f_learner(x.to(device)) # Use the updated model for that task
                    # Accumulate the loss over all tasks in the meta-testing set
                    meta_loss += self.meta_loss(y_pred, y) / len(m_test)

                if return_loss:
                    meta_losses.append(meta_loss.detach())

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                meta_loss.backward()

        self.meta_opt.step()
        avg_loss = sum(meta_losses) / len(tasks_batch) if return_loss else 0
        # t.toc()
        return avg_loss


    def fit(self, dataset, tasks_per_iter: int, iterations: int):
        self.learner.train()
        # t = TicToc()
        # t.tic()
        for i in range(iterations):
            random.shuffle(dataset)
            loss = self.forward2(dataset[:tasks_per_iter], i%1000 == 0)
            if i % 1000 == 0:
                print(f"[{i}] Meta-testing Average Loss={loss}")
                # t.toc()
                # t.tic()
        # TODO: Meta-testing here?
        # TODO: Computation of the meta-objective here?
        # TODO: Meta-optimization here?

    def eval(self, dataset: List[tuple]):
        pass

