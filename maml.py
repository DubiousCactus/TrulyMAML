#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
MAML module
"""

import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import random
import higher
import torch
import math
import os

from pytictoc import TicToc
from typing import List
from tqdm import tqdm

from const import device


try: # otherwise it complains: context has already been set
  mp.set_start_method("spawn")
except: pass

def forward_on_task(rank, inner_steps, task, learner, inner_opt, optimizer, return_dict):
    meta_loss = 0
    optimizer.zero_grad()
    with higher.innerloop_ctx(
            learner, inner_opt, copy_initial_weights=False
            ) as (f_learner, diff_opt):
        m_train, m_test = task[0], task[1]
        for s in range(inner_steps):
            step_loss = 0
            for x, y in m_train:
                # m_train is an iterator returning batches
                y_pred = f_learner(x)
                step_loss += F.mse_loss(y_pred, y)
            diff_opt.step(step_loss)

        for x, y in m_test:
            y_pred = f_learner(x) # Use the updated model for that task
            # Accumulate the loss over all tasks in the meta-testing set
            meta_loss += F.mse_loss(y_pred, y) / (len(x)*len(m_test))
    return_dict[rank] = meta_loss.detach()


class MAML(torch.nn.Module):
    def __init__(self, learner: torch.nn.Module,
            meta_lr=1e-3, inner_lr=1e-3, steps=1):
        super().__init__()
        self.meta_lr = meta_lr # This term is beta in the paper
        # TODO: Make the inner learning rate optionally learnable
        self.inner_lr = inner_lr # This term is alpha in the paper
        self.learner = learner
        self.inner_steps = steps
        self.meta_opt = torch.optim.Adam(self.learner.parameters(),
                lr=self.meta_lr)
        self.inner_opt = torch.optim.SGD(self.learner.parameters(),
                lr=self.inner_lr)
        self.inner_loss = torch.nn.MSELoss(reduction='sum')
        self.meta_loss = torch.nn.MSELoss(reduction='sum')


    def forward(self, tasks_batch, return_loss=False):
        # m_train should never intersect with m_test! So only shuffle the task
        # at creation!
        # For each task in the batch
        inner_losses, meta_losses = [], []
        self.meta_opt.zero_grad()
        # t = TicToc()
        # t.tic()
        # TODO: Adapt for OmniGlot
        for i, task in enumerate(tasks_batch):
            with higher.innerloop_ctx(
                    self.learner, self.inner_opt, copy_initial_weights=False
                    ) as (f_learner, diff_opt):
                meta_loss, inner_loss = 0, 0
                m_train, m_test = task[0], task[1]
                f_learner.train()
                for s in range(self.inner_steps):
                    step_loss, batch_len = 0, 1
                    for x, y in m_train:
                        # m_train is an iterator returning batches
                        y_pred = f_learner(x)
                        step_loss += self.inner_loss(y_pred, y)
                        batch_len = len(x)
                    inner_loss += step_loss.detach() / batch_len
                    diff_opt.step(step_loss)

                f_learner.eval()
                for x, y in m_test:
                    y_pred = f_learner(x) # Use the updated model for that task
                    # Accumulate the loss over all tasks in the meta-testing set
                    meta_loss += self.meta_loss(y_pred, y) / len(x)

                if return_loss:
                    meta_losses.append(meta_loss.detach()/len(m_test))
                    inner_losses.append(inner_loss/(self.inner_steps*len(m_train)))

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                meta_loss.backward()

        self.meta_opt.step()
        avg_inner_loss = sum(inner_losses) / len(tasks_batch) if return_loss else 0
        avg_meta_loss = sum(meta_losses) / len(tasks_batch) if return_loss else 0
        # t.toc()
        return avg_inner_loss, avg_meta_loss


    def forward_mp(self, tasks_batch, return_loss=False):
        # m_train should never intersect with m_test! So only shuffle the task
        # at creation!
        # For each task in the batch
        '''
        See https://discuss.pytorch.org/t/multiprocessing-with-tensors-requires-grad/87475/2
        '''
        torch.manual_seed(42)
        self.learner.share_memory()
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()
        for rank in range(len(tasks_batch)):
            p = mp.Process(target=forward_on_task,
                    args=(rank, self.inner_steps, tasks_batch[rank],
                        self.learner, self.inner_opt, self.meta_opt,
                        return_dict))
            # We first train the model across `num_processes` processes
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # Update the model's meta-parameters to optimize the query
        # losses across all of the tasks sampled in this batch.
        # This unrolls through the gradient steps.
        assert return_dict, "Empty meta-loss list"
        total_meta_loss = sum(return_dict.values())
        print(total_meta_loss)
        total_meta_loss.backward()

        self.meta_opt.step()
        return total_meta_loss / len(tasks_batch) if return_loss else 0


    def fit(self, dataset, tasks_per_iter: int, iterations: int,
            save_path: str, epoch: int):
        self.learner.train()
        # t = TicToc()
        # t.tic()
        try:
            os.makedirs(save_path)
        except Exception:
            pass
        for i in range(epoch, iterations):
            if type(dataset) == list:
                random.shuffle(dataset)
                inner_loss, meta_loss = self.forward(dataset[:tasks_per_iter], i%1000 == 0)
            else:
                inner_loss, meta_loss = self.forward(dataset, i%1000 == 0)
            if i % 1000 == 0:
                print(f"[{i}] Avg Inner Loss={inner_loss} - Avg Meta-testing Loss={meta_loss}")
                torch.save({
                    'epoch': i,
                    'model_state_dict': self.learner.state_dict(),
                    'inner_opt_state_dict': self.inner_opt.state_dict(),
                    'meta_opt_state_dict': self.meta_opt.state_dict(),
                    'inner_loss': self.inner_loss,
                    'meta_loss': self.meta_loss
                    }, os.path.join(save_path, f"epoch_{i}_loss-{meta_loss}.tar"))
                # t.toc()
                # t.tic()


    def eval(self, dataset: List[tuple]):
        self.learner.train()
        total_loss = 0
        for i, task in tqdm(enumerate(dataset)):
            inner_loss, meta_loss = self.forward([task], True)
            # print(f"[Task {i}] Inner Loss={inner_loss} - Meta-testing Loss={meta_loss}")
            total_loss += meta_loss
        print(f"Total average loss: {total_loss/len(dataset)}")


    def restore(self, checkpoint):
        self.learner.load_state_dict(checkpoint['model_state_dict'])
        self.inner_opt.load_state_dict(checkpoint['inner_opt_state_dict'])
        self.meta_opt.load_state_dict(checkpoint['meta_opt_state_dict'])
        self.meta_loss = checkpoint['meta_loss']
        self.inner_loss = checkpoint['inner_loss']
