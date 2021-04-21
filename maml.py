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

from tqdm.contrib import tenumerate
from pytictoc import TicToc
from copy import deepcopy
from typing import List
from tqdm import tqdm

from dataset import SineWaveDataset
from const import device


try: # otherwise it complains: context has already been set
  mp.set_start_method("spawn")
except: pass

def train_on_batch_on_task(rank, inner_steps, task, learner, inner_opt, optimizer, return_dict):
    meta_loss = 0
    optimizer.zero_grad()
    with higher.innerloop_ctx(
            learner, inner_opt, copy_initial_weights=False
            ) as (f_learner, diff_opt):
        sprt, qry = task[0], task[1]
        for s in range(inner_steps):
            step_loss = 0
            for x, y in sprt:
                # sprt is an iterator returning batches
                y_pred = f_learner(x)
                # TODO: Parametrize the loss function
                step_loss += F.mse_loss(y_pred, y)
            diff_opt.step(step_loss)

        for x, y in qry:
            y_pred = f_learner(x) # Use the updated model for that task
            # Accumulate the loss over all tasks in the meta-testing set
            meta_loss += F.mse_loss(y_pred, y) / (len(x)*len(qry))
    return_dict[rank] = meta_loss.detach()


class MAML(torch.nn.Module):
    def __init__(self, learner: torch.nn.Module, meta_lr=1e-4, inner_lr=1e-3, steps=1,
            loss_function=torch.nn.MSELoss(reduction='sum')):
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
        self.inner_loss = loss_function
        self.meta_loss = loss_function
        self._compute_accuracy = type(self.inner_loss) is torch.nn.CrossEntropyLoss

    def train_on_batch(self, tasks_batch):
        # sprt should never intersect with qry! So only shuffle the task
        # at creation!
        # For each task in the batch
        inner_losses, meta_losses, accuracies = [], [], []
        self.meta_opt.zero_grad()
        for i, task in enumerate(tasks_batch):
            with higher.innerloop_ctx(
                    self.learner, self.inner_opt, copy_initial_weights=False
                    ) as (f_learner, diff_opt):
                meta_loss, inner_loss, task_accuracies = 0, 0, []
                sprt, qry = task
                f_learner.train()
                for s in range(self.inner_steps):
                    step_loss = 0
                    for x, y in sprt:
                        # sprt is an iterator returning batches
                        y_pred = f_learner(x)
                        step_loss += self.inner_loss(y_pred, y)
                    inner_loss += step_loss.detach()
                    diff_opt.step(step_loss)

                f_learner.eval()
                for x, y in qry:
                    y_pred = f_learner(x) # Use the updated model for that task
                    # Accumulate the loss over all tasks in the meta-testing set
                    meta_loss += self.meta_loss(y_pred, y)
                    if self._compute_accuracy:
                        scores, indices = y_pred.max(dim=1)
                        acc = (y == indices).sum() / y.size(0) # Mean accuracy per batch
                        task_accuracies.append(acc)

                # Divide by the number of samples because reduction is set to 'sum' so that
                # the meta-objective can be computed correctly.
                meta_losses.append(meta_loss.detach().div_(self.inner_steps*len(sprt.dataset)))
                inner_losses.append(inner_loss.mean().div_(len(qry.dataset)))
                if self._compute_accuracy:
                    accuracies.append(torch.tensor(task_accuracies).mean())

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                meta_loss.backward()

        self.meta_opt.step()
        avg_inner_loss = torch.tensor(inner_losses).mean().item()
        avg_meta_loss = torch.tensor(meta_losses).mean().item()
        avg_accuracy = torch.tensor(accuracies).mean().item()
        return avg_inner_loss, avg_meta_loss, avg_accuracy

    def eval_task_batch(self, task_batch):
        '''
        Use the Higher innerloop context to evaluate a task batch.
        Not suited for inference, only for evaluation.
        '''
        batch_loss = [] # Average loss of the batch of tasks
        for task in task_batch:
            with higher.innerloop_ctx(self.learner, self.inner_opt) as (f_learner, diff_opt):
                qry_loss = 0
                sprt, qry = task
                f_learner.train()
                for s in range(self.inner_steps):
                    step_loss = 0
                    for x, y in sprt:
                        y_pred = f_learner(x)
                        step_loss += self.inner_loss(y_pred, y)
                    diff_opt.step(step_loss)

                f_learner.eval()
                for x, y in qry:
                    y_pred = f_learner(x)
                    qry_loss += self.inner_loss(y_pred, y)
                batch_loss.append(qry_loss.detach().div_(len(qry.dataset)))
        return torch.tensor(batch_loss).mean()

    def adapt(self, task_support):
        '''
        Adapt the model to the task using the support set. This is typically used for inference on
        a novel task.
        '''
        self.learner.train()
        for s in range(self.inner_steps):
            self.inner_opt.zero_grad()
            # step_loss = 0
            for x, y in task_support:
                y_pred = self.learner(x)
                loss = self.inner_loss(y_pred, y)
                loss.backward()
            # step_loss.backward()
            self.inner_opt.step()

    def train_on_batch_mp(self, tasks_batch, return_loss=False):
        # sprt should never intersect with qry! So only shuffle the task
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
            p = mp.Process(target=train_on_batch_on_task,
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

    def fit(self, dataset, iterations: int, save_path: str, epoch: int, epochs_per_avg=1000):
        self.learner.train()
        # t = TicToc()
        # t.tic()
        try:
            os.makedirs(save_path)
        except Exception:
            pass
        global_avg_inner_loss, global_avg_meta_loss, global_avg_accuracy, best_ckpt = 0, 0, 0, 1000
        for i in range(epoch, iterations):
            inner_loss, meta_loss, accuracy = self.train_on_batch(next(dataset))
            global_avg_inner_loss += inner_loss
            global_avg_meta_loss += meta_loss
            global_avg_accuracy += accuracy
            if i % epochs_per_avg == 0:
                if i != 0:
                    global_avg_inner_loss /= epochs_per_avg
                    global_avg_meta_loss /= epochs_per_avg
                    global_avg_accuracy /= epochs_per_avg
                print(f"[{i}] Avg Inner Loss={global_avg_inner_loss} - Avg Outer Loss={global_avg_meta_loss} - Avg Outer Accuracy={accuracy:.2f} (over {epochs_per_avg} epochs) - Last Outer loss={meta_loss}")
                if global_avg_meta_loss < best_ckpt:
                    torch.save({
                        'epoch': i,
                        'model_state_dict': self.learner.state_dict(),
                        'inner_opt_state_dict': self.inner_opt.state_dict(),
                        'meta_opt_state_dict': self.meta_opt.state_dict(),
                        'inner_loss': self.inner_loss,
                        'meta_loss': self.meta_loss
                        }, os.path.join(save_path,
                            f"epoch_{i}_loss-{global_avg_meta_loss:.6f}_accuracy-{global_avg_accuracy:.2f}.tar"))
                    best_ckpt = global_avg_meta_loss
                global_avg_inner_loss = 0
                global_avg_meta_loss = 0
                global_avg_accuracy = 0
                # t.toc()
                # t.tic()

    # def eval_with_higher(self, dataset):
        # total_loss, batch_size, avg_batch_loss = 0, 32, 0
        # batch_count = len(dataset)//batch_size + 1
        # for i in tqdm(range(batch_count)):
            # start = i*batch_size
            # end = min(len(dataset), start + batch_size)
            # avg_batch_loss += self.eval_task_batch(dataset[start:end])
    #     print(f"Total average loss: {avg_batch_loss/batch_count}")

    def eval(self, dataset, compute_accuracy=False):
        def fit_and_test(task, state_dict, comp_acc=False):
            # Restore the model parameters
            self.learner.load_state_dict(state_dict)
            self.adapt(task[0])
            task_loss, task_accuracies = 0, [] # Average loss per point in the task
            with torch.no_grad():
                self.learner.eval()
                for x, y in task[1]:
                    y_pred = self.learner(x)
                    task_loss += self.inner_loss(y_pred, y)
                    if comp_acc:
                        _, indices = y_pred.max(dim=1)
                        mean_acc = (y == indices).sum() / y.size(0)
                        task_accuracies.append(mean_acc) # Mean accuracy per batch
             # Average per item because the inner loss reduction is 'sum'
            return task_loss.div_(len(task[1].dataset)), torch.tensor(task_accuracies).mean()

        total_loss, total_acc = [], []
        # Save the model parameters
        state_dict = deepcopy(self.learner.state_dict())
        t = tqdm(total=len(dataset))
        for i, batch in enumerate(dataset):
            if not batch:
                t.close()
                break
            batch_loss, batch_accuracy = [], []
            for task in batch:
                loss, acc = fit_and_test(task, state_dict, comp_acc=compute_accuracy)
                batch_loss.append(loss)
                batch_accuracy.append(acc)
            total_loss.append(torch.tensor(batch_loss).mean())
            total_acc.append(torch.tensor(batch_accuracy).mean())
            t.update(len(batch))
        total_loss = torch.tensor(total_loss).mean().item()
        total_acc = torch.tensor(total_acc).mean().item() if compute_accuracy else .0
        print(f"Total average loss: {total_loss} - Total average accuracy: {total_acc:.4f}")

    def restore(self, checkpoint, resume_training=True):
        self.learner.load_state_dict(checkpoint['model_state_dict'])
        self.meta_opt.load_state_dict(checkpoint['meta_opt_state_dict'])
        self.meta_loss = checkpoint['meta_loss']
        if resume_training:
            self.inner_opt.load_state_dict(checkpoint['inner_opt_state_dict'])
            self.inner_loss = checkpoint['inner_loss']
