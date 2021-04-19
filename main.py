#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import argparse
import random
import torch
import sys
import os

from dataset import SineWaveDataset, OmniglotDataset, HarmonicDataset, SinusoidAndLineDataset
from learner import DummiePolyLearner, MLP, ConvNetClassifier
from const import device
from maml import MAML

from typing import List, Tuple
from pytictoc import TicToc



# TODO:
# [x] Save model state
# [x] Restore model state
# [x] Implement meta-testing (model evaluation)
# [x] Implement OmniGlot classification
# [ ] Dataset factory
# [ ] Clip the gradients to prevent NaN loss!
# [ ] Normalize OmniGlot
# [ ] Implement data generator for sine waves, and use it for on-the-fly batch generation
# [ ] Implement multiprocessing if possible (https://discuss.pytorch.org/t/multiprocessing-with-tensors-requires-grad/87475/2)


def train_with_maml(dataset, learner, save_path: str, steps: int,
        meta_batch_size: int, iterations: int, checkpoint=None, loss_fn=None):
    print("[*] Training...")
    model = MAML(learner, steps=steps, loss_function=loss_fn)
    model.to(device)
    epoch = 0
    if checkpoint:
        model.restore(checkpoint)
        epoch = checkpoint['epoch']
    model.fit(dataset, iterations, save_path, epoch, 1000)
    print("[*] Done!")
    return model


def test_with_maml(dataset, learner, checkpoint, steps, loss_fn):
    print("[*] Testing...")
    model = MAML(learner, steps=steps, loss_function=loss_fn)
    model.to(device)
    if checkpoint:
        model.restore(checkpoint, resume_training=False)
    else:
        print("[!] You are running inference on a randomly initialized model!")
    model.eval(dataset)
    print("[*] Done!")


def conventional_training(dataset, learner):
    print("[*] Training with a conventional optimizer...")
    # Make the training / eval splits
    model = learner
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss(reduction='sum')
    train_dataset, eval_dataset = dataset[0], dataset[0][1]
    print("[*] Evaluating with random initialization...")
    total_loss = 0
    for i, (x, y) in enumerate(eval_dataset):
        y_pred = model(x.to(device))
        loss = criterion(y_pred, y.to(device))
        # print(f"-> Batch {i}: {loss}")
        total_loss += loss
    avg_loss = total_loss.item() / len(eval_dataset)

    print(f"[*] Average evaluation loss: {avg_loss}")

    t = TicToc()
    model.train()
    t.tic()
    for i in range(2000):
        loss = 0
        optimizer.zero_grad()
        for x, y in train_dataset:
            y_pred = model(x)
            loss += criterion(y_pred, y)
        if i % 100 == 99:
            print(i, loss.item()/len(train_dataset))
            t.toc()
            t.tic()
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


def parse_args():
    parser = argparse.ArgumentParser(description="Model-Agnostic Meta-Learning")
    parser.add_argument('--checkpoint_path', type=str, help='''path to checkpoint
            saving directory''', default='ckpt')
    parser.add_argument('--load', type=str, help='''path to model
            checkpoint''')
    parser.add_argument('--eval', action='store_true', help='''Evaluation
    moed''')
    parser.add_argument('--samples', type=int, default=25, help='''Number of
    samples per task. The resulting number of test samples will be this value
    minus <K>.''')
    parser.add_argument('-k', type=int, default=10, help='''Number of shots
    for meta-training''')
    parser.add_argument('-q', type=int, default=15, help='''Number of
    meta-testing samples''')
    parser.add_argument('-n', type=int, default=5, help='''Number of classes for n-way
    classification''')
    parser.add_argument('-s', type=int, default=1, help='''Number of inner loop
    optimization steps during meta-training''')
    parser.add_argument('--dataset', choices=['omniglot', 'sinusoid', 'harmonic'])
    parser.add_argument('--meta-batch-size', type=int, default=25, help='''Number
    of tasks per meta-update''')
    parser.add_argument('--iterations', type=int, default=80000, help='''Number
    of outer-loop iterations''')
    return parser.parse_args()



def main():
    args = parse_args()
    # np.random.seed(5)

    learner = ConvNetClassifier(device, 1, 20) if args.dataset == "omniglot" else MLP(device)
    checkpoint = None
    if args.load:
        checkpoint = torch.load(args.load)
    learner.to(device)
    # TODO: Factory for the Dataset
    if args.eval:
        test_dataset = (SineWaveDataset(1000, args.samples, args.k,
            args.q, args.meta_batch_size) if args.dataset == 'sinusoid' else
            OmniglotDataset(args.meta_batch_size, 28, args.k, args.q, args.n, evaluation=True))
        test_with_maml(test_dataset, learner, checkpoint, args.s, torch.nn.MSELoss(reduction='sum')
                if args.dataset == "sinusoid" else torch.nn.CrossEntropyLoss(reduction='sum'))
    else:
        train_dataset = (SineWaveDataset(100000, args.samples, args.k,
                args.q, args.meta_batch_size) if args.dataset == 'sinusoid' else
                OmniglotDataset(args.meta_batch_size, 28, args.k, args.q, args.n, evaluation=False))
        # train_dataset = SineWaveDataset(1000, args.samples, args.k, args.q, args.meta_batch_size)
        # conventional_train(dataset, learner)
        train_with_maml(train_dataset, learner,
                args.checkpoint_path, args.s, args.meta_batch_size,
                args.iterations, checkpoint, torch.nn.CrossEntropyLoss(reduction='sum') if
                args.dataset == 'omniglot' else torch.nn.MSELoss(reduction='sum'))


if __name__ == "__main__":
    main()
