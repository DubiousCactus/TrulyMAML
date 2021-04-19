#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Constants file
"""

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
