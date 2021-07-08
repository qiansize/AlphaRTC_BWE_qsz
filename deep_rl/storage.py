#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch


class Storage:
    def __init__(self):
        self.actions = []
        self.values = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.returns = []
        self.switch_interval = 1000
        self.counter = -1

    def compute_returns(self, next_value, gamma):
        # compute returns for advantages
        returns_tmp = []
        returns = np.zeros(len(self.rewards)+1)
        returns[-1] = next_value
        for i in range(len(self.rewards) - 1, self.counter, -1):
            returns[i] = returns[i+1] * gamma * (1-self.is_terminals[i]) + self.rewards[i]
            returns_tmp.append(torch.tensor([returns[i]]))
            self.counter += 1
        returns_tmp.reverse()
        for element in returns_tmp:
            self.returns.append(element)

    def clear_storage(self):
        self.actions.clear()
        self.values.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.returns.clear()
        self.counter = -1
