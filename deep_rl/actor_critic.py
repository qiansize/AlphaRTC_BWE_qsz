#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, state_length, action_dim, exploration_param=0.05, device="cpu"):
        super(ActorCritic, self).__init__()
        # output of actor in [0, 1]
        # self.actor =  nn.Sequential(
        #         nn.Linear(state_dim, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, action_dim),
        #         nn.Sigmoid()
        #         )
        # # critic
        # self.critic = nn.Sequential(
        #         nn.Linear(state_dim, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 64),
        #         nn.ReLU(),
        #         nn.Linear(64,32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1)
        #         )
        self.layer1_shape = 128
        self.layer2_shape = 128
        self.numFcInput = 4096

        self.layer1_shape_lstm = 256
        self.layer2_shape_lstm = 256
        self.numFcInput_lstm = 2560

        # LSTM的网络结构
        # self.rLSTM = nn.LSTM(1, self.layer1_shape_lstm, 2)
        # self.dLSTM = nn.LSTM(1, self.layer1_shape_lstm, 2)
        # self.lLSTM = nn.LSTM(1, self.layer1_shape_lstm, 2)
        # self.pLSTM = nn.LSTM(1, self.layer1_shape_lstm, 2)

        self.lstm = nn.LSTM(4, self.layer1_shape_lstm, 2)
        self.lstm_crtic = nn.LSTM(4, self.layer1_shape_lstm, 2)
        self.fc1 = nn.Linear(self.numFcInput_lstm, self.layer2_shape)
        self.fc2 = nn.Linear(self.layer2_shape, self.layer2_shape)
        self.fc3 = nn.Linear(self.numFcInput_lstm, self.layer2_shape)

        self.actor_output = nn.Linear(self.layer2_shape, action_dim)
        self.critic_output = nn.Linear(self.layer2_shape, 1)
        self.device = device
        self.action_var = torch.full((action_dim,), exploration_param ** 2).to(self.device)
        self.random_action = True

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data, 0.1)
            elif isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data, 0.1)
            elif isinstance(m, nn.LSTM):
                init.xavier_uniform_(m.weight.data)

    def forward(self, inputs):
        # actor
        lstm_out, _ = self.lstm(inputs[:1, :, :].view(10, 1, 4))
        # print("inputs.shape = " + str(inputs.shape))
        # print("lstm_out.shape = "+str(lstm_out.shape))
        fcOut_tmp = F.relu(self.fc1(lstm_out.view(1, -1)), inplace=True)
        fcOut = F.relu(self.fc2(fcOut_tmp), inplace=True)
        action_mean = torch.sigmoid(self.actor_output(fcOut))

        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        if not self.random_action:
            action = action_mean
        else:
            action = dist.sample()
        action_logprobs = dist.log_prob(action)
        # critic
        lstm_out, _ = self.lstm_crtic(inputs[:1, :, :].view(10, 1, 4))

        fcOut_critic = F.relu(self.fc3(lstm_out.view(1, -1)), inplace=True)
        value = self.critic_output(fcOut_critic)

        return action.detach(), action_logprobs, value, action_mean

    def evaluate(self, state, action):
        _, _, value, action_mean = self.forward(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(value), dist_entropy
