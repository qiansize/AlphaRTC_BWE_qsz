#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import matplotlib.pyplot as plt
import utils_gcc
import utils_ppo
from rtc_env_ppo_gcc import GymEnv
from deep_rl.storage import Storage
from deep_rl.ppo_agent import PPO
import torch
import os
import gym
import datetime
import time
import logging
from utils_ppo import load_config

import torch.multiprocessing as mp
import numpy as np
from ActorCritic import ActorCritic
from rtc_env_ppo_gcc_save import GymEnv


def main():
    ############## Hyperparameters for the experiments ##############
    env_name = "AlphaRTC"
    max_num_episodes = 1001     # maximal episodes

    update_interval = 4000     # update policy every update_interval timesteps
    save_interval = 20         # save model every save_interval episode
    exploration_param = 0.05    # the std var of action distribution
    K_epochs = 37               # update policy for K_epochs
    ppo_clip = 0.2              # clip parameter of PPO
    gamma = 0.99                # discount factor

    lr = 3e-5                 # Adam parameters
    betas = (0.9, 0.999)
    state_dim = 4
    state_length=10
    action_dim = 1
    data_path = './data/' # Save model and reward curve here
    #############################################

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    config=load_config()
    env = GymEnv(config=config)
    storage = Storage() # used for storing data
    ppo = PPO(state_dim, state_length,action_dim, exploration_param, lr, betas, gamma, K_epochs, ppo_clip)
    # ppo.policy.load_state_dict(torch.load('data/ppo_2021_07_07_18_59_43.pth'))
    # ppo.policy_old.load_state_dict(torch.load('data/ppo_2021_07_07_18_59_43.pth'))
    record_episode_reward = []
    episode_reward  = 0
    time_step = 0
    time_to_guide= False
    # training loop
    # for episode in range(max_num_episodes):
    #     while time_step < update_interval:
    #         done = False
    #         state = torch.Tensor(env.reset()) #state tensor 4*10
    #         last_estimation = 300000
    #         action = 0
    #         print("still working on")
    #         if len(storage.is_terminals) != 0:
    #             storage.is_terminals[-1] = True
    #         while not done and time_step < update_interval:
    #             if time_step % 6 == 5:
    #                 action = ppo.select_action(state, storage)
    #                 time_to_guide=True
    #
    #             state, reward, done, last_estimation,delay,loss= env.step(action, last_estimation, time_to_guide)
    #             time_to_guide= False
    #             state = torch.Tensor(state)
    #             # Collect data for update
    #             if time_step % 6 == 5:
    #                 storage.rewards.append(reward)
    #                 storage.is_terminals.append(done)
    #             time_step += 1
    #             episode_reward += reward
    #
    #     storage.is_terminals[-1] = True
    #     next_value = ppo.get_value(state)
    #     storage.compute_returns(next_value, gamma)
    #
    #     # update
    #     policy_loss, val_loss = ppo.update(storage, state)
    #     storage.clear_storage()
    #     episode_reward /= time_step
    #     record_episode_reward.append(episode_reward)
    #     print('Episode {} \t Average policy loss, value loss, reward {}, {}, {}'.format(episode, policy_loss, val_loss, episode_reward))
    #
    #     if episode > 0 and not (episode % save_interval):
    #         ppo.save_model(data_path)
    #         plt.plot(range(len(record_episode_reward)), record_episode_reward)
    #         plt.xlabel('Episode')
    #         plt.ylabel('Averaged episode reward')
    #         plt.savefig('%sreward_record.jpg' % (data_path))
    #
    #     episode_reward = 0
    #     time_step = 0

    ppo.policy.load_state_dict(torch.load('data/ppo_2021_07_07_20_43_24.pth'))
    utils_ppo.draw_module(config, ppo.policy, data_path)


if __name__ == '__main__':
    main()
