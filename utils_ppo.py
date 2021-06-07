import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from rtc_env_ppo import GymEnv
from deep_rl.storage import Storage
from deep_rl.actor_critic import ActorCritic
import rtc_env_ppo

UNIT_M = 1000000
MAX_BANDWIDTH_MBPS = 8
MIN_BANDWIDTH_MBPS = 0.01
HISTORY_LENGTH = 10
STATE_DIMENSION = 4
LOG_MAX_BANDWIDTH_MBPS = np.log(MAX_BANDWIDTH_MBPS)
LOG_MIN_BANDWIDTH_MBPS = np.log(MIN_BANDWIDTH_MBPS)


def liner_to_log(value):
    # from 10kbps~8Mbps to 0~1
    value = np.clip(value / UNIT_M, MIN_BANDWIDTH_MBPS, MAX_BANDWIDTH_MBPS)
    log_value = np.log(value)
    return (log_value - LOG_MIN_BANDWIDTH_MBPS) / (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS)


def log_to_linear(value):
    # from 0~1 to 10kbps to 8Mbps
    value = np.clip(value, 0, 1)
    log_bwe = value * (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS) + LOG_MIN_BANDWIDTH_MBPS
    return np.exp(log_bwe) * UNIT_M


def load_config():
    config = {
        #todo: add parameters regarding configuration
        'actor_learning_rate': 0.01,
        'critic_learning_rate': 0.001,
        'num_agents': 16,
        'save_interval': 20,

        'default_bwe': 2,
        'train_seq_length': 1000,
        'state_dim': 3,
        'state_length': 10,
        'action_dim': 1,
        'device': 'cpu',
        'discount_factor': 0.99,
        'load_model': False,
        'saved_actor_model_path': '',
        'saved_critic_model_path': '',
        'layer1_shape': 128,
        'layer2_shape': 128,

        'sending_rate': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        'entropy_weight': 0.5,

        'trace_dir': './traces',
        'log_dir': './logs',
        'model_dir': './models'
    }

    return config


def draw_state(record_action, record_state, path):
    length = len(record_action)

    plt.plot(range(length), record_action)
    plt.xlabel('episode')
    plt.ylabel('action')
    plt.ylim((0,1000000))
    # ylabel = ['receiving rate', 'delay', 'packet loss']
    # record_state = [t.numpy() for t in record_state]
    # record_state = np.array(record_state)
    # for i in range(3):
    #     plt.subplot(411+i+1)
    #     plt.plot(range(length), record_state[i])
    #     plt.xlabel('episode')
    #     plt.ylabel(ylabel[i])
    plt.tight_layout()
    plt.savefig("{}test_result.jpg".format(path))


def draw_module(config,model, data_path, max_num_steps = 1000):
    env = GymEnv(config=config)
    record_reward = []
    record_state = []
    record_action = []
    episode_reward  = 0
    time_step = 0
    tmp = model.random_action
    model.random_action = False
    while time_step < max_num_steps:
        done = False
        state = torch.Tensor(env.reset())
        while not done:
            action, _, _, _ = model.forward(state)
            state, reward, done, _ = env.step(action)
            state = torch.Tensor(state)
            #record_state.append(state)
            #record_reward.append(reward)
            record_action.append(log_to_linear(action))
            time_step += 1
    model.random_action = True
    draw_state(record_action, record_state, data_path)