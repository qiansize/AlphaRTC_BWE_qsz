import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from rtc_env_ppo_gcc import GymEnv
from deep_rl.storage import Storage
from deep_rl.actor_critic_cnn import ActorCritic
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
        'state_dim': 4,
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


def draw_state(record_action, record_delay, record_loss ,trace_y, path):
    length1 = len(record_action)

    length2= len(trace_y)
    plt.figure(1)
    plt.plot(range(length1), record_action, range(length2), trace_y)
    plt.xlabel('step')
    plt.ylabel('action')
    plt.ylim((0,2000000))
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
    plt.figure(2)
    plt.plot(range(length1),record_delay)
    plt.xlabel('step')
    plt.ylabel('delay')
    plt.savefig("{}test_result_RL_delay.jpg".format(path))


def draw_trace(trace_path):
    with open(trace_path, "r") as trace_file:
        duration_list = []
        capacity_list = []

        load_dict = json.load(trace_file)
        uplink_info = load_dict["uplink"]["trace_pattern"]
        for info in uplink_info:
            duration_list.append(info["duration"])
            capacity_list.append(info["capacity"] * 1000)
        print(duration_list)
        print(capacity_list)
        # duration_sum = sum(duration_list)
        t = 0
        x = []
        y = []
        for i in range(len(duration_list)):
            x_tmp = np.arange(t, t + duration_list[i], 1)
            for element in x_tmp:
                x.append(element)
                y.append(capacity_list[i])
            t += duration_list[i]
        plt.plot(x, y)
        plt.show()


def draw_module(config,model, data_path, max_num_steps = 1000):
    env = GymEnv(config=config)
    record_reward = []
    record_state = []
    record_action = []
    record_delay=[]
    record_loss=[]
    episode_reward  = 0
    time_step = 0

    tmp = model.random_action
    model.random_action = False
    time_to_guide = False

    done = False
    state = torch.Tensor(env.reset())
    trace_path = 'traces/Serial_268629871.json'
    last_estimation=300000
    action = 0
    while not done and time_step<=1000:
        if time_step % 6 == 5:
            action, _, _, _ = model.forward(state)
            time_to_guide = True
            print("action", pow(2,(action*2-1)))
        state, reward, done, last_estimation, delay, loss = env.step(action, last_estimation, time_to_guide)
        time_to_guide = False
        state = torch.Tensor(state)
        #record_state.append(state)
        #record_reward.append(reward)
        real_estimation=last_estimation
        record_action.append(real_estimation)
        record_delay.append(delay)
        record_loss.append(loss)
        print("real", real_estimation)

        time_step += 1
    model.random_action = True
    with open(trace_path, "r") as trace_file:
        duration_list = []
        capacity_list = []
        time_list=[]
        load_dict = json.load(trace_file)
        uplink_info = load_dict["uplink"]["trace_pattern"]
        for info in uplink_info:
            duration_list.append(info["duration"])
            capacity_list.append(info["capacity"] * 1000)
            time_list.append(sum(duration_list))
        print(duration_list)
        print(capacity_list)
        print(time_list)
        # duration_sum = sum(duration_list)
        t = 0
        trace_x = []
        trace_y = []

        i=0
        for a in range(1000):
            if t <= time_list[i]:
                trace_y.append(capacity_list[i])
            else:
                i+=1
                trace_y.append(capacity_list[i])
            t += 200
        # plt.plot(trace_x, trace_y)
        # plt.ylim((0, 2000000))
        # plt.savefig("{}test_result_gcc.jpg".format(path))
    draw_state(record_action, record_delay,record_loss, trace_y, path=data_path)