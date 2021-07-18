import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from rtc_env_ppo_gcc import GymEnv
from rtc_env_pure_RL import RLGymEnv
from deep_rl.storage import Storage
from deep_rl.actor_critic_cnn import ActorCritic
import rtc_env_ppo
import numpy
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


def draw_state(record_action, record_delay, record_action_gcc,record_delay_gcc ,record_purerl_action,record_purerl_delay,trace_y, path):
    length1 = len(record_action)

    length2= len(trace_y)
    length3= len(record_action_gcc)
    length4=len(record_purerl_action)
    plt.figure(1,figsize=(12,4))
    plt.plot(np.arange(0,length1/5,0.2), record_action, label='Navigator')

    plt.plot(np.arange(0,length3/5,0.2),record_action_gcc,label='GCC')
    plt.plot(np.arange(0,length4/5,0.2),record_purerl_action,label='Pure-RL')
    plt.plot(np.arange(0, length2 / 5, 0.2), trace_y, label='Capacity')
    plt.legend()
    plt.xlabel('time(s)')
    plt.ylabel('sending rate(bps)')
    # plt.ylim((0,2000000))
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
    plt.figure(2,figsize=(12,4))
    plt.plot(np.arange(0,length1/5,0.2),record_delay, label='Navigator')
    plt.plot(np.arange(0, length1 / 5, 0.2), record_delay_gcc,label='GCC')
    plt.plot(np.arange(0, length1 / 5, 0.2), record_purerl_delay,label='Pure-RL')
    plt.xlabel('time(s)')
    plt.ylabel('delay(ms)')
    plt.tight_layout()
    plt.legend()
    plt.savefig("{}a_test_result_delay.jpg".format(path))

def draw_stationary(path,delay_list_RL,average_cap_list_RL,delay_list_gcc,average_cap_list_gcc,delay_list_pureRL,average_cap_list_pureRL):

    plt.scatter( delay_list_RL,np.array(average_cap_list_RL)*100, label='Navigator',marker="s")
    plt.scatter(delay_list_gcc,np.array(average_cap_list_gcc)*100,  label='GCC',marker="o")
    plt.scatter( delay_list_pureRL,np.array(average_cap_list_pureRL)*100, label='Pure-RL',marker="v")
    plt.xlabel('delay(ms)')
    plt.ylabel('bandwidth_utilization')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.legend()
    plt.savefig("{}a_test_result_station.jpg".format(path))

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


def draw_module(config,model, model2,data_path, max_num_steps = 1000):
    env = GymEnv(config=config)
    rlenv= RLGymEnv(config=config)
    record_reward = []
    record_state = []

    episode_reward  = 0
    model2.random_action = False
    trace_list = os.listdir('traces')
    tmp = model.random_action
    model.random_action = False
    time_to_guide = False
    average_delay_list_RL=[]
    p50_delay_list_RL=[]
    p95_delay_list_RL=[]
    band_uil_list_RL=[]
    loss_RL=[]
    average_delay_list_gcc=[]
    p50_delay_list_gcc = []
    p95_delay_list_gcc = []
    band_uil_list_gcc = []
    loss_gcc=[]
    average_delay_list_purerl = []
    p50_delay_list_purerl = []
    p95_delay_list_purerl = []
    band_uil_list_purerl = []
    loss_purerl=[]
    var_list = []
    ave_cap_list = []

    qoe_list_RL=[]
    qoe_list_gcc=[]
    qoe_list_pureRL=[]
    # for i in range(2):
    for i in range(len(trace_list)):
        #for RL

        #432,436
        state = torch.Tensor(env.reset(trace_list[i]))
        trace_path = 'traces/'+trace_list[i]
        last_estimation=300000
        action = 0
        done = False
        model.random_action = False
        time_to_guide = False
        time_step = 0
        record_action_RL = []
        record_delay_RL = []
        record_loss_RL = []

        while not done and time_step<=1000:
            if time_step % 6 == 5:
                action, _, _, _ = model.forward(state)
                time_to_guide = True
                #print("action", pow(2,(action*2-1)))
            state, reward, done, last_estimation, delay, loss ,receiving_rate= env.step(action, last_estimation, time_to_guide)

            time_to_guide = False
            state = torch.Tensor(state)
            #record_state.append(state)
            #record_reward.append(reward)
            real_estimation=last_estimation
            record_action_RL.append(real_estimation)
            record_delay_RL.append(delay)
            record_loss_RL.append(loss)
            # print("real", real_estimation)

            time_step += 1

        average_delay_list_RL.append(np.mean(record_delay_RL))
        delay_sort_RL=record_delay_RL[:]
        delay_sort_RL.sort()
        p50_delay_list_RL.append(delay_sort_RL[500])
        p95_delay_list_RL.append(delay_sort_RL[950])
        loss_RL.append(np.mean(record_loss_RL))
        max_delay_RL=max(record_delay_RL)
        min_delay_RL=min(record_delay_RL)
        delay_score_RL=(max_delay_RL-delay_sort_RL[950])/(max_delay_RL-min_delay_RL/2)
        loss_score_RL=(1-np.mean(record_loss_RL))
        #for gcc
        record_action_gcc = []
        record_delay_gcc = []
        record_loss_gcc = []
        done = False
        state = torch.Tensor(env.reset(trace_list[i]))

        last_estimation = 300000
        action = 0
        time_to_guide = False
        time_step=0
        while not done and time_step <= 1000:
            state, reward, done, last_estimation, delay, loss,receiving_rate = env.step(action, last_estimation, time_to_guide)
            state = torch.Tensor(state)
            # record_state.append(state)
            # record_reward.append(reward)
            real_estimation = last_estimation
            record_action_gcc.append(real_estimation)
            record_delay_gcc.append(delay)
            record_loss_gcc.append(loss)
            time_step += 1

        average_delay_list_gcc.append(np.mean(record_delay_gcc))
        delay_sort_gcc=record_delay_gcc[:]
        delay_sort_gcc.sort()
        p50_delay_list_gcc.append(delay_sort_gcc[500])
        p95_delay_list_gcc.append(delay_sort_gcc[950])
        loss_gcc.append(np.mean(record_loss_gcc))
        max_delay_gcc=max(record_delay_gcc)
        min_delay_gcc=min(record_delay_gcc)
        delay_score_gcc=(max_delay_gcc-delay_sort_gcc[950])/(max_delay_gcc-min_delay_gcc/2)
        loss_score_gcc=(1-np.mean(record_loss_gcc))

        #for purerl
        state = torch.Tensor(rlenv.reset(trace_list[i]))
        trace_path = 'traces/' + trace_list[i]
        last_estimation = 300000
        action = 0
        done = False
        model2.random_action = False
        time_to_guide = True
        time_step = 0
        record_action_pureRL = []
        record_delay_pureRL = []
        record_loss_pureRL = []
        while not done and time_step <= 1000:
            action, _, _, _ = model2.forward(state)
            state, reward, done, last_estimation, delay, loss ,receiving_rate= rlenv.step(action, last_estimation, time_to_guide)
            state = torch.Tensor(state)
            # record_state.append(state)
            # record_reward.append(reward)
            real_estimation = last_estimation
            record_action_pureRL.append(real_estimation)
            record_delay_pureRL.append(delay)
            record_loss_pureRL.append(loss)
            time_step += 1
        average_delay_list_purerl.append(np.mean(record_delay_pureRL))
        delay_sort_purerl = record_delay_pureRL[:]
        delay_sort_purerl.sort()
        p50_delay_list_purerl.append(delay_sort_purerl[500])
        p95_delay_list_purerl.append(delay_sort_purerl[950])
        loss_purerl.append(np.mean(record_loss_pureRL))
        # #for trace
        max_delay_purerl=max(record_delay_pureRL)
        min_delay_purerl=min(record_delay_pureRL)
        delay_score_purerl=(max_delay_purerl-delay_sort_purerl[950])/(max_delay_purerl-min_delay_purerl/2)
        loss_score_purerl=(1-np.mean(record_loss_pureRL))

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
            # print(duration_list)
            # print(capacity_list)
            # print(time_list)
            # duration_sum = sum(duration_list)
            t = 0
            trace_x = []
            trace_y = []

            j=0
            for a in range(500):
                if t <= time_list[j]:
                    trace_y.append(capacity_list[j])
                else:
                    j+=1
                    trace_y.append(capacity_list[j])
                t += 200
            ave_cap_list.append(np.mean(trace_y))
            var_list.append(np.std(trace_y))
            band_uil_list_RL.append(np.mean(record_action_RL) / np.mean(trace_y))
            band_uil_list_gcc.append(np.mean(record_action_gcc) / np.mean(trace_y))

            if (np.mean(record_action_pureRL) / np.mean(trace_y)) >= 1:
                band_uil_list_purerl.append(1)
                purerl_band=1
            else:
                band_uil_list_purerl.append(np.mean(record_action_pureRL) / np.mean(trace_y))
                purerl_band=np.mean(record_action_pureRL) / np.mean(trace_y)

            qoe_list_RL.append((20*delay_score_RL+20*(np.mean(record_action_RL) / np.mean(trace_y))+30*loss_score_RL))
            qoe_list_gcc.append((20 * delay_score_gcc + 20 * (np.mean(record_action_gcc) / np.mean(trace_y)) + 30 * loss_score_gcc))
            qoe_list_pureRL.append((20 * delay_score_purerl + 20 * purerl_band + 30 * loss_score_purerl))
            # j=0
            # plt.plot(trace_x, trace_y)
            # plt.ylim((0, 2000000))
            # plt.savefig("{}test_result_gcc.jpg".format(path))
        print(i)
    draw_stationary(data_path,average_delay_list_RL,band_uil_list_RL,average_delay_list_gcc,band_uil_list_gcc,average_delay_list_purerl,band_uil_list_purerl)
    # print(max(ave_cap_list),'max')
    # print(min(ave_cap_list), 'min')
    # print(max(var_list), 'max std')
    # print(min(var_list), 'min std')
    # draw_state(record_action_RL, record_delay_RL, record_action_gcc,record_delay_gcc,record_action_pureRL,record_delay_pureRL,trace_y, path=data_path)
    print(np.mean(band_uil_list_RL), 'band RL')
    print(np.mean(band_uil_list_gcc), 'band gcc')
    print(np.mean(band_uil_list_purerl), 'band purerl')
    print(np.mean(average_delay_list_RL), 'avg delay RL')
    print(np.mean(average_delay_list_gcc), 'avg delay gcc')
    print(np.mean(average_delay_list_purerl), 'avg delay purerl')
    print(np.mean(p50_delay_list_RL), 'p50 delay RL')
    print(np.mean(p50_delay_list_gcc), 'p50 delay gcc')
    print(np.mean(p50_delay_list_purerl), 'p50 delay purerl')
    print(np.mean(p95_delay_list_RL), 'p95 delay RL')
    print(np.mean(p95_delay_list_gcc), 'p95 delay gcc')
    print(np.mean(p95_delay_list_purerl), 'p95 delay purerl')
    print(np.mean(loss_gcc), 'loss_gcc')
    print(np.mean(loss_RL), 'loss_RL')
    print(np.mean(loss_purerl), 'loss_purerl')
    print(np.mean(qoe_list_RL),'qoe_RL')
    print(np.mean(qoe_list_gcc), 'qoe_gcc')
    print(np.mean(qoe_list_pureRL), 'qoe_pureRL')