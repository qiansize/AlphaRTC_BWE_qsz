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
from alphartc_gym.utils.packet_record import PacketRecord
from BandwidthEstimator_gcc_change import GCCEstimator
import torch.multiprocessing as mp
import numpy as np
from ActorCritic import ActorCritic

class Estimator(object):
    def __init__(self):
        self.time_to_guide=False
        self.now_ms = -1
        self.last_bandwidth_estimation=300000
        self.packet_record = PacketRecord()
        self.gcc_estimator = GCCEstimator()
        self.step_time=200
        self.gcc_decision=300000

    def report_states(self,stats:dict):
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.ssrc = pkt["ssrc"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.padding_length = pkt["padding_length"]
        packet_info.header_length = pkt["header_length"]
        packet_info.payload_size = pkt["payload_size"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        packet_info.bandwidth_prediction = self.last_bandwidth_estimation
        self.now_ms = packet_info.receive_timestamp  # 以最后一个包的到达时间作为系统时间
        self.packet_record.on_receive(packet_info)
        self.gcc_estimator.report_states(pkt)
        # calculate state:
        self.receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)
        #todo
        self.receiving_rate_list.append(self.receiving_rate)
        # states.append(liner_to_log(receiving_rate))
        # self.receiving_rate.append(receiving_rate)
        # np.delete(self.receiving_rate, 0, axis=0)
        self.delay = self.packet_record.calculate_average_delay(interval=self.step_time)
        self.delay_list.append(self.delay)
        # states.append(min(delay/1000, 1))
        # self.delay.append(delay)
        # np.delete(self.delay, 0, axis=0)
        self.loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
        self.loss_ratio_list.append(self.loss_ratio)
        self.bandwidth_prediction=bandwidth_prediction
        self.bandwidth_prediction_list.append(bandwidth_prediction)
        self.gcc_decision = self.gcc_estimator.get_estimated_bandwidth()

        self.state = self.state.clone().detach()
        self.state = torch.roll(self.state, -1, dims=-1)
        # states.append(loss_ratio)
        # self.loss_ratio.append(loss_ratio)
        # np.delete(self.loss_ratio, 0, axis=0)
        # latest_prediction = self.packet_record.calculate_latest_prediction()
        # states.append(liner_to_log(latest_prediction))
        # self.prediction_history.append(latest_prediction)
        # np.delete(self.prediction_history, 0, axis=0)
        # states = np.vstack((self.receiving_rate, self.delay, self.loss_ratio, self.prediction_history))
        # todo: regularization needs to be fixed
        self.state[0, 0, -1] = self.receiving_rate / 300000.0
        self.state[0, 1, -1] = self.delay / 1000.0
        self.state[0, 2, -1] = self.loss_ratio
        self.state[0, 3, -1] = self.bandwidth_prediction/ 300000.0

        # maintain list length
        if len(self.receiving_rate_list) == self.config['state_length']:
            self.receiving_rate_list.pop(0)
            self.delay_list.pop(0)
            self.loss_ratio_list.pop(0)
            self.bandwidth_prediction_list.pop(0)

    def get_estimated_bandwidth(self) -> int:




class PacketInfo:
    def __init__(self):
        self.payload_type = None
        self.sequence_number = None  # int
        self.send_timestamp = None  # int, ms
        self.ssrc = None  # int
        self.padding_length = None  # int, B
        self.header_length = None  # int, B
        self.receive_timestamp = None  # int, ms
        self.payload_size = None  # int, B
        self.bandwidth_prediction = None  # int, bps


# 定义包组的类，记录一个包组的相关信息
class PacketGroup:
    def __init__(self, pkt_group):
        self.pkts = pkt_group
        self.arrival_time_list = [pkt.receive_timestamp for pkt in pkt_group]
        self.send_time_list = [pkt.send_timestamp for pkt in pkt_group]
        self.pkt_group_size = sum([pkt.size for pkt in pkt_group])
        self.pkt_num_in_group = len(pkt_group)
        self.complete_time = self.arrival_time_list[-1]
        self.transfer_duration = self.arrival_time_list[-1] - self.arrival_time_list[0]