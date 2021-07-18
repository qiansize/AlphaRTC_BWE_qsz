import json
import os
import random
import numpy as np
TRACE_FOLDER = 'D/home/qsz/PycharmProjects/AlphaRTC_BWE_qsz/trace_bak'

def convert2trace_belgium(trace_file):
    f = open (trace_file,'r')
    opt_file = 'D:/pythonProject/' + trace_file + '.json'
    lines = f.readlines()
    target_dic = {
        "uplink":{
            "trace_pattern":[
            ]
        }
    }
    for line in lines:
        line = line.strip('\n')
        data = line.split(' ')
        duration = int(data[-1])
        capacity = int(data[-2]) * 8 / duration
        target_dic["uplink"]["trace_pattern"].append({"capacity":capacity,
                                                      "duration":duration})
    j = json.dumps(target_dic,sort_keys=True, indent=4)
    with open (opt_file,'w') as json_file:
        json_file.write(j)

def convert2trace(trace_file):
    timestamp = []
    duration_list = []
    capacity_list = []
    f = open(TRACE_FOLDER + trace_file,'r')
    opt_file = 'E:/gym/test_traces2/' + trace_file.strip('log') + 'json'
    lines = f.readlines()
    target_dic = {
        "uplink":{
            "trace_pattern":[
            ]
        }
    }
    for line in lines:
        line = line.strip('\n')
        data = line.split('\t')
        timestamp.append(float(data[0]) * 1000)
        capacity_list.append(float(data[1]) * 1000)

    for i in range(len(timestamp) - 1):
        duration_list.append(float(format(timestamp[i + 1] - timestamp[i],'.1f')))
    capacity_list.pop()
    for i in range(len(duration_list)):
        target_dic["uplink"]["trace_pattern"].append({"capacity": float(format(capacity_list[i],'.1f')),
                                                      "duration": duration_list[i]})

    j = json.dumps(target_dic,sort_keys=True, indent=4)
    with open (opt_file,'w') as json_file:
        json_file.write(j)
# Overall, our study indicates that for QoE-based performance evaluations, it is in most cases sufficient to only consider easier to understand and easier to model uniformly distributed random loss.
if __name__ == "__main__":
    # trace_files = os.listdir(TRACE_FOLDER)
    # for trace_file in trace_files:
    #     convert2trace(trace_file)
        # convert2trace_belgium(trace_file)
    for j in range(10):
        change_interval= 500
        capacity_set=[300,500,700,850,1000,1200,1400,1600]
        init_pi=[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]
        A=np.array([[0.95,0.05,0,0,0,0,0,0],
                    [0.025,0.95,0.025,0,0,0,0,0],
                    [0,0.025,0.95,0.025,0,0,0,0],
                    [0,0,0.025,0.95,0.025,0,0,0],
                    [0,0,0,0.025,0.95,0.025,0,0],
                    [0,0,0,0,0.025,0.95,0.025,0],
                    [0,0,0,0,0,0.025,0.95,0.025],
                    [0,0,0,0,0,0,0.05,0.95]])
        q_t = np.random.choice(capacity_set, size=1, p=init_pi)
        duration_all = 2000*200
        random_noise=True
        noise_range= 0.05
        loss=0.02
        capacity_list=[]
        duration_list=[]
        for i in range(int(duration_all/change_interval)):
            q_t = np.random.choice(capacity_set, size=1, p=A[capacity_set.index(q_t)])
            capacity_list.append(float(q_t + random.gauss(0, 1) * noise_range * q_t))
            duration_list.append(change_interval)


        target_dic = {
            "uplink":{
                "trace_pattern":[
                ]
            }
        }

        # for i in range(len(duration_set)):
        #     for step in range(int(duration_set[i]/change_interval)):
        #         capacity_list.append(float(capacity_set[i]+random.gauss(0,1)*noise_range*capacity_set[i]))
        #         duration_list.append(change_interval)

        for i in range(int(duration_all/change_interval)):
            target_dic["uplink"]["trace_pattern"].append({"capacity": float(format(capacity_list[i],'.1f')),
                                                          "duration": duration_list[i],
                                                          "loss":loss,
                                                          "rtt":300})

        f = json.dumps(target_dic, sort_keys=True, indent=4)
        opt_file='/home/qsz/PycharmProjects/AlphaRTC_BWE_qsz/traces/loss_0.02_'+str(j)+'.json'

        with open (opt_file,'w') as json_file:
            json_file.write(f)
