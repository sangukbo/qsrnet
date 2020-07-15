from scipy.stats import norm
import numpy as np
import pickle
import csv
import time

import pdb, traceback, sys
from qsrnet.dbn.particle_filter import *

import socket

IP = ''; PORT = 5050; SIZE = 4*1024; ADDR = (IP, PORT)

qsrnet_dir = '/home/appuser/qsrnet'
detectron_dir = '/home/appuser/detectron2_repo'
densepose_dir = '/home/appuser/detectron2_repo/projects/DensePose'

open(qsrnet_dir + '/data_output/qsr/' + 'qsr_' + '(39,100)' + '.csv', 'w')

full_ids = [39, 100]
pair_list = ['(39,100)']

qsr_particle_number = 200

pf_iteration = 0
resample_num = 5

def dbn_qsr_main():

    global pf_iteration
    global resample_num

    qsr_dbn_dict = {}
    qsr_pf_dict = {}
    qsr_initial_state_dict = {}
    qsr_initial_state_dict['(39,100)'] = {'rcc': 'dc', 'qtc1': '0', 'qtc2': '0', 'qtc3': '0', 'qdc': 'close'}
    for pair_id in pair_list:
        qsr_dbn_dict[pair_id] = dynamic_bayesian_network()
        qsr_dbn_dict[pair_id] = qsr_construct_discrete_observation(qsr_dbn_dict[pair_id])
        qsr_pf_dict[pair_id] = particle_filter(qsr_dbn_dict[pair_id], qsr_particle_number, qsr_initial_state_dict[pair_id])

    # socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(ADDR)  # 주소 바인딩
    server_socket.listen()  # 클라이언트의 요청을 받을 준비
    while True:
        metric_msg = []
        client_socket, client_addr = server_socket.accept()  # 수신대기, 접속한 클라이언트 정보 (소켓, 주소) 반환
        while True:
            packet = client_socket.recv(SIZE)
            if not packet: break
            metric_msg.append(packet)
        metrics = pickle.loads(b"".join(metric_msg))
        print(metrics)
        client_socket.close()

        timestamp = time.time()
        qsr_joint_max_state_dict = {}
        for pair_id in pair_list:
            qsr_observation_dict = qsr_get_observation_dict(metrics, pair_id)
            qsr_pf_dict[pair_id].update(qsr_observation_dict)
            if (pf_iteration+1) % resample_num == 0:
                qsr_pf_dict[pair_id].resample()
            qsr_joint_max_state = qsr_pf_dict[pair_id].compute_max_joint()
            """
            elementwise_max_state_dict = pf_qrk.compute_max_elementwise()
            with open(main_dir + '/' + 'qsr/elementwise/' + 'qsr' + str(time_step + args.pre_num) + '.pickle', 'wb') as f:
                pickle.dump(elementwise_max_state_dict, f, protocol=2)
            """
            with open(qsrnet_dir + '/data_output/qsr/' + 'qsr_' + pair_id + '.csv', 'a') as f:
                field_name = ['rcc', 'qtc1', 'qtc2', 'qtc3', 'qdc', 'probability']
                qsr_dict = {}
                qsr_dict['rcc'] = qsr_joint_max_state['state']['rcc']
                qsr_dict['qtc1'] = qsr_joint_max_state['state']['qtc1']
                qsr_dict['qtc2'] = qsr_joint_max_state['state']['qtc2']
                qsr_dict['qtc3'] = qsr_joint_max_state['state']['qtc3']
                qsr_dict['qdc'] = qsr_joint_max_state['state']['qdc']
                qsr_dict['probability'] = qsr_joint_max_state['probability']
                writer = csv.DictWriter(f, fieldnames=field_name)
                writer.writerow(qsr_dict)
            qsr_joint_max_state_dict[pair_id] = qsr_joint_max_state
        print(qsr_joint_max_state_dict)
        with open(qsrnet_dir + '/data_output/qsr/' + 'qsr_' + str(pf_iteration) + '.pickle', 'wb') as f:
            pickle.dump(qsr_joint_max_state_dict, f, protocol=2)

        pf_iteration += 1
        print(time.time() - timestamp)

if __name__ == '__main__':
    try:
        dbn_qsr_main()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
