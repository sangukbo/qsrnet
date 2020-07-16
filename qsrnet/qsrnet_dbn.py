#!/usr/bin/python

from scipy.stats import norm
import numpy as np
import pickle
import csv
import time
import json
import socket

import pdb, traceback, sys
from qsrnet.dbn.particle_filter import *
from qsrnet.utils.etc import *
from qsrnet.utils.dbn_nodes import *

IP = ''; PORT = 5050; SIZE = 4*1024; ADDR = (IP, PORT)

with open('/home/appuser/qsrnet/configs/config.json') as json_file:
    configuration = json.load(json_file)
object_ids = get_object_ids(configuration['NAMES']['object_names'], configuration['NAMES']['class_names'])
pair_ids = get_pair_ids(configuration['NAMES']['pair_names'], configuration['NAMES']['class_names'])

for pair_id_csv in pair_ids:
    open(configuration['DIRECTORIES']['qsrnet_dir'] + '/data_output/metric/' + 'metric_' + pair_id_csv + '.csv', 'w')
for pair_id_csv in pair_ids:
    open(configuration['DIRECTORIES']['qsrnet_dir'] + '/data_output/qsr/' + 'qsr_' + pair_id_csv + '.csv', 'w')

def dbn_qsr_main():
    pf_iteration = 0
    qsr_dbn_dict = {}
    qsr_pf_dict = {}
    qsr_initial_state_dict = {}
    for pair_id in pair_ids:
        qsr_dbn_dict[pair_id] = dynamic_bayesian_network()
        qsr_dbn_dict[pair_id] = qsr_construct_discrete_observation(qsr_dbn_dict[pair_id])
        qsr_initial_state_dict[pair_id] = configuration['QSR INITIAL STATES'][pair_name_from_id(pair_id, configuration['NAMES']['class_names'])]
        qsr_pf_dict[pair_id] = particle_filter(qsr_dbn_dict[pair_id], configuration['ARGUMENTS']['qsr_n_particles'], qsr_initial_state_dict[pair_id])

    # socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(ADDR)  # address binding
    server_socket.listen()  # listen to client
    while True:
        metric_msg = []
        client_socket, client_addr = server_socket.accept()  # receive from client
        while True:
            packet = client_socket.recv(SIZE)
            if not packet: break
            metric_msg.append(packet)
        metrics = pickle.loads(b"".join(metric_msg))
        print(metrics)
        # save metric
        if metrics['velocity1'] != {}:
            for pair_id in pair_ids:
                with open(configuration['DIRECTORIES']['qsrnet_dir'] + '/data_output/metric/' + 'metric_' + pair_id + '.csv', 'a') as f:
                    field_name = ['pcd distance', 'center distance', 'velocity 1', 'velocity 2', 'velocity 3']
                    metric_dict = {}
                    metric_dict['pcd distance'] = metrics['pcd_distances'][pair_id]
                    metric_dict['center distance'] = metrics['center_distances'][pair_id]
                    metric_dict['velocity 1'] = metrics['velocity1'][pair_id]
                    metric_dict['velocity 2'] = metrics['velocity2'][pair_id]
                    metric_dict['velocity 3'] = metrics['velocity3'][pair_id]
                    writer = csv.DictWriter(f, fieldnames=field_name)
                    writer.writerow(metric_dict)
        with open(configuration['DIRECTORIES']['qsrnet_dir'] + '/data_output/metric/' + 'metric_' + str(pf_iteration) + '.pickle', 'wb') as f:
            pickle.dump(metrics, f, protocol=2)
        client_socket.close()

        timestamp = time.time()
        qsr_joint_max_state_dict = {}
        for pair_id in pair_ids:
            qsr_observation_dict = qsr_get_observation_dict(metrics, pair_id)
            qsr_pf_dict[pair_id].update(qsr_observation_dict)
            if (pf_iteration+1) % configuration['ARGUMENTS']['resample_at'] == 0:
                qsr_pf_dict[pair_id].resample()
            qsr_joint_max_state = qsr_pf_dict[pair_id].compute_max_joint()
            """
            elementwise_max_state_dict = pf_qrk.compute_max_elementwise()
            with open(main_dir + '/' + 'qsr/elementwise/' + 'qsr' + str(time_step + args.pre_num) + '.pickle', 'wb') as f:
                pickle.dump(elementwise_max_state_dict, f, protocol=2)
            """
            with open(configuration['DIRECTORIES']['qsrnet_dir'] + '/data_output/qsr/' + 'qsr_' + pair_id + '.csv', 'a') as f:
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
        with open(configuration['DIRECTORIES']['qsrnet_dir'] + '/data_output/qsr/' + 'qsr_' + str(pf_iteration) + '.pickle', 'wb') as f:
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
