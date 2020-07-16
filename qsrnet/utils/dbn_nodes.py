#!/usr/bin/python

import numpy as np

from qsrnet.dbn.conditional_probabilities import *

def qsr_construct_discrete_observation(dbn_qsr):
    dbn_qsr.add_hidden_variable('rcc', ['dc', 'po', 'eq', 'pp', 'ppi'], ['qtc3'], [], 'rcc_transition_function')
    dbn_qsr.add_hidden_variable('qtc1', ['+', '0', '-'], ['qtc3'], [], 'qtc1_transition_function')
    dbn_qsr.add_hidden_variable('qtc2', ['+', '0', '-'], ['qtc3'], [], 'qtc2_transition_function')
    dbn_qsr.add_hidden_variable('qtc3', ['+', '0', '-'], [], [], 'qtc3_transition_function')
    dbn_qsr.add_hidden_variable('qdc', ['far', 'close'], [], [], 'qdc_transition_function')
    dbn_qsr.add_observation_variable('pcd_distances', 'discrete', ['rcc'], 'd1_observation_function')
    dbn_qsr.add_observation_variable('center_distances', 'discrete', ['qdc'], 'd2_observation_function')
    dbn_qsr.add_observation_variable('velocity1', 'discrete', ['qtc1'], 'v1_observation_function')
    dbn_qsr.add_observation_variable('velocity2', 'discrete', ['qtc2'], 'v2_observation_function')
    dbn_qsr.add_observation_variable('velocity3', 'discrete', ['qtc3'], 'v3_observation_function')
    return dbn_qsr

def qsr_get_observation_dict(metrics, pair_id):
    observation_dict = {}
    if pair_id in metrics['pcd_distances']: observation_dict['pcd_distances'] = metrics['pcd_distances'][pair_id]
    else: observation_dict['pcd_distances'] = 'empty'
    if pair_id in metrics['center_distances']: observation_dict['center_distances'] = metrics['center_distances'][pair_id]
    else: observation_dict['center_distances'] = 'empty'
    if pair_id in metrics['velocity1']: observation_dict['velocity1'] = metrics['velocity1'][pair_id]
    else: observation_dict['velocity1'] = 'empty'
    if pair_id in metrics['velocity2']: observation_dict['velocity2'] = metrics['velocity2'][pair_id]
    else: observation_dict['velocity2'] = 'empty'
    if pair_id in metrics['velocity3']: observation_dict['velocity3'] = metrics['velocity3'][pair_id]
    else: observation_dict['velocity3'] = 'empty'
    return observation_dict
