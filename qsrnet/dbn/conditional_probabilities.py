from scipy.stats import norm

def rcc_transition_function(state_now, state_next):
    rcc_state = state_now['rcc']
    qtc3_state = state_now['qtc3']

    y1 = 0.05   # intuitive case
    s1 = 0.01
    n1 = 0.002  # non-intuitive case

    if qtc3_state == '+':
        transition_prob_mat = [[1.0-n1, n1, 0.0, 0.0, 0.0], [y1, 1.0-y1-n1, 0.0, n1, 0.0], [0.0, y1, 1.0-y1, 0.0, 0.0], [0.0, y1, 0.0, 1.0-y1, 0.0], [0.0, y1, 0.0, 0.0, 1.0-y1]]
    elif qtc3_state == '0':
        transition_prob_mat = [[1.0-s1, s1, 0.0, 0.0, 0.0], [s1, 1.0-s1-s1, 0.0, s1, 0.0], [0.0, s1, 1.0-s1, 0.0, 0.0], [0.0, s1, 0.0, 1.0-s1, 0.0], [0.0, s1, 0.0, 0.0, 1.0-s1]]
    elif qtc3_state == '-':
        transition_prob_mat = [[1.0-y1, y1, 0.0, 0.0, 0.0], [n1, 1.0-y1-n1, 0.0, y1, 0.0], [0.0, n1, 1.0-n1, 0.0, 0.0], [0.0, n1, 0.0, 1.0-n1, 0.0], [0.0, n1, 0.0, 0.0, 1.0-n1]]

    possible_states = ['dc', 'po', 'eq', 'pp', 'ppi']
    return transition_prob_mat[possible_states.index(rcc_state)]

def qtc1_transition_function(state_now, state_next):
    qtc1_state = state_now['qtc1']

    sq = 0.05
    nq = 0.025

    transition_prob_mat = [[1.0-sq, sq, 0.0], [nq, 1.0-nq-nq, nq], [0.0, sq, 1.0-sq]]

    possible_states = ['+', '0', '-']
    return transition_prob_mat[possible_states.index(qtc1_state)]

def qtc2_transition_function(state_now, state_next):
    qtc2_state = state_now['qtc2']

    sq = 0.05
    nq = 0.025

    transition_prob_mat = [[1.0-sq, sq, 0.0], [nq, 1.0-nq-nq, nq], [0.0, sq, 1.0-sq]]

    possible_states = ['+', '0', '-']
    return transition_prob_mat[possible_states.index(qtc2_state)]

def qtc3_transition_function(state_now, state_next):
    qtc3_state = state_now['qtc3']

    sq = 0.05
    nq = 0.025

    transition_prob_mat = [[1.0-sq, sq, 0.0], [nq, 1.0-nq-nq, nq], [0.0, sq, 1.0-sq]]

    possible_states = ['+', '0', '-']
    return transition_prob_mat[possible_states.index(qtc3_state)]

def qdc_transition_function(state_now, state_next):
    qdc_state = state_now['qdc']

    sq = 0.02
    nq = 0.02

    transition_prob_mat = [[1.0-sq, sq], [nq, 1.0-nq]]

    possible_states = ['far', 'close']
    return transition_prob_mat[possible_states.index(qdc_state)]

"""
def qtc3_transition_function(state_now, state_next):
    rcc_state_now = state_now['rcc']
    rcc_state_next = state_next['rcc']
    qtc1_state = state_now['qtc1']
    qtc2_state = state_now['qtc2']
    qtc3_state = state_now['qtc3']

    y1 = 0.2   # intuitive case
    s1 = 0.05
    n1 = 0.005  # non-intuitive case

    rcc_plus_index = (rcc_state_now == 'po' and rcc_state_next == 'dc') or (rcc_state_now == 'pp' and rcc_state_next == 'po')
    rcc_0_index = (rcc_state_now == 'dc' and rcc_state_next == 'dc') or (rcc_state_now == 'po' and rcc_state_next == 'po') or (rcc_state_now == 'pp' and rcc_state_next == 'pp')
    rcc_minus_index = (rcc_state_now == 'dc' and rcc_state_next == 'po') or (rcc_state_now == 'po' and rcc_state_next == 'pp')

    plus_index = (qtc1_state == '+' or qtc2_state == '+') or rcc_plus_index
    stable_index = (qtc1_state == '0' and qtc2_state == '0') and rcc_0_index
    minus_index = (qtc1_state == '-' or qtc2_state == '-') or rcc_minus_index

    if plus_index:
        transition_prob_mat = [[1.0-n1, n1, 0.0], [y1, 1.0-y1-n1, n1], [0.0, y1, 1.0-y1]]
    if stable_index:
        transition_prob_mat = [[1.0-y1, y1, 0.0], [n1, 1.0-n1-n1, n1], [0.0, y1, 1.0-y1]]
    if minus_index:
        transition_prob_mat = [[1.0-y1, y1, 0.0], [n1, 1.0-y1-n1, y1], [0.0, n1, 1.0-n1]]
    else:
        transition_prob_mat = [[1.0-s1, s1, 0.0], [s1, 1.0-s1-s1, s1], [0.0, s1, 1.0-s1]]

    possible_states = ['+', '0', '-']
    return transition_prob_mat[possible_states.index(qtc3_state)]
"""

def d1_observation_function(state_next, observation):
    rcc_state = state_next['rcc']

    y1 = 0.02
    y2 = 0.05
    y3 = 0.5

    if rcc_state == 'dc':
        observation_prob_mat = [1.0-y1, y1, 0.0, 0.0, 0.0]
    elif rcc_state == 'po':
        observation_prob_mat = [y2, 1.0-y2, 0.0, 0.0, 0.0]
    elif rcc_state == 'eq':
        observation_prob_mat = [y3, 0.0, 1.0 - y3, 0.0, 0.0]
    elif rcc_state == 'pp':
        observation_prob_mat = [y3, 0.0, 0.0, 1.0 - y3, 0.0]
    elif rcc_state == 'ppi':
        observation_prob_mat = [y3, 0.0, 0.0, 0.0, 1.0 - y3]

    possible_states = ['dc', 'po', 'eq', 'pp', 'ppi']

    if observation>150.0: observation = 'dc'
    elif observation<=150.0: observation = 'po'

    return observation_prob_mat[possible_states.index(observation)]

def d2_observation_function(state_next, observation):
    qdc_state = state_next['qdc']

    y1 = 0.01
    y2 = 0.01

    if qdc_state == 'far':
        observation_prob_mat = [1.0-y1, y1]
    elif qdc_state == 'close':
        observation_prob_mat = [y2, 1.0-y2]

    possible_states = ['far', 'close']

    if observation>1000.0: observation = 'far'
    elif observation<=1000.0: observation = 'close'

    return observation_prob_mat[possible_states.index(observation)]

def v1_observation_function(state_next, observation):
    qtc1_state = state_next['qtc1']

    y1 = 0.05
    y2 = 0.01
    y3 = 0.025

    if qtc1_state == '+':
        observation_prob_mat = [1.0-y1 - y2, y1, y2]
    elif qtc1_state == '0':
        observation_prob_mat = [y3, 1.0-y3 - y3, y3]
    elif qtc1_state == '-':
        observation_prob_mat = [y2, y1, 1.0-y1 - y2]

    possible_states = ['+', '0', '-']

    v1_threshold_vel = 150.0
    if observation>v1_threshold_vel: observation = '+'
    elif -v1_threshold_vel<=observation<=v1_threshold_vel: observation = '0'
    elif observation<-v1_threshold_vel: observation = '-'

    return observation_prob_mat[possible_states.index(observation)]

def v2_observation_function(state_next, observation):
    qtc2_state = state_next['qtc2']

    y1 = 0.05
    y2 = 0.01
    y3 = 0.025

    if qtc2_state == '+':
        observation_prob_mat = [1.0-y1 - y2, y1, y2]
    elif qtc2_state == '0':
        observation_prob_mat = [y3, 1.0-y3 - y3, y3]
    elif qtc2_state == '-':
        observation_prob_mat = [y2, y1, 1.0-y1 - y2]

    possible_states = ['+', '0', '-']

    v2_threshold_vel = 150.0
    if observation>v2_threshold_vel: observation = '+'
    elif -v2_threshold_vel<=observation<=v2_threshold_vel: observation = '0'
    elif observation<-v2_threshold_vel: observation = '-'

    return observation_prob_mat[possible_states.index(observation)]

def v3_observation_function(state_next, observation):
    qtc3_state = state_next['qtc3']

    y1 = 0.05
    y2 = 0.01
    y3 = 0.025

    if qtc3_state == '+':
        observation_prob_mat = [1.0-y1 - y2, y1, y2]
    elif qtc3_state == '0':
        observation_prob_mat = [y3, 1.0-y3 - y3, y3]
    elif qtc3_state == '-':
        observation_prob_mat = [y2, y1, 1.0-y1 - y2]

    possible_states = ['+', '0', '-']

    v3_threshold_vel = 150.0
    if observation>v3_threshold_vel: observation = '+'
    elif -v3_threshold_vel<=observation<=v3_threshold_vel: observation = '0'
    elif observation<-v3_threshold_vel: observation = '-'

    return observation_prob_mat[possible_states.index(observation)]
