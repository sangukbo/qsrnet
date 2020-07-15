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

########################################

def atloc_transition_function_0_68(state_now, state_next):
    return atloc_transition_function(state_now, state_next, pair_id = '(0,68)')
def approachingloc_transition_function_0_68(state_now, state_next):
    return approachingloc_transition_function(state_now, state_next, pair_id = '(0,68)')
def goto_transition_function_0_68(state_now, state_next):
    action_activated = state_now['choice1'] == 'large'
    return goto_transition_function(state_now, state_next, action_activated, pair_id = '(0,68)')
def atloc_observation_function_0_68(state_next, observation):
    return atloc_observation_function(state_next, observation, pair_id = '(0,68)')
def approachingloc_observation_function_0_68(state_next, observation):
    return approachingloc_observation_function(state_next, observation, pair_id = '(0,68)')

def atloc_transition_function_0_72(state_now, state_next):
    return atloc_transition_function(state_now, state_next, pair_id = '(0,72)')
def approachingloc_transition_function_0_72(state_now, state_next):
    return approachingloc_transition_function(state_now, state_next, pair_id = '(0,72)')
def goto_transition_function_0_72(state_now, state_next):
    action_activated = state_now['choice1'] == 'small'
    return goto_transition_function(state_now, state_next, action_activated, pair_id = '(0,72)')
def atloc_observation_function_0_72(state_next, observation):
    return atloc_observation_function(state_next, observation, pair_id = '(0,72)')
def approachingloc_observation_function_0_72(state_next, observation):
    return approachingloc_observation_function(state_next, observation, pair_id = '(0,72)')

def holding_transition_function_24_100(state_now, state_next):
    return holding_transition_function(state_now, state_next, pair_id = '(24,100)')
def approaching2_transition_function_24_100(state_now, state_next):
    return approaching2_transition_function(state_now, state_next, pair_id = '(24,100)')
def movingaway2_transition_function_24_100(state_now, state_next):
    return movingaway2_transition_function(state_now, state_next, pair_id = '(24,100)')
def get_transition_function_24_100(state_now, state_next):
    action_activated = state_now['choice4'] == 'clean'
    return get_transition_function(state_now, state_next, action_activated, pair_id = '(24,100)')
def place_transition_function_24_100(state_now, state_next):
    action_activated = state_now['choice4'] == 'clean'
    return place_transition_function(state_now, state_next, action_activated, pair_id = '(24,100)')
def holding_observation_function_24_100(state_next, observation):
    return holding_observation_function(state_next, observation, pair_id = '(24,100)')
def approaching2_observation_function_24_100(state_next, observation):
    return approaching2_observation_function(state_next, observation, pair_id = '(24,100)')
def movingaway2_observation_function_24_100(state_next, observation):
    return movingaway2_observation_function(state_next, observation, pair_id = '(24,100)')

def holding_transition_function_39_100(state_now, state_next):
    return holding_transition_function(state_now, state_next, pair_id = '(39,100)')
def approaching2_transition_function_39_100(state_now, state_next):
    return approaching2_transition_function(state_now, state_next, pair_id = '(39,100)')
def movingaway2_transition_function_39_100(state_now, state_next):
    return movingaway2_transition_function(state_now, state_next, pair_id = '(39,100)')
def get_transition_function_39_100(state_now, state_next):
    action_activated = state_now['choice1'] == 'small' and state_now['choice2'] == 'knife'
    return get_transition_function(state_now, state_next, action_activated, pair_id = '(39,100)')
def place_transition_function_39_100(state_now, state_next):
    action_activated = state_now['choice1'] == 'small' and state_now['choice2'] == 'knife'
    return place_transition_function(state_now, state_next, action_activated, pair_id = '(39,100)')
def holding_observation_function_39_100(state_next, observation):
    return holding_observation_function(state_next, observation, pair_id = '(39,100)')
def approaching2_observation_function_39_100(state_next, observation):
    return approaching2_observation_function(state_next, observation, pair_id = '(39,100)')
def movingaway2_observation_function_39_100(state_next, observation):
    return movingaway2_observation_function(state_next, observation, pair_id = '(39,100)')

def holding_transition_function_40_100(state_now, state_next):
    return holding_transition_function(state_now, state_next, pair_id = '(40,100)')
def approaching2_transition_function_40_100(state_now, state_next):
    return approaching2_transition_function(state_now, state_next, pair_id = '(40,100)')
def movingaway2_transition_function_40_100(state_now, state_next):
    return movingaway2_transition_function(state_now, state_next, pair_id = '(40,100)')
def get_transition_function_40_100(state_now, state_next):
    action_activated = state_now['choice1'] == 'small' and state_now['choice3'] == 'cereal'
    return get_transition_function(state_now, state_next, action_activated, pair_id = '(40,100)')
def place_transition_function_40_100(state_now, state_next):
    action_activated = state_now['choice1'] == 'small' and state_now['choice3'] == 'cereal'
    return place_transition_function(state_now, state_next, action_activated, pair_id = '(40,100)')
def holding_observation_function_40_100(state_next, observation):
    return holding_observation_function(state_next, observation, pair_id = '(40,100)')
def approaching2_observation_function_40_100(state_next, observation):
    return approaching2_observation_function(state_next, observation, pair_id = '(40,100)')
def movingaway2_observation_function_40_100(state_next, observation):
    return movingaway2_observation_function(state_next, observation, pair_id = '(40,100)')

def holding_transition_function_45_100(state_now, state_next):
    return holding_transition_function(state_now, state_next, pair_id = '(45,100)')
def approaching2_transition_function_45_100(state_now, state_next):
    return approaching2_transition_function(state_now, state_next, pair_id = '(45,100)')
def movingaway2_transition_function_45_100(state_now, state_next):
    return movingaway2_transition_function(state_now, state_next, pair_id = '(45,100)')
def get_transition_function_45_100(state_now, state_next):
    action_activated = state_now['choice1'] == 'small' and state_now['choice2'] == 'bowl'
    return get_transition_function(state_now, state_next, action_activated, pair_id = '(45,100)')
def place_transition_function_45_100(state_now, state_next):
    action_activated = state_now['choice1'] == 'small' and state_now['choice2'] == 'bowl'
    return place_transition_function(state_now, state_next, action_activated, pair_id = '(45,100)')
def holding_observation_function_45_100(state_next, observation):
    return holding_observation_function(state_next, observation, pair_id = '(45,100)')
def approaching2_observation_function_45_100(state_next, observation):
    return approaching2_observation_function(state_next, observation, pair_id = '(45,100)')
def movingaway2_observation_function_45_100(state_next, observation):
    return movingaway2_observation_function(state_next, observation, pair_id = '(45,100)')

def holding_transition_function_47_100(state_now, state_next):
    return holding_transition_function(state_now, state_next, pair_id = '(47,100)')
def approaching2_transition_function_47_100(state_now, state_next):
    return approaching2_transition_function(state_now, state_next, pair_id = '(47,100)')
def movingaway2_transition_function_47_100(state_now, state_next):
    return movingaway2_transition_function(state_now, state_next, pair_id = '(47,100)')
def get_transition_function_47_100(state_now, state_next):
    action_activated = state_now['choice1'] == 'large'
    return get_transition_function(state_now, state_next, action_activated, pair_id = '(47,100)')
def place_transition_function_47_100(state_now, state_next):
    action_activated = state_now['choice1'] == 'large'
    return place_transition_function(state_now, state_next, action_activated, pair_id = '(47,100)')
def holding_observation_function_47_100(state_next, observation):
    return holding_observation_function(state_next, observation, pair_id = '(47,100)')
def approaching2_observation_function_47_100(state_next, observation):
    return approaching2_observation_function(state_next, observation, pair_id = '(47,100)')
def movingaway2_observation_function_47_100(state_next, observation):
    return movingaway2_observation_function(state_next, observation, pair_id = '(47,100)')

def holding_transition_function_64_100(state_now, state_next):
    return holding_transition_function(state_now, state_next, pair_id = '(64,100)')
def approaching2_transition_function_64_100(state_now, state_next):
    return approaching2_transition_function(state_now, state_next, pair_id = '(64,100)')
def movingaway2_transition_function_64_100(state_now, state_next):
    return movingaway2_transition_function(state_now, state_next, pair_id = '(64,100)')
def get_transition_function_64_100(state_now, state_next):
    action_activated = state_now['choice4'] == 'call'
    return get_transition_function(state_now, state_next, action_activated, pair_id = '(64,100)')
def place_transition_function_64_100(state_now, state_next):
    action_activated = state_now['choice4'] == 'call'
    return place_transition_function(state_now, state_next, action_activated, pair_id = '(64,100)')
def holding_observation_function_64_100(state_next, observation):
    return holding_observation_function(state_next, observation, pair_id = '(64,100)')
def approaching2_observation_function_64_100(state_next, observation):
    return approaching2_observation_function(state_next, observation, pair_id = '(64,100)')
def movingaway2_observation_function_64_100(state_next, observation):
    return movingaway2_observation_function(state_next, observation, pair_id = '(64,100)')

def holding_transition_function_65_100(state_now, state_next):
    return holding_transition_function(state_now, state_next, pair_id = '(65,100)')
def approaching2_transition_function_65_100(state_now, state_next):
    return approaching2_transition_function(state_now, state_next, pair_id = '(65,100)')
def movingaway2_transition_function_65_100(state_now, state_next):
    return movingaway2_transition_function(state_now, state_next, pair_id = '(65,100)')
def get_transition_function_65_100(state_now, state_next):
    action_activated = state_now['choice1'] == 'small' and state_now['choice3'] == 'orange'
    return get_transition_function(state_now, state_next, action_activated, pair_id = '(65,100)')
def place_transition_function_65_100(state_now, state_next):
    action_activated = state_now['choice1'] == 'small' and state_now['choice3'] == 'orange'
    return place_transition_function(state_now, state_next, action_activated, pair_id = '(65,100)')
def holding_observation_function_65_100(state_next, observation):
    return holding_observation_function(state_next, observation, pair_id = '(65,100)')
def approaching2_observation_function_65_100(state_next, observation):
    return approaching2_observation_function(state_next, observation, pair_id = '(65,100)')
def movingaway2_observation_function_65_100(state_next, observation):
    return movingaway2_observation_function(state_next, observation, pair_id = '(65,100)')

###############################

def choice1_transition_function(state_now, state_next):
    choice1_state = state_now['choice1']
    tpn_state = state_now['tpn']
    if tpn_state == '0': transition_prob_mat = [[0.0, 0.5, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    else: transition_prob_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    possible_states = ['nil', 'large', 'small']
    return transition_prob_mat[possible_states.index(choice1_state)]

def choice2_transition_function(state_now, state_next):
    choice2_state = state_now['choice2']
    tpn_state = state_now['tpn']
    if tpn_state == '8': transition_prob_mat = [[0.0, 0.5, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    else: transition_prob_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    possible_states = ['nil', 'bowl', 'knife']
    return transition_prob_mat[possible_states.index(choice2_state)]

def choice3_transition_function(state_now, state_next):
    choice3_state = state_now['choice3']
    tpn_state = state_now['tpn']
    choice2_state = state_now['choice2']
    if tpn_state == '17' and choice2_state == 'bowl': transition_prob_mat = [[0.0, 0.8, 0.2], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    elif tpn_state == '17' and choice2_state == 'knife': transition_prob_mat = [[0.0, 0.2, 0.8], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    else: transition_prob_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    possible_states = ['nil', 'cereal', 'orange']
    return transition_prob_mat[possible_states.index(choice3_state)]

def choice4_transition_function(state_now, state_next):
    choice4_state = state_now['choice4']
    tpn_state = state_now['tpn']
    choice1_state = state_now['choice1']
    if tpn_state == '27' and choice1_state == 'large': transition_prob_mat = [[0.0, 0.1, 0.9], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if tpn_state == '27' and choice1_state == 'small': transition_prob_mat = [[0.0, 0.9, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    else: transition_prob_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    possible_states = ['nil', 'clean', 'call']
    return transition_prob_mat[possible_states.index(choice4_state)]

def choice5_transition_function(state_now, state_next):
    choice5_state = state_now['choice5']
    tpn_state = state_now['tpn']
    if tpn_state == '35': transition_prob_mat = [[0.0, 0.9, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    else: transition_prob_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    possible_states = ['nil', 'fast', 'late']
    return transition_prob_mat[possible_states.index(choice5_state)]

def atloc_transition_function(state_now, state_next, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        atloc_state = state_now['atloc']
        goto_state = state_now['goto']
    else:
        atloc_state = state_now['atloc_' + pair_id]
        goto_state = state_now['goto_' + pair_id]

    y1 = 0.1
    s1 = 0.02
    n1 = 0.005

    if goto_state == 'almost':
        transition_prob_mat = [[1.0-n1, n1], [y1, 1.0-y1]]
    else:
        transition_prob_mat = [[1.0-s1, s1], [s1, 1.0-s1]]

    possible_states = ['true', 'false']
    return transition_prob_mat[possible_states.index(atloc_state)]

def approachingloc_transition_function(state_now, state_next, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        approachingloc_state = state_now['approachingloc']
        goto_state = state_now['goto']
    else:
        approachingloc_state = state_now['approachingloc_' + pair_id]
        goto_state = state_now['goto_' + pair_id]

    y1 = 0.1
    s1 = 0.02
    n1 = 0.005

    if goto_state == 'ready':
        transition_prob_mat = [[1.0-n1, n1], [y1, 1.0-y1]]
    else:
        transition_prob_mat = [[1.0-s1, s1], [s1, 1.0-s1]]

    possible_states = ['true', 'false']
    return transition_prob_mat[possible_states.index(approachingloc_state)]

def holding_transition_function(state_now, state_next, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        holding_state = state_now['holding']
        get_state = state_now['get']
        place_state = state_now['place']
    else:
        holding_state = state_now['holding_' + pair_id]
        get_state = state_now['get_' + pair_id]
        place_state = state_now['place_' + pair_id]

    y1 = 0.1
    s1 = 0.02
    n1 = 0.005

    if get_state == 'almost':
        transition_prob_mat = [[1.0-n1, n1], [y1, 1.0-y1]]
    elif place_state == 'ready':
        transition_prob_mat = [[1.0-y1, y1], [n1, 1.0-n1]]
    else:
        transition_prob_mat = [[1.0-s1, s1], [s1, 1.0-s1]]

    possible_states = ['true', 'false']
    return transition_prob_mat[possible_states.index(holding_state)]

def approaching2_transition_function(state_now, state_next, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        approaching2_state = state_now['approaching2']
        get_state = state_now['get']
    else:
        approaching2_state = state_now['approaching2_' + pair_id]
        get_state = state_now['get_' + pair_id]

    y1 = 0.1
    s1 = 0.02
    n1 = 0.005

    if get_state == 'ready':
        transition_prob_mat = [[1.0-n1, n1], [y1, 1.0-y1]]
    else:
        transition_prob_mat = [[1.0-s1, s1], [s1, 1.0-s1]]

    possible_states = ['true', 'false']
    return transition_prob_mat[possible_states.index(approaching2_state)]

def movingaway2_transition_function(state_now, state_next, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        movingaway2_state = state_now['movingaway2']
        place_state = state_now['place']
    else:
        movingaway2_state = state_now['movingaway2_' + pair_id]
        place_state = state_now['place_' + pair_id]

    y1 = 0.1
    s1 = 0.02
    n1 = 0.005

    if place_state == 'almost':
        transition_prob_mat = [[1.0-n1, n1], [y1, 1.0-y1]]
    else:
        transition_prob_mat = [[1.0-s1, s1], [s1, 1.0-s1]]

    possible_states = ['true', 'false']
    return transition_prob_mat[possible_states.index(movingaway2_state)]

def goto_transition_function(state_now, state_next, action_activated, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        goto_state = state_now['goto']
        atloc_state_t_1 = state_now['atloc']
        approachingloc_state_t_1 = state_now['approachingloc']
    else:
        goto_state = state_now['goto_' + pair_id]
        atloc_state_t_1 = state_now['atloc_' + pair_id]
        approachingloc_state_t_1 = state_now['approachingloc_' + pair_id]
    # goto_t_e = state_now['goto_t_e_' + pair_id]
    # goto_t_f = state_now['goto_t_f_' + pair_id]

    # change this
    at_start_precondition = approachingloc_state_t_1 == 'false' and action_activated
    at_start_effect = approachingloc_state_t_1 == 'true' and action_activated
    overall = approachingloc_state_t_1 == 'true' and atloc_state_t_1 == 'false' and action_activated
    at_end_precondition = approachingloc_state_t_1 == 'true' and atloc_state_t_1 == 'false' and action_activated
    at_end_effect = atloc_state_t_1 == 'true' and action_activated

    guard_1 = at_start_precondition                                                     # nil to ready
    guard_2 = at_start_effect                                                           # ready to execute
    guard_3 = at_end_precondition                                                       # execute to almost
    guard_4 = at_end_effect                                                             # almost to finish
    guard_5 = True                                                                      # finish to nil
    guard_6 = not at_start_precondition                                                 # ready to fail
    guard_7 = not overall                                                               # execute to fail
    guard_8 = not at_end_precondition                                                   # almost to fail
    guard_9 = True                                                                      # fail to nil

    if guard_1: p1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    else: p1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if guard_2: p2 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    elif guard_6: p2 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else: p2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    if guard_3: p3 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    elif guard_7: p3 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else: p3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if guard_4: p4 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    elif guard_8: p4 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else: p4 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    if guard_5: p5 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else: p5 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    if guard_9: p6 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else: p6 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    transition_prob_mat = [p1, p2, p3, p4, p5, p6]

    possible_states = ['nil', 'ready', 'executing', 'almost', 'finished', 'failed']
    return transition_prob_mat[possible_states.index(goto_state)]

def get_transition_function(state_now, state_next, action_activated, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        get_state = state_now['get']
        holding_state_t_1 = state_now['holding']
        approaching2_state_t_1 = state_now['approaching2']
    else:
        get_state = state_now['get_' + pair_id]
        holding_state_t_1 = state_now['holding_' + pair_id]
        approaching2_state_t_1 = state_now['approaching2_' + pair_id]
    # get_t_e = state_now['get_t_e_' + pair_id]
    # get_t_f = state_now['get_t_f_' + pair_id]

    # change this
    at_start_precondition = approaching2_state_t_1 == 'false' and action_activated
    at_start_effect = approaching2_state_t_1 == 'true' and action_activated
    overall = approaching2_state_t_1 == 'true' and holding_state_t_1 == 'false' and action_activated
    at_end_precondition = approaching2_state_t_1 == 'true' and holding_state_t_1 == 'false' and action_activated
    at_end_effect = holding_state_t_1 == 'true' and action_activated

    guard_1 = at_start_precondition                                                     # nil to ready
    guard_2 = at_start_effect                                                           # ready to execute
    guard_3 = at_end_precondition                                                       # execute to almost
    guard_4 = at_end_effect                                                             # almost to finish
    guard_5 = True                                                                      # finish to nil
    guard_6 = not at_start_precondition                                                 # ready to fail
    guard_7 = not overall                                                               # execute to fail
    guard_8 = not at_end_precondition                                                   # almost to fail
    guard_9 = True                                                                      # fail to nil

    if guard_1: p1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    else: p1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if guard_2: p2 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    elif guard_6: p2 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else: p2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    if guard_3: p3 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    elif guard_7: p3 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else: p3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if guard_4: p4 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    elif guard_8: p4 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else: p4 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    if guard_5: p5 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else: p5 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    if guard_9: p6 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else: p6 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    transition_prob_mat = [p1, p2, p3, p4, p5, p6]

    possible_states = ['nil', 'ready', 'executing', 'almost', 'finished', 'failed']
    return transition_prob_mat[possible_states.index(get_state)]

def place_transition_function(state_now, state_next, action_activated, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        place_state = state_now['place']
        holding_state_t_1 = state_now['holding']
        movingaway2_state_t_1 = state_now['movingaway2']
    else:
        place_state = state_now['place_' + pair_id]
        holding_state_t_1 = state_now['holding_' + pair_id]
        movingaway2_state_t_1 = state_now['movingaway2_' + pair_id]
    # place_t_e = state_now['place_t_e_' + pair_id]
    # place_t_f = state_now['place_t_f_' + pair_id]

    # change this
    at_start_precondition = holding_state_t_1 == 'true' and action_activated
    at_start_effect = holding_state_t_1 == 'false' and action_activated
    overall = holding_state_t_1 == 'false' and movingaway2_state_t_1 == 'true' and action_activated
    at_end_precondition = holding_state_t_1 == 'false' and movingaway2_state_t_1 == 'true' and action_activated
    at_end_effect = movingaway2_state_t_1 == 'false' and action_activated

    guard_1 = at_start_precondition                                                     # nil to ready
    guard_2 = at_start_effect                                                           # ready to execute
    guard_3 = at_end_precondition                                                       # execute to almost
    guard_4 = at_end_effect                                                             # almost to finish
    guard_5 = True                                                                      # finish to nil
    guard_6 = not at_start_precondition                                                 # ready to fail
    guard_7 = not overall                                                               # execute to fail
    guard_8 = not at_end_precondition                                                   # almost to fail
    guard_9 = True                                                                      # fail to nil

    if guard_1: p1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    else: p1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if guard_2: p2 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    elif guard_6: p2 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else: p2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    if guard_3: p3 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    elif guard_7: p3 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else: p3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if guard_4: p4 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    elif guard_8: p4 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else: p4 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    if guard_5: p5 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else: p5 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    if guard_9: p6 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else: p6 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    transition_prob_mat = [p1, p2, p3, p4, p5, p6]

    possible_states = ['nil', 'ready', 'executing', 'almost', 'finished', 'failed']
    return transition_prob_mat[possible_states.index(place_state)]

def atloc_observation_function(state_next, observation, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        atloc_state = state_next['atloc']
    else:
        atloc_state = state_next['atloc_' + pair_id]

    s1 = 0.1
    s2 = 0.05

    if atloc_state == 'true':
        observation_prob_mat = [1.0-s1, s1]
    else:
        observation_prob_mat = [s2, 1.0-s2]

    possible_states = ['true', 'false']

    return observation_prob_mat[possible_states.index(observation)]

def approachingloc_observation_function(state_next, observation, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        approachingloc_state = state_next['approachingloc']
    else:
        approachingloc_state = state_next['approachingloc_' + pair_id]

    s1 = 0.1
    s2 = 0.05

    if approachingloc_state == 'true':
        observation_prob_mat = [1.0-s1, s1]
    else:
        observation_prob_mat = [s2, 1.0-s2]

    possible_states = ['true', 'false']

    return observation_prob_mat[possible_states.index(observation)]

def holding_observation_function(state_next, observation, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        holding_state = state_next['holding']
    else:
        holding_state = state_next['holding_' + pair_id]

    s1 = 0.1
    s2 = 0.05

    if holding_state == 'true':
        observation_prob_mat = [1.0-s1, s1]
    else:
        observation_prob_mat = [s2, 1.0-s2]

    possible_states = ['true', 'false']

    return observation_prob_mat[possible_states.index(observation)]

def approaching2_observation_function(state_next, observation, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        approaching2_state = state_next['approaching2']
    else:
        approaching2_state = state_next['approaching2_' + pair_id]

    s1 = 0.1
    s2 = 0.05

    if approaching2_state == 'true':
        observation_prob_mat = [1.0-s1, s1]
    else:
        observation_prob_mat = [s2, 1.0-s2]

    possible_states = ['true', 'false']

    return observation_prob_mat[possible_states.index(observation)]

def movingaway2_observation_function(state_next, observation, pair_id = 'no pair id'):
    if pair_id == 'no pair id':
        movingaway2_state = state_next['movingaway2']
    else:
        movingaway2_state = state_next['movingaway2_' + pair_id]

    s1 = 0.1
    s2 = 0.05

    if movingaway2_state == 'true':
        observation_prob_mat = [1.0-s1, s1]
    else:
        observation_prob_mat = [s2, 1.0-s2]

    possible_states = ['true', 'false']

    return observation_prob_mat[possible_states.index(observation)]

def transition_to(list_length, to_where):
    transition_prob = [0.0]*list_length
    transition_prob[to_where] = 1.0
    return transition_prob

def tpn_transition_function(state_now, state_next):
    list_length = 43; transition_prob = transition_to(list_length, int(state_now['tpn']))
    if state_now['tpn'] == '0':
        if state_now['choice1'] == 'large': transition_prob = transition_to(list_length, 1)
        elif state_now['choice1'] == 'small': transition_prob = transition_to(list_length, 8)

    if state_now['tpn'] == '1':
        if state_now['get_(47,100)'] == 'executing': transition_prob = transition_to(list_length, 2)
    if state_now['tpn'] == '2':
        if state_now['get_(47,100)'] == 'finished': transition_prob = transition_to(list_length, 3)
    if state_now['tpn'] == '3':
        if state_now['goto_(0,68)'] == 'executing': transition_prob = transition_to(list_length, 4)
    if state_now['tpn'] == '4':
        if state_now['goto_(0,68)'] == 'finished': transition_prob = transition_to(list_length, 5)
    if state_now['tpn'] == '5':
        if state_now['place_(47,100)'] == 'executing': transition_prob = transition_to(list_length, 6)                      # large_meal
    if state_now['tpn'] == '6':
        if state_now['place_(47,100)'] == 'finished': transition_prob = transition_to(list_length, 7)                       # large_meal

    if state_now['tpn'] == '8':
        if state_now['choice2'] == 'bowl': transition_prob = transition_to(list_length, 9)
        elif state_now['choice2'] == 'knife': transition_prob = transition_to(list_length, 12)
    if state_now['tpn'] == '9':
        if state_now['get_(45,100)'] == 'executing': transition_prob = transition_to(list_length, 10)
    if state_now['tpn'] == '10':
        if state_now['get_(45,100)'] == 'finished': transition_prob = transition_to(list_length, 11)

    if state_now['tpn'] == '12':
        if state_now['get_(39,100)'] == 'executing': transition_prob = transition_to(list_length, 13)
    if state_now['tpn'] == '13':
        if state_now['get_(39,100)'] == 'finished': transition_prob = transition_to(list_length, 14)

    if state_now['tpn'] == '11' or state_now['tpn'] == '14': transition_prob = transition_to(list_length, 15)

    if state_now['tpn'] == '15':
        if state_now['goto_(0,72)'] == 'executing': transition_prob = transition_to(list_length, 16)
    if state_now['tpn'] == '16':
        if state_now['goto_(0,72)'] == 'finished': transition_prob = transition_to(list_length, 17)

    if state_now['tpn'] == '17':
        if state_now['choice3'] == 'cereal': transition_prob = transition_to(list_length, 18)
        elif state_now['choice3'] == 'orange': transition_prob = transition_to(list_length, 21)
    if state_now['tpn'] == '18':
        if state_now['get_(40,100)'] == 'executing': transition_prob = transition_to(list_length, 19)
    if state_now['tpn'] == '19':
        if state_now['get_(40,100)'] == 'finished': transition_prob = transition_to(list_length, 20)

    if state_now['tpn'] == '21':
        if state_now['get_(65,100)'] == 'executing': transition_prob = transition_to(list_length, 22)
    if state_now['tpn'] == '22':
        if state_now['get_(65,100)'] == 'finished': transition_prob = transition_to(list_length, 23)

    if state_now['tpn'] == '20' or state_now['tpn'] == '23': transition_prob = transition_to(list_length, 24)

    if state_now['tpn'] == '24':
        if state_now['place_(40,100)'] == 'executing' or state_now['place_(65,100)'] == 'executing': transition_prob = transition_to(list_length, 25)                     # small_meal
    if state_now['tpn'] == '25':
        if state_now['place_(40,100)'] == 'finished' or state_now['place_(65,100)'] == 'finished': transition_prob = transition_to(list_length, 26)                       # small_meal

    if state_now['tpn'] == '7' or state_now['tpn'] == '26': transition_prob = transition_to(list_length, 27)

    if state_now['tpn'] == '27':
        if state_now['choice4'] == 'clean': transition_prob = transition_to(list_length, 28)
        elif state_now['choice4'] == 'call': transition_prob = transition_to(list_length, 33)
    if state_now['tpn'] == '28':
        if state_now['get_(24,100)'] == 'executing': transition_prob = transition_to(list_length, 29)
    if state_now['tpn'] == '29':
        if state_now['get_(24,100)'] == 'finished': transition_prob = transition_to(list_length, 30)
    if state_now['tpn'] == '30':
        if state_now['place_(24,100)'] == 'executing': transition_prob = transition_to(list_length, 31)                      # clean_livingroom_self
    if state_now['tpn'] == '31':
        if state_now['place_(24,100)'] == 'finished': transition_prob = transition_to(list_length, 32)                       # clean_livingroom_self

    if state_now['tpn'] == '33':
        if state_now['get_(64,100)'] == 'executing': transition_prob = transition_to(list_length, 34)
    if state_now['tpn'] == '34':
        if state_now['get_(64,100)'] == 'finished': transition_prob = transition_to(list_length, 35)

    if state_now['tpn'] == '35':
        if state_now['choice5'] == 'fast': transition_prob = transition_to(list_length, 36)
        elif state_now['choice5'] == 'late': transition_prob = transition_to(list_length, 39)
    if state_now['tpn'] == '36':
        if state_now['place_(64,100)'] == 'executing': transition_prob = transition_to(list_length, 37)                      # clean_livingroom_jack
    if state_now['tpn'] == '37':
        if state_now['place_(64,100)'] == 'finished': transition_prob = transition_to(list_length, 38)                       # clean_livingroom_jack

    if state_now['tpn'] == '39':
        if state_now['get_(39,100)'] == 'executing': transition_prob = transition_to(list_length, 40)                      # jack_late
    if state_now['tpn'] == '40':
        if state_now['get_(39,100)'] == 'finished': transition_prob = transition_to(list_length, 41)                       # jack_late

    if state_now['tpn'] == '38' or state_now['tpn'] == '41': transition_prob = transition_to(list_length, 42)

    return transition_prob
