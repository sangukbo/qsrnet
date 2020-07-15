import numpy as np

from qsrnet.dbn.conditional_probabilities import *

class hidden_variable:
    def __init__(self, variable_name, possible_states_list, parent_list_t_1, parent_list_t, cpd_function_name):
        self.var_type = 'hidden'
        self.var_name = variable_name
        self.possible_states = possible_states_list
        self.parents_t_1 = parent_list_t_1    # except itself
        self.parents_t = parent_list_t
        self.transition_function_name = cpd_function_name

class observation_variable:
    def __init__(self, variable_name, possible_states_list, parent_list, cpd_function_name):
        self.var_type = 'observation'
        self.var_name = variable_name
        self.possible_states = possible_states_list
        self.parents = parent_list    # corresponds to parent_list_t
        self.observation_function_name = cpd_function_name

class dynamic_bayesian_network:

    def __init__(self):
        self.var_list = []

    def add_hidden_variable(self, variable_name, possible_states_list, parent_list_t_1, parent_list_t, cpd_function_name):
        self.var_list.append(hidden_variable(variable_name, possible_states_list, parent_list_t_1, parent_list_t, cpd_function_name))
    def add_observation_variable(self, variable_name, possible_states_list, parent_list, cpd_function_name):
        self.var_list.append(observation_variable(variable_name, possible_states_list, parent_list, cpd_function_name))


class particle:

    def __init__(self, initial_state_dict, initial_weight):
        self.state_now = initial_state_dict
        self.weight = initial_weight
        self.state_next = {}

    def update_particle(self, dbn, observations):
        for var in dbn.var_list:
            if var.var_type == 'hidden':
                transition_prob_list = globals()[var.transition_function_name](self.state_now, self.state_next)
                var_next_index = np.argwhere(np.random.multinomial(1, transition_prob_list, size = 1)[0]>0.1)[0][0]
                self.state_next[var.var_name] = var.possible_states[var_next_index]
            if var.var_type == 'observation' and observations[var.var_name] != 'empty':
                observation_prob = globals()[var.observation_function_name](self.state_next, observations[var.var_name])
                self.weight = self.weight * observation_prob
        self.state_now = self.state_next
        self.state_next = {}

class particle_filter:

    def __init__(self, dbn, num_particles, initial_state_dict):
        self.particle_list = []
        self.dbn = dbn
        self.num_particles = num_particles
        self.total_weight = 1.0
        for i in range(self.num_particles):
            self.particle_list.append(particle(initial_state_dict, self.total_weight/self.num_particles))

    def update(self, observations):
        self.total_weight = 0.0
        for p in self.particle_list:
            p.update_particle(self.dbn, observations)
            self.total_weight += p.weight
        self.normalize()
    def normalize(self):
        for p in self.particle_list:
            p.weight = p.weight/self.total_weight
        self.total_weight = 1.0
    def combine_particle_list(self):
        combined_particle_list = []
        for p1 in self.particle_list:
            find_ind = False
            for p2 in combined_particle_list:
                if p1.state_now == p2.state_now:
                    p2.weight += p1.weight
                    find_ind = True
                    break
            if not find_ind:
                combined_particle_list.append(particle(p1.state_now, p1.weight))
        return combined_particle_list
    def compute_max_joint(self):
        combined_particle_list = self.combine_particle_list()
        sorted_combined_particle_list = sorted(combined_particle_list, key = lambda i:i.weight, reverse = True)
        return {'state': sorted_combined_particle_list[0].state_now, 'probability': sorted_combined_particle_list[0].weight}
    def resample(self):                             # this
        combined_particle_list = self.combine_particle_list()
        prob_list = []
        for p in combined_particle_list:
            if p.weight<1.0:
                prob_list.append(p.weight)
            else:
                prob_list.append(1-1e-20)
        for i in range(self.num_particles):
            resampled_state_dict = combined_particle_list[np.argwhere(np.random.multinomial(1, prob_list, size = 1)[0]>0.1)[0][0]].state_now
            self.particle_list[i].state_now = resampled_state_dict
            self.particle_list[i].weight = self.total_weight/self.num_particles
            # self.particle_list.append(particle(resampled_state_dict, self.total_weight/self.num_particles))

########################################

# qsr
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
