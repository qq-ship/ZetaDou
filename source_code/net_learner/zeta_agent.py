import numpy as np
from game_model.auxiliary_means.data_formate import _get_obs_landlord,_get_obs_landlord_down,_get_obs_landlord_up
from game_model.auxiliary_means.catalogue import create_catalogue, create_csv, add_data_csv
from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION
from game_model.action_map.action_map import action_to_ab_id, kdata_ana
from game_model.auxiliary_means.data_formate import action_2_vector
from game_model.auxiliary_means.trajectories_formate import state_formate, score_count, state_formate_matrix

class Tagent(object):

    def __init__(self, id, lnet, kicker, num_actions):
        self.lnet = lnet
        self.use_raw = False
        self.num_actions = num_actions
        self.kicker = kicker
        self.kicker_datas = []
        self.kic_access = 'neural'
        self.probs = []
        self.id = id
        self.track = np.zeros((6, 180), dtype=int)
        self.datas = []
        self.last_action = ""

    def update_data(self, kic_access='neural'):
        self.kicker_datas = []
        self.datas = []
        self.track = np.zeros((6, 180), dtype=int)
        self.kic_access = kic_access

    def update_model(self, net, kicker, kic_access='neural'):
        self.lnet = net
        self.kicker = kicker
        self.kicker_datas = []
        self.datas = []
        self.track = np.zeros((6, 180), dtype=int)
        self.kic_access = kic_access

    def step(self, state):
        st, action_seq = state_formate_matrix(state,self.id)
        self.track = np.delete(self.track, [3, 4, 5], axis=0)
        self.track = np.concatenate((action_seq, self.track), axis=0)
        legal_actions_id = list(state['legal_actions'].keys())
        rule_actions = state['rule_actions']
        action, prob, kicker_data = self.lnet.choose_action(np.expand_dims(st, 0), np.expand_dims(self.track, 0),
                                                            legal_actions_id, rule_actions, self.kicker,
                                                            model='train', k_model=self.kic_access)
        self.last_action = ID_2_ACTION[action]
        # medic = {'obs': st, 'ac_seq': self.track, 'action': action_to_ab_id(self.last_action), 'prob': prob, 'val': 0,'ab_legal':[],'st_action':[],'st_batch':[],'ac_seq_batch':[]}
        medic = {'obs': st, 'action': action_to_ab_id(ID_2_ACTION[action]), 'prob': prob, 'seq': self.track,
                 'val': 0}

        self.datas.append(medic)
        self.kicker_datas.append(kicker_data)

        return action

    def eval_step(self, state):

        pa = 0
        return self.step(state),pa
