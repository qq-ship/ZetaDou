from op_agent.rhcp_shang_model.utils import to_char, to_value
from op_agent.rhcp_shang_model.env import Env as CEnv
from game_model.auxiliary_means.trajectories_formate import state_formate
from op_agent.rhcp_shang_model.card import Card
from rlcard.games.doudizhu.utils import ACTION_2_ID
import numpy as np

class RhcpShangAgent():
    ''' A random agent. Random agents is for running toy examples on the card games
       '''
    def __init__(self, action_num):
        ''' Initilize the random agent

        Args:
            action_num (int): the size of the ouput action space
        '''
        self.action_num = action_num
        self.use_raw = False
        self.cards = {0: '3', 1: '3', 2: '3', 3: '3', 4: '4', 5: '4', 6: '4', 7: '4', 8: '5', 9: '5', 10: '5', 11: '5',
                 12: '6', 13: '6', 14: '6', 15: '6', 16: '7', 17: '7', 18: '7', 19: '7', 20: '8', 21: '8', 22: '8',
                 23: '8', 24: '9', 25: '9', 26: '9', 27: '9', 28: '10', 29: '10', 30: '10', 31: '10', 32: 'J', 33: 'J',
                 34: 'J', 35: 'J', 36: 'Q', 37: 'Q', 38: 'Q', 39: 'Q', 40: 'K', 41: 'K', 42: 'K', 43: 'K', 44: 'A',
                 45: 'A', 46: 'A', 47: 'A', 48: '2', 49: '2', 50: '2', 51: '2', 52: '*', 53: '$'}

        self.id_card = {0:'3', 1:'4', 2:'5', 3:'6', 4:'7', 5:'8', 6:'9', 7:'T', 8:'J', 9:'Q', 10:'K', 11:'A', 12:'2', 13:'B', 14:'R'}


        self.card_id = {'3':0, '4':1, '5':2, '6':3, '7':4, '8':5, '9':6, '10':7, 'J':8, 'Q':9, 'K':10, 'A':11, '2':12, '*':13, '$':14}


    def state_str(self,state):

        current_hand = []
        last_action = []
        before_last_action = []

        for i in range(len(state[0])):
            if state[0][i] == 1:
                current_hand.append(self.cards[i])

        for i in range(len(state[10])):
            if state[10][i] == 1:
                before_last_action.append(self.cards[i])

        for i in range(len(state[11])):
            if state[11][i] == 1:
                last_action.append(self.cards[i])

        return current_hand,last_action,before_last_action


    def step(self,state):

        st = state_formate(state)
        curreny_hand,ll,bb = self.state_str(st)


        if len(ll) > 0:
            upcard = ll
        elif len(ll) == 0 and len(bb) > 0:
            upcard = bb
        else:
            upcard = []

        putcard = to_char(CEnv.step_auto_static(Card.char2color(curreny_hand), to_value(upcard)))

        if len(putcard) == 0:
            acstr = 'pass'
        else:
            acstr = ''
            ii = []
            for item in putcard:
                ii.append(self.card_id[item])
            ii.sort()

            for item in ii:
                acstr = acstr + self.id_card[item]

        action = ACTION_2_ID[acstr]

        #print("********",curreny_hand,":::::",acstr)

        return action

    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        pa = 0
        return self.step(state),pa