import json
import os
from collections import OrderedDict
from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION
import random
import numpy as np
import collections
ROOT_PATH = './'
with open(os.path.join(ROOT_PATH, 'game_model/action_map/specific_map.json'), 'r') as file:
    SPECIFIC_MAP = json.load(file, object_pairs_hook=OrderedDict)

with open(os.path.join(ROOT_PATH, 'game_model/action_map/action_space.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)

ID_SPACE =dict(zip(ACTION_SPACE.values(),ACTION_SPACE.keys()))

CARD_ID = {'3':0,'4':1,'5':2,'6':3,'7':4,'8':5,'9':6,'T':7,'J':8,'Q':9,'K':10,'A':11,'2':12,'B':13,'R':14}
ID_CARD = dict(zip(CARD_ID.values(),CARD_ID.keys()))

KICKER_CARD_ID = {'3':0,'4':1,'5':2,'6':3,'7':4,'8':5,'9':6,'T':7,'J':8,'Q':9,'K':10,'A':11,'2':12,'B':13,'R':14,'33':15,'44':16,'55':17,'66':18,'77':19,'88':20,'99':21,'TT':22,'JJ':23,'QQ':24,'KK':25,'AA':26,'22':27}
KICKER_ID_CARD = dict(zip(KICKER_CARD_ID.values(),KICKER_CARD_ID.keys()))

def vec_to_matrix(vec):
    max = vec[-1]
    vec = np.delete(vec, -1)
    min = vec[-1]
    king = np.array([min, 0, 0, 0,max,0,0,0])
    vec = np.delete(vec, -1)
    vec = np.append(vec,king)
    vv = vec.reshape(15, 4)
    """
    vec = np.sum(vec.reshape(15, 4), axis=1)
    vv = np.zeros((15, 4),dtype=int)

    for i in range(15):
        if vec[i] != 0:
            tmp = int(vec[i])
            vv[i][tmp-1] = 1
    """

    vv = np.transpose(vv)
    return vv

def ws_matrix(st):
    matrix = []
    for item in st:
        matrix.append(vec_to_matrix(item))

    return np.array(matrix)



def action_to_ab_id(action):
    action_abstact = SPECIFIC_MAP[action][0]
    iid = ACTION_SPACE[action_abstact]
    return iid

def id_to_ab_action(id):
    return ID_SPACE[id]


def marix_to_str(state):
    current_hand = ''
    tmp = state[0][0]
    for i in range(15):
        t = tmp[:,i]
        for item in t:
            if item == 1:
                current_hand = current_hand + ID_CARD[i]
    return current_hand

def r_decode_action(action_id,legal_actions,rule_actions):
    ''' Action id -> the action in the game. Must be implemented in the child class.

    Args:
        action_id (int): the id of the action

    Returns:
        action (string): the action that will be passed to the game engine.
    '''
    abstract_action = ID_SPACE[action_id]
    # without kicker

    if '*' not in abstract_action:
        return abstract_action

    specific_actions = []
    kickers = []

    for legal_action in legal_actions:
        for abstract in SPECIFIC_MAP[legal_action]:
            main = abstract.strip('*')
            if abstract == abstract_action:
                specific_actions.append(legal_action)
                kickers.append(legal_action.replace(main, '', 1))
                break

    kicker_scores = []
    for kicker in kickers:
        score = 0
        for action in rule_actions:
            if kicker in action:
                score += 1
        kicker_scores.append(score + CARD_ID[kicker[0]])
    min_index = 0
    min_score = kicker_scores[0]
    for index, score in enumerate(kicker_scores):
        if score < min_score:
            min_score = score
            min_index = index

    return specific_actions[min_index]

def action_2_vector(action):
    dd = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12, 'B': 13, 'R': 14}

    if action == 'pass':
        return np.zeros([54],dtype=np.int)
    else:
        aa = np.zeros([4,13],dtype=np.int)
        kk = np.zeros([2],dtype=np.int)
        for item in action:

            if item == 'B':
                kk[0] = 1
            elif item == 'R':
                kk[1] = 1
            else:
                id = dd[item]

                for m in range(4):
                    if aa[m][id] == 0:
                        aa[m][id] = 1
                        break

        aa_f = np.array([])
        for m in range(13):
            aa_f = np.hstack((aa_f,aa[:,m]))


        aa_f = np.hstack((aa_f,kk))
        return aa_f

def kicker_state(state,action,num,kicker_card = None):
    abstract = ID_SPACE[action]
    main = abstract.strip('*')
    main_state_matrix = np.zeros((4, 15))
    kicker_state_matrix = np.zeros((4, 15))

    for item in main:
        id = CARD_ID[item]
        for i in range(4):
            if main_state_matrix[i][id] == 0:
                main_state_matrix[i][id] = 1
                break
    if num != 0:
        for item in kicker_card:
            id = CARD_ID[item]
            for i in range(4):
                if kicker_state_matrix[i][id] == 0:
                    kicker_state_matrix[i][id] = 1
                    break

    main_matrix = main_state_matrix.reshape((1, 1, 4, 15))
    kick_matrix = kicker_state_matrix.reshape((1, 1, 4, 15))

    n_state = np.concatenate((state, main_matrix), axis=1)
    n_state = np.concatenate((n_state, kick_matrix), axis=1)

    return n_state

def n_decode_action(state,seq,action,legal_actions, r_actions,kicker,model):

    k_state = []
    k_action = []
    k_prob = []
    k_seq = []
    k_reward = []
    data = {}
    if 41<=action<=66: # trio with single
        #print("trio with single")
        kicker_legal_action = []
        abstract = ID_SPACE[action]
        main = abstract.strip('*')
        action_length = len(abstract)
        b_str = main + main[0]

        ks = kicker_state(state,action,0)

        k_state.append(ks)


        for item in legal_actions:

            if main in item and len(item) == action_length and item != b_str:
                kk = item.replace(main,'')
                kicker_legal_action.append(KICKER_CARD_ID[kk])
        kicker_id,pp = kicker.choose_action(ks,seq,kicker_legal_action,model)

        k_action.append(kicker_id)
        k_reward.append(0)
        k_prob.append(pp)
        k_seq.append(seq)

        kicker_card = KICKER_ID_CARD[kicker_id]
        aa = main + kicker_card
        ids = sorted([KICKER_CARD_ID[item] for item in aa])

        action_str = ''
        for item in ids:
            action_str  = action_str + ID_CARD[item]

        data = {'k_state': k_state, 'k_action': k_action, 'k_reward': k_reward,'k_prob':k_prob,'k_seq':k_seq}
        return action_str,data

    elif 200<=action<=237: #Plane with solo
        #print("Plane with solo")
        #print("=======",legal_actions)
        kicker_legal_action_str = []
        kicker_cards= ''
        abstract = ID_SPACE[action]
        main = abstract.strip('*')
        action_length = len(abstract)
        current_hand = marix_to_str(state)
        tmp_state = state
        tmp_curr = current_hand
        ff_card = main[0]
        ll_card = main[-1]
       # print("ff",ff_card,"ll",ll_card,CARD_ID[ff_card],CARD_ID[ll_card],main)

        for item  in main:
            current_hand = current_hand.replace(item,'')

        num = int(len(main)/3)

        for item in legal_actions:
            if main in item and len(item) == action_length:
                kk = item.replace(main,'')
                kicker_legal_action_str.append(kk)

        kicker_aa = ''

        for i in range(num):
            ks = kicker_state(state, action, i,kicker_aa)
            kicker_legal_action = set([KICKER_CARD_ID[item] for item in current_hand])
            kicker_legal_action = list(kicker_legal_action)

            if len(kicker_legal_action) == 0:
                print("tmp_curr:",tmp_curr,"tmp_state:",tmp_state,"current_hand",current_hand,"kicker_cards",kicker_cards,"main::",main)

            kicker_id ,pp= kicker.choose_action(ks,seq,kicker_legal_action,model)

            k_state.append(ks)
            k_action.append(kicker_id)
            k_reward.append(0)
            k_prob.append(pp)
            k_seq.append(seq)

            kicker_card = KICKER_ID_CARD[kicker_id]
            kicker_aa = kicker_card
            kicker_cards = kicker_cards + kicker_card

            obj = collections.Counter(kicker_cards)
            values = obj.values()

            #print("current=======",current_hand)
            if 2 not in values:
                current_hand = current_hand.replace(kicker_card, '', 1)
            else:
                if CARD_ID[kicker_card] == CARD_ID[ff_card]-1 or CARD_ID[kicker_card] == CARD_ID[ll_card]+1:
                    current_hand = current_hand.replace(kicker_card, '')
                else:
                    current_hand = current_hand.replace(kicker_card, '', 1)

            if 3 in values:
                current_hand = current_hand.replace(kicker_card, '')

            if 'B' in kicker_cards:
                current_hand = current_hand.replace('R', '')
            elif 'R' in kicker_cards:
                current_hand = current_hand.replace('B', '')


        tmp = []
        for item in kicker_cards:
            tmp.append(CARD_ID[item])
        tmp = sorted(tmp)
        kicker_cards = ''

        for item in tmp:
            kicker_cards = kicker_cards + ID_CARD[item]

        if kicker_cards not in kicker_legal_action_str:
            kicker_cards = random.sample(kicker_legal_action_str, 1)[0]

        aa = main + kicker_cards
        ids = sorted([KICKER_CARD_ID[item] for item in aa])
        action_str = ''
        for item in ids:
            action_str = action_str + ID_CARD[item]

        data = {'k_state': k_state, 'k_action': k_action, 'k_reward': k_reward,'k_prob':k_prob,'k_seq':k_seq}
        return action_str,data

    elif 238<=action<=267: #Plane with pair
        #print("Plane with pair")
        kicker_legal_action_str = []
        kicker_legal_action = []
        kicker_cards_str = ''
        kicker_cards= []
        abstract = ID_SPACE[action]
        main = abstract.strip('*')
        action_length = len(abstract)
        b_str = main + main[0]
        num = int(len(main) / 3)

        for item in legal_actions:
            if main in item and len(item) == action_length and item != b_str:
                kk = item.replace(main, '')
                kicker_legal_action_str.append(kk)

        for item in kicker_legal_action_str:
            for i in range(15,28):
                if KICKER_ID_CARD[i] in item:
                    kicker_legal_action.append(i)

        kicker_legal_action = list(set(kicker_legal_action))

        kicker_aa = ''
        for i in range(num):
            ks = kicker_state(state, action, i,kicker_aa)
            if len(kicker_legal_action) == 0:
                print("tmp_curr:",tmp_curr,"tmp_state:",tmp_state,"current_hand",current_hand,"kicker_cards",kicker_cards,"main::",main)

            kicker_id,pp = kicker.choose_action(ks,seq,kicker_legal_action,model)
            kicker_card = KICKER_ID_CARD[kicker_id]

            k_state.append(ks)
            k_action.append(kicker_id)
            k_reward.append(0)
            k_prob.append(pp)
            k_seq.append(seq)

            kicker_aa  = kicker_card
            kicker_cards.append(kicker_card)
            kicker_legal_action.remove(kicker_id)

        tmp = [KICKER_CARD_ID[item] for item in kicker_cards]
        tmp = sorted(tmp)
        for item in tmp:
            kicker_cards_str = kicker_cards_str + KICKER_ID_CARD[item]

        if kicker_cards_str not in kicker_cards:
            kicker_cards_str = random.sample(kicker_legal_action_str, 1)[0]

        aa = main + kicker_cards_str
        ids = sorted([CARD_ID[item] for item in aa])

        action_str = ''
        for item in ids:
            action_str = action_str + ID_CARD[item]
        #return action_str
        data = {'k_state': k_state, 'k_action': k_action, 'k_reward': k_reward,'k_prob':k_prob,'k_seq':k_seq}
        return action_str,data

    elif 268<=action<=280: #Quad with solo
        #print("Quad with solo")
        abstract = ID_SPACE[action]
        action_length = len(abstract)
        main = abstract.strip('*')
        kicker_legal_action_str = []
        current_hand = marix_to_str(state)
        kicker_cards = ''

        for item in main:
            current_hand = current_hand.replace(item, '')

        for item in legal_actions:
            if main in item and len(item) == action_length:
                kk = item.replace(main,'')
                kicker_legal_action_str.append(kk)

        kicker_aa = ''
        for i in range(2):
            ks = kicker_state(state, action, i,kicker_aa)
            kicker_legal_action = set([KICKER_CARD_ID[item] for item in current_hand])
            kicker_legal_action = list(kicker_legal_action)

            kicker_id,pp = kicker.choose_action(ks,seq,kicker_legal_action,model)
            kicker_card = KICKER_ID_CARD[kicker_id]

            kicker_aa = kicker_card
            k_state.append(ks)
            k_prob.append(pp)
            k_seq.append(seq)
            k_action.append(kicker_id)
            k_reward.append(0)

            kicker_cards = kicker_cards + kicker_card

            if 'B' in kicker_cards:
                current_hand = current_hand.replace(kicker_card,'')
                current_hand = current_hand.replace('R', '')
            elif 'R' in kicker_cards:
                current_hand = current_hand.replace(kicker_card,'')
                current_hand = current_hand.replace('B', '')


            for i in range(len(current_hand)):
                if current_hand[i] == kicker_card:
                    current_hand = current_hand.replace(current_hand[i] ,'',1)
                    break
        tmp = []
        for item in kicker_cards:
            tmp.append(CARD_ID[item])
        tmp = sorted(tmp)
        kicker_cards = ''

        for item in tmp:
            kicker_cards = kicker_cards + ID_CARD[item]

        if kicker_cards not in kicker_legal_action_str:
            kicker_cards = random.sample(kicker_legal_action_str, 1)[0]

        aa = main + kicker_cards
        ids = sorted([KICKER_CARD_ID[item] for item in aa])
        action_str = ''
        for item in ids:
            action_str = action_str + ID_CARD[item]

        data = {'k_state': k_state, 'k_action': k_action, 'k_reward': k_reward,'k_prob':k_prob,'k_seq':k_seq}
        return action_str,data
    elif 281<=action<=293: #Quad with pair
        #print("Quad with pair")
        abstract = ID_SPACE[action]
        action_length = len(abstract)
        main = abstract.strip('*')
        kicker_legal_action_str = []
        kicker_legal_action = []
        kicker_cards_str = ''
        kicker_cards = []

        for item in legal_actions:
            if main in item and len(item) == action_length:
                kk = item.replace(main, '')
                kicker_legal_action_str.append(kk)

        for item in kicker_legal_action_str:
            for i in range(15,28):
                if KICKER_ID_CARD[i] in item:
                    kicker_legal_action.append(i)

        kicker_legal_action = list(set(kicker_legal_action))

        kicker_aa = ''
        for i in range(2):
            ks = kicker_state(state, action, i,kicker_aa)
            kicker_id,pp = kicker.choose_action(ks,seq,kicker_legal_action,model)
            kicker_card = KICKER_ID_CARD[kicker_id]

            k_state.append(ks)
            k_prob.append(pp)
            k_seq.append(seq)
            k_action.append(kicker_id)
            k_reward.append(0)

            kicker_aa = kicker_card
            kicker_cards.append(kicker_card)
            kicker_legal_action.remove(kicker_id)

        tmp = [KICKER_CARD_ID[item] for item in kicker_cards]
        tmp = sorted(tmp)

        for item in tmp:
            kicker_cards_str = kicker_cards_str + KICKER_ID_CARD[item]

        if kicker_cards_str not in kicker_cards:
            kicker_cards_str = random.sample(kicker_legal_action_str, 1)[0]

        aa = main + kicker_cards_str
        ids = sorted([CARD_ID[item] for item in aa])

        action_str = ''
        for item in ids:
            action_str = action_str + ID_CARD[item]

        data = {'k_state': k_state, 'k_action': k_action, 'k_reward': k_reward,'k_prob':k_prob,'k_seq':k_seq}
        return action_str,data


def kdata_ana(kdaras):
    obs, seq, act, val, prob = [],[],[],[],[]

    if len(kdaras) == 0:
        return obs, seq, act, val, prob
    else:
        for ts in kdaras:
            obs.append(ts['k_state'])
            act.append(ts['k_action'])
            seq.append(ts['k_seq'])
            val.append(ts['k_reward'])
            prob.append(ts['k_prob'])
        return obs, seq, act, val, prob
"""
if __name__ == "__main__":
    id = 41
    #34445
    legal_actions = ['pass','3444','4445']
    rule_actions = ['3','4','5','44','3444','4445']

    aa = decode_action(41,legal_actions,rule_actions)
    print("===========",aa)
"""