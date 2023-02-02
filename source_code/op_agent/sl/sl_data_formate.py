import numpy as np
from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION

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

def _get_obs_landlord(state):

    bb_obs = np.hstack((state['obs'][0:162],state['obs'][648:901]))

    legal_actions = np.array(list(state['legal_actions'].values()))

    one_hot_bomb_num = np.zeros(15)

    bomb_num = 0

    actions_str = []
    actions = []

    role_actions = state['raw_obs']['trace']

    for item in role_actions:
        actions.append(ACTION_2_ID[item[1]])
        actions_str.append(item[1])

    for item in actions:
        if 27467 <= item <= 27469:
            bomb_num = bomb_num + 1

    bomb_num = 0
    one_hot_bomb_num[bomb_num] = 1

    bb_obs = np.hstack((bb_obs, one_hot_bomb_num))

    x_batch = np.zeros([len(legal_actions),373])

    for i in range(len(legal_actions)):
        x_batch[i] = np.hstack((bb_obs,legal_actions[i]))

    z_batch = np.zeros([15,54])
    
    rev_actions_str = list(actions_str)
    rev_actions_str.reverse()

    for i in range(len(rev_actions_str)):
        if i < 15:
            z_batch[i] = action_2_vector(rev_actions_str[i])
        else:
            break

    zz_bb = np.flip(z_batch, axis = 0).reshape((5,162))

    zz_batch_final = np.zeros([len(legal_actions),5,162])

    for i in range(len(legal_actions)):
        zz_batch_final[i] = zz_bb

    return x_batch.astype(np.float32), zz_batch_final.astype(np.float32)

def _get_obs_landlord_down(state):


    bb_obs = np.array([])
    legal_actions = np.array(list(state['legal_actions'].values()))
    z_batch = np.zeros([15, 54])

    role_actions = state['raw_obs']['trace']
    actions_str = []
    actions = []

    for item in role_actions:
        actions.append(ACTION_2_ID[item[1]])
        actions_str.append(item[1])

    rev_actions_str = list(actions_str)
    rev_actions_str.reverse()

    for i in range(len(rev_actions_str)):
        if i < 15:
            z_batch[i] = action_2_vector(rev_actions_str[i])
        else:
            break

    zz_bb = np.flip(z_batch, axis=0).reshape((5, 162))

    zz_batch_final = np.zeros([len(legal_actions), 5, 162])

    for i in range(len(legal_actions)):
        zz_batch_final[i] = zz_bb


    my_handcards = state['obs'][0:54]
    bb_obs = np.hstack((bb_obs, my_handcards))

    other_handcards = state['obs'][54:108]
    bb_obs = np.hstack((bb_obs, other_handcards))

    landlord_played_cards = state['obs'][648:702]
    bb_obs = np.hstack((bb_obs, landlord_played_cards))

    teammate_played_cards = state['obs'][702:756]
    bb_obs = np.hstack((bb_obs, teammate_played_cards))

    last_action = state['obs'][108:162]
    bb_obs = np.hstack((bb_obs, last_action))

    last_landlord_action = z_batch[0]
    bb_obs = np.hstack((bb_obs, last_landlord_action))

    last_teammate_action = z_batch[1]
    bb_obs = np.hstack((bb_obs, last_teammate_action))

    """
    last_landlord_action = state['obs'][756:810]
    bb_obs = np.hstack((bb_obs, last_landlord_action))

    last_teammate_action = state['obs'][810:864]
    bb_obs = np.hstack((bb_obs, last_teammate_action))
    """

    landlord_num_cards_left = state['obs'][864:884]
    bb_obs = np.hstack((bb_obs, landlord_num_cards_left))

    teammate_num_cards_left = state['obs'][884:901]
    bb_obs = np.hstack((bb_obs, teammate_num_cards_left))

    one_hot_bomb_num = np.zeros(15)

    bomb_num = 0

    for item in actions:
        if 27467 <= item <= 27469:
            bomb_num = bomb_num + 1

    one_hot_bomb_num[bomb_num] = 1

    bb_obs = np.hstack((bb_obs, one_hot_bomb_num))

    x_batch = np.zeros([len(legal_actions), 484])

    for i in range(len(legal_actions)):
        x_batch[i] = np.hstack((bb_obs, legal_actions[i]))

    return x_batch.astype(np.float32), zz_batch_final.astype(np.float32)

def _get_obs_landlord_up(state):

    bb_obs = np.array([])
    legal_actions = np.array(list(state['legal_actions'].values()))

    z_batch = np.zeros([15, 54])

    role_actions = state['raw_obs']['trace']
    actions_str = []
    actions = []

    for item in role_actions:
        actions.append(ACTION_2_ID[item[1]])
        actions_str.append(item[1])


    rev_actions_str = list(actions_str)
    rev_actions_str.reverse()

    for i in range(len(rev_actions_str)):
        if i < 15:
            z_batch[i] = action_2_vector(rev_actions_str[i])
        else:
            break

    zz_bb = np.flip(z_batch, axis=0).reshape((5, 162))

    zz_batch_final = np.zeros([len(legal_actions), 5, 162])

    for i in range(len(legal_actions)):
        zz_batch_final[i] = zz_bb


    my_handcards = state['obs'][0:54]
    bb_obs = np.hstack((bb_obs, my_handcards))

    other_handcards = state['obs'][54:108]
    bb_obs = np.hstack((bb_obs, other_handcards))

    landlord_played_cards = state['obs'][648:702]
    bb_obs = np.hstack((bb_obs, landlord_played_cards))

    teammate_played_cards = state['obs'][702:756]
    bb_obs = np.hstack((bb_obs, teammate_played_cards))

    last_action = state['obs'][108:162]
    bb_obs = np.hstack((bb_obs, last_action))

    last_landlord_action = z_batch[1]
    bb_obs = np.hstack((bb_obs, last_landlord_action))

    last_teammate_action = z_batch[0]
    bb_obs = np.hstack((bb_obs, last_teammate_action))

    """
    last_landlord_action = state['obs'][756:810]
    bb_obs = np.hstack((bb_obs, last_landlord_action))

    last_teammate_action = state['obs'][810:864]
    bb_obs = np.hstack((bb_obs, last_teammate_action))
    """
    landlord_num_cards_left = state['obs'][864:884]
    bb_obs = np.hstack((bb_obs, landlord_num_cards_left))

    teammate_num_cards_left = state['obs'][884:901]
    bb_obs = np.hstack((bb_obs, teammate_num_cards_left))

    one_hot_bomb_num = np.zeros(15)

    bomb_num = 0


    actions = []

    role_actions = state['raw_obs']['trace']

    for item in actions:
        if 27467 <= item <= 27469:
            bomb_num = bomb_num + 1

    one_hot_bomb_num[bomb_num] = 1

    bb_obs = np.hstack((bb_obs, one_hot_bomb_num))

    x_batch = np.zeros([len(legal_actions), 484])

    for i in range(len(legal_actions)):
        x_batch[i] = np.hstack((bb_obs, legal_actions[i]))

    return x_batch.astype(np.float32), zz_batch_final.astype(np.float32)






