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

def vec_to_matrix(vec):
    max = vec[-1]
    vec = np.delete(vec, -1)
    min = vec[-1]
    king = np.array([min, 0, 0, 0,max,0,0,0])
    vec = np.delete(vec, -1)
    vec = np.append(vec,king)
    vv = vec.reshape(15, 4)
    vv = np.transpose(vv)
    return vv

def ws_matrix(st):
    matrix = []
    for item in st:
        matrix.append(vec_to_matrix(item))

    return np.array(matrix)


def _get_obs_landlord(state):

    bb_obs = np.hstack((state['obs'][0:162],state['obs'][648:790]))

    one_hot_bomb_num = np.zeros(20)

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

    one_hot_bomb_num[bomb_num] = 1

    bb_obs = np.hstack((bb_obs, one_hot_bomb_num))

    z_batch = np.zeros([15,54])

    rev_actions_str = list(actions_str)
    rev_actions_str.reverse()

    for i in range(len(rev_actions_str)):
        if i < 15:
            z_batch[i] = action_2_vector(rev_actions_str[i])
        else:
            break

    zz_bb = np.flip(z_batch, axis = 0).reshape((5,162))
    bb_matrix = ws_matrix(bb_obs.astype(np.float32).reshape([6, 54]))

    return bb_matrix.astype(np.float32).reshape([6,4,15]), zz_bb.astype(np.float32), [], []


def _get_obs_landlord_down(state):

    bb_obs = np.hstack((state['obs'][0:162], state['obs'][648:901]))

    one_hot_bomb_num = np.zeros(17)

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


    z_batch = np.zeros([15, 54])

    rev_actions_str = list(actions_str)
    rev_actions_str.reverse()

    for i in range(len(rev_actions_str)):
        if i < 15:
            z_batch[i] = action_2_vector(rev_actions_str[i])
        else:
            break

    zz_bb = np.flip(z_batch, axis=0).reshape((5, 162))
    bb_matrix = ws_matrix(bb_obs.astype(np.float32).reshape([8, 54]))


    return bb_matrix.astype(np.float32).reshape([8,4,15]), zz_bb.astype(np.float32),[],[]


def _get_obs_landlord_up(state):

    bb_obs = np.hstack((state['obs'][0:162], state['obs'][648:901]))

    one_hot_bomb_num = np.zeros(17)

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

    z_batch = np.zeros([15, 54])

    rev_actions_str = list(actions_str)
    rev_actions_str.reverse()

    for i in range(len(rev_actions_str)):
        if i < 15:
            z_batch[i] = action_2_vector(rev_actions_str[i])
        else:
            break

    zz_bb = np.flip(z_batch, axis=0).reshape((5, 162))
    bb_matrix = ws_matrix(bb_obs.astype(np.float32).reshape([8, 54]))

    return bb_matrix.astype(np.float32).reshape([8, 4,15]), zz_bb.astype(np.float32), [], []








