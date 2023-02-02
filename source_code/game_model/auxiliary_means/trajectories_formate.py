import numpy as np
from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION
from game_model.action_map.action_map import action_to_ab_id
import itertools


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

def action_2_matrix(action):
    action = ID_2_ACTION[action]
    dd = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12, 'B': 13, 'R': 14}
    if action == 'pass':
        return np.zeros([1,4,15],dtype=np.int)
    else:
        aa = np.zeros([1,4,15],dtype=np.int)

        for item in action:
            id = dd[item]
            for m in range(4):
                if aa[0][m][id] == 0:
                    aa[0][m][id] = 1
                    break

        return aa


def played(str):
    dd = {'3': 0,'4': 0,'5': 0,'6': 0,'7': 0,'8': 0,'9': 0,'T': 0,'J': 0,'Q': 0,'K': 0,'A': 0,'2': 0,'B': 0,'R': 0}
    tmp = ''
    tmp = tmp + str[0]
    tmp = tmp + str[1]
    tmp = tmp + str[2]
    if len(tmp) == 0:
        return np.zeros([1,54])
    else:
        for item in tmp:
            dd[item] = dd[item] + 1
        kk = list(dd.values())
        bb = dd['B']
        rr = dd['R']
        kk.pop()
        kk.pop()
        tt = []
        for item in kk:
            tm = [0, 0, 0, 0]
            for i in range(item):
                tm[i] = 1
            tt = tt + tm
        tt.append(bb)
        tt.append(rr)
        return np.array(tt).reshape(1,54).tolist()

def obs_matrix(obs,role_id):
    st = []
    ac = []

    if role_id == 0:
        st.append(obs[0])
        st.append(obs[1])
        st.append(obs[2])
        st.append(obs[12])
        st.append(obs[13])
        st.append(obs[14])

        ac.append(obs[11])
        ac.append(obs[10])
        ac.append(obs[9])
        ac.append(obs[8])
        ac.append(obs[7])
        ac.append(obs[6])
        ac.append(obs[5])
        ac.append(obs[4])
        ac.append(obs[3])

        st_obs = np.zeros((6, 4, 15), dtype=int)
        ac_obs = np.zeros((9, 4, 15), dtype=int)

        for i in range(len(st)):
            st_obs[i] = vec_to_matrix(st[i])

        for i in range(len(ac)):
            ac_obs[i] = vec_to_matrix(ac[i])

        ac_obs = ac_obs.reshape(3, 180)
    elif role_id == 1:
        st.append(obs[0])
        st.append(obs[1])
        st.append(obs[2])
        st.append(obs[12])
        st.append(obs[13])
        #st.append(obs[14])
        #st.append(obs[15])
        st.append(obs[16])

        ac.append(obs[11])
        ac.append(obs[10])
        ac.append(obs[9])
        ac.append(obs[8])
        ac.append(obs[7])
        ac.append(obs[6])
        ac.append(obs[5])
        ac.append(obs[4])
        ac.append(obs[3])

        st_obs = np.zeros((6, 4, 15), dtype=int)
        ac_obs = np.zeros((9, 4, 15), dtype=int)

        for i in range(len(st)):
            st_obs[i] = vec_to_matrix(st[i])

        for i in range(len(ac)):
            ac_obs[i] = vec_to_matrix(ac[i])

        ac_obs = ac_obs.reshape(3, 180)

    else:

        st.append(obs[0])
        st.append(obs[1])
        st.append(obs[2])
        st.append(obs[12])
        st.append(obs[13])
        # st.append(obs[14])
        # st.append(obs[15])
        st.append(obs[16])

        ac.append(obs[11])
        ac.append(obs[10])
        ac.append(obs[9])
        ac.append(obs[8])
        ac.append(obs[7])
        ac.append(obs[6])
        ac.append(obs[5])
        ac.append(obs[4])
        ac.append(obs[3])

        st_obs = np.zeros((6, 4, 15), dtype=int)
        ac_obs = np.zeros((9, 4, 15), dtype=int)

        for i in range(len(st)):
            st_obs[i] = vec_to_matrix(st[i])

        for i in range(len(ac)):
            ac_obs[i] = vec_to_matrix(ac[i])

        ac_obs = ac_obs.reshape(3, 180)

    return st_obs,ac_obs



def state_formate(state):
    state = state['obs'][0:756]
    state = state.reshape(14, 54)

    return state



def data_store(data00,data01,data02,nexs,kicker_datas,kicker_datas01,kicker_datas02,payoff,gamma,Lmem,Pmem01,Pmem02,kmem,kmem01,kmem02):
    vv = []
    if payoff[0] == 1:
        score = 1
    else:
        score = 0.001

    length = len(data00)
    vv.append(float(score))

    for m in range(1, length):
        score = score * gamma
        vv.append(score)

    vv.reverse()


    if payoff[0] == 0:
        score01 = 1
        score02 = 1
    else:
        score01 = 0.001
        score02 = 0.001

    vv_d = []
    length01 = int(len(data01))
    vv_d.append(float(score01))

    for m in range(1, length01):
        score01 = score01 * gamma
        vv_d.append(score01)

    vv_d.reverse()

    vv_u = []
    length02 = int(len(data02))
    vv_u.append(float(score02))

    for m in range(1, length02):
        score02 = score02 * gamma
        vv_u.append(score02)

    vv_u.reverse()

    for i in range(length):
        data00[i]['val'] = vv[i]

        if i >= length02:
            st_p = nexs[2][0]
        else:
            st_p = data02[i]['obs'][0]

        if i >= length01:
            st_p_p = nexs[1][0]
        else:
            st_p_p = data01[i]['obs'][0]

        com_state00 = np.concatenate((st_p_p, st_p), axis=0).reshape(2, 4, 15)
        com_state00 = np.concatenate((com_state00, data00[i]['obs']), axis=0)
        data00[i]['combine'] = com_state00

        if kicker_datas[i] != 0:
            for m in range(len(kicker_datas[i]['k_action'])):
                me = {}
                me['k_state'] = kicker_datas[i]['k_state'][m].squeeze()
                me['k_action'] = kicker_datas[i]['k_action'][m]
                me['k_seq'] = kicker_datas[i]['k_seq'][m].squeeze()
                me['k_prob'] = kicker_datas[i]['k_prob'][m]
                me['k_reward'] = vv[i]
                kmem.addmemory(me)
        Lmem.addmemory(data00[i])

    for i in range(length01):
        pea_datas = []
        data01[i]['val'] = vv_d[i]

        if i + 1 >= length:
            st_p = nexs[0][0]
        else:
            st_p = data00[i + 1]['obs'][0]

        if i >= length02:
            st_p_p = nexs[2][0]
        else:
            st_p_p = data02[i]['obs'][0]

        #com_state01 = data01[i]['obs']
        com_state01 = np.concatenate((st_p_p, st_p), axis=0).reshape(2,4,15)
        com_state01 = np.concatenate((com_state01, data01[i]['obs']), axis=0)
        data01[i]['combine'] = com_state01

        pea_datas.append(data01[i])

        if kicker_datas01[i] != 0:
            for m in range(len(kicker_datas01[i]['k_action'])):
                me = {}
                me['k_state'] = kicker_datas01[i]['k_state'][m].squeeze()
                me['k_action'] = kicker_datas01[i]['k_action'][m]
                me['k_seq'] = kicker_datas01[i]['k_seq'][m].squeeze()
                me['k_prob'] = kicker_datas01[i]['k_prob'][m]
                me['k_reward'] = vv_d[i]
                kmem01.addmemory(me)

        if i < length02:
            data02[i]['val'] = vv_u[i]

            if i + 1 >= length01:
                st_p = nexs[1][0]
            else:
                st_p = data01[i + 1]['obs'][0]

            if i + 1 >= length:
                st_p_p = nexs[0][0]
            else:
                st_p_p = data00[i + 1]['obs'][0]

            com_state02 = np.concatenate((st_p_p, st_p), axis=0).reshape(2, 4, 15)
            com_state02 = np.concatenate((com_state02, data02[i]['obs']), axis=0)
            data02[i]['combine'] = com_state02

            if kicker_datas02[i] != 0:
                for m in range(len(kicker_datas02[i]['k_action'])):
                    me = {}
                    me['k_state'] = kicker_datas02[i]['k_state'][m].squeeze()
                    me['k_action'] = kicker_datas02[i]['k_action'][m]
                    me['k_seq'] = kicker_datas02[i]['k_seq'][m].squeeze()
                    me['k_prob'] = kicker_datas02[i]['k_prob'][m]
                    me['k_reward'] = vv_u[i]
                    kmem02.addmemory(me)
            pea_datas.append(data02[i])

        else:
            """
            if length01 - length02 > 1:
                print("==============",length,length01,length02)
            """
            pea_datas.append(data02[-1])

        Pmem01.addmemory(pea_datas)

def state_formate_matrix(state,role_id):

    if role_id == 0:
        one_hot_bomb_num = np.zeros(20)
    else:
        one_hot_bomb_num = np.zeros(17)

    role_actions = state['raw_obs']['trace']

    actions = []
    for item in role_actions:
        actions.append(ACTION_2_ID[item[1]])

    bomb_num = 0
    for item in actions:
        if 27467 <= item <= 27469:
            bomb_num = bomb_num + 1

    one_hot_bomb_num[bomb_num] = 1

    obs = np.concatenate((state['obs'], one_hot_bomb_num))

    if role_id == 0:
        playered_cards = played(state['raw_obs']["played_cards"])

        state = obs.reshape(15, 54)

        state = np.concatenate((state, playered_cards), axis=0)
        state, action_seq = obs_matrix(state,role_id)
        # print(state)
    else:

        playered_cards = played(state['raw_obs']["played_cards"])

        state = obs.reshape(17, 54)

        state = np.concatenate((state, playered_cards), axis=0)
        state, action_seq = obs_matrix(state,role_id)
        # print(state)

    return state,action_seq


def data_store_pp01(data00, kicker_datas, data01, kicker_datas01, data02, kicker_datas02, nexs, payoff, gamma, Lmem, Pmem,
                  kmem, kmem01, kmem02):
    vv = []
    if payoff[0] == 1:
        score = 1
    else:
        score = 0.001

    length = len(data00)
    vv.append(float(score))

    for m in range(1, length):
        score = score * gamma
        vv.append(score)
    vv.reverse()

    for i in range(length):

        if i < len(data01) and i < len(data02):
            z_d01 = data01[i]['obs'][0].reshape(1, 4, 15)
            z_d02 = data02[i]['obs'][0].reshape(1, 4, 15)
            z_d00 = data00[i]['obs'].reshape(6, 4, 15)

            # combine_state = np.concatenate((tr00,z_d01), axis=0)
            combine_state = np.concatenate((z_d01, z_d02), axis=0)
            combine_state = np.concatenate((combine_state, z_d00), axis=0)

            data00[i]['combine'] = combine_state
            data00[i]['val'] = vv[i]

        elif i < len(data01) and i == len(data02):
            z_d01 = data01[i]['obs'][0].reshape(1, 4, 15)
            z_nexs_2 = nexs[2].reshape(1, 4, 15)
            z_d00 = data00[i]['obs'].reshape(6, 4, 15)

            # combine_state = np.concatenate((tr00, z_d01), axis=0)
            combine_state = np.concatenate((z_d01, z_nexs_2), axis=0)
            combine_state = np.concatenate((combine_state, z_d00), axis=0)

            data00[i]['combine'] = combine_state
            data00[i]['val'] = vv[i]


        elif i == len(data01) and i == len(data02):
            z_nexs_1 = nexs[1].reshape(1, 4, 15)
            z_nexs_2 = nexs[2].reshape(1, 4, 15)
            z_d00 = data00[i]['obs'].reshape(6, 4, 15)

            # combine_state = np.concatenate((tr00, z_nexs_1), axis=0)
            combine_state = np.concatenate((z_nexs_1, z_nexs_2), axis=0)
            combine_state = np.concatenate((combine_state, z_d00), axis=0)

            data00[i]['combine'] = combine_state
            data00[i]['val'] = vv[i]

        if kicker_datas[i] != 0:
            for m in range(len(kicker_datas[i]['k_action'])):
                me = {}
                me['k_state'] = kicker_datas[i]['k_state'][m].squeeze()
                me['k_action'] = kicker_datas[i]['k_action'][m]
                me['k_seq'] = kicker_datas[i]['k_seq'][m].squeeze()
                me['k_prob'] = kicker_datas[i]['k_prob'][m]
                me['k_reward'] = vv[i]
                kmem.addmemory(me)
        # print("data===::",data00[i]['val'])

        Lmem.addmemory(data00[i])

    if payoff[0] == 0:
        score01 = 1
        score02 = 1
    else:
        score01 = 0.001
        score02 = 0.001

    vv_d = []
    length01 = int(len(data01))
    vv_d.append(float(score01))

    for m in range(1, length01):
        score01 = score01 * gamma
        vv_d.append(score01)

    vv_d.reverse()

    vv_u = []
    length02 = int(len(data02))
    vv_u.append(float(score02))

    for m in range(1, length02):
        score02 = score02 * gamma
        vv_u.append(score02)

    vv_u.reverse()

    for i in range(length01):
        if i < length02:

            datas = []
            if i + 1 == len(data00):
                z_d02 = data02[i]['obs'][0].reshape(1, 4, 15)
                z_nexs_0 = nexs[0].reshape(1, 4, 15)
                z_d01 = data01[i]['obs'].reshape(6, 4, 15)

                # combine_state01 =np.concatenate((tr01, z_d02), axis=0)
                combine_state01 = np.concatenate((z_d02, z_nexs_0), axis=0)
                combine_state01 = np.concatenate((combine_state01, z_d01), axis=0)


            else:
                z_d02 = data02[i]['obs'][0].reshape(1, 4, 15)
                z_d00 = data00[i + 1]['obs'][0].reshape(1, 4, 15)
                z_d01 = data01[i]['obs'].reshape(6, 4, 15)

                # combine_state01 = np.concatenate((tr01, z_d02), axis=0)
                combine_state01 = np.concatenate((z_d02, z_d00), axis=0)
                combine_state01 = np.concatenate((combine_state01, z_d01), axis=0)

            data01[i]['val'] = vv_d[i]
            data01[i]['combine'] = combine_state01

            if i + 1 < len(data00) and i + 1 == len(data01):
                z_d00 = data00[i + 1]['obs'][0].reshape(1, 4, 15)
                z_nexs_0 = nexs[0].reshape(1, 4, 15)
                z_d02 = data02[i]['obs'].reshape(6, 4, 15)

                # combine_state02 = np.concatenate((tr02, z_d00), axis=0)
                combine_state02 = np.concatenate((z_d00, z_nexs_0), axis=0)
                combine_state02 = np.concatenate((combine_state02, z_d02), axis=0)

            elif i + 1 == len(data00) and i + 1 == len(data01):
                z_nexs_0 = nexs[0].reshape(1, 4, 15)
                z_nexs_1 = nexs[1].reshape(1, 4, 15)
                z_d02 = data02[i]['obs'].reshape(6, 4, 15)

                # combine_state02 = np.concatenate((tr02, z_nexs_0), axis=0)
                combine_state02 = np.concatenate((z_nexs_0, z_nexs_1), axis=0)
                combine_state02 = np.concatenate((combine_state02, z_d02), axis=0)

            else:
                z_d00 = data00[i + 1]['obs'][0].reshape(1, 4, 15)
                z_d01 = data01[i + 1]['obs'][0].reshape(1, 4, 15)
                z_d02 = data02[i]['obs'].reshape(6, 4, 15)

                # combine_state02 = np.concatenate((tr02, z_d00), axis=0)
                combine_state02 = np.concatenate((z_d00, z_d01), axis=0)
                combine_state02 = np.concatenate((combine_state02, z_d02), axis=0)

            data02[i]['val'] = vv_u[i]
            data02[i]['combine'] = combine_state02

            if kicker_datas01[i] != 0:
                for m in range(len(kicker_datas01[i]['k_action'])):
                    me = {}
                    me['k_state'] = kicker_datas01[i]['k_state'][m].squeeze()
                    me['k_action'] = kicker_datas01[i]['k_action'][m]
                    me['k_seq'] = kicker_datas01[i]['k_seq'][m].squeeze()
                    me['k_prob'] = kicker_datas01[i]['k_prob'][m]
                    me['k_reward'] = vv_d[i]
                    kmem01.addmemory(me)

            if kicker_datas02[i] != 0:

                for m in range(len(kicker_datas02[i]['k_action'])):
                    me = {}
                    me['k_state'] = kicker_datas02[i]['k_state'][m].squeeze()
                    me['k_action'] = kicker_datas02[i]['k_action'][m]
                    me['k_seq'] = kicker_datas02[i]['k_seq'][m].squeeze()
                    me['k_prob'] = kicker_datas02[i]['k_prob'][m]
                    me['k_reward'] = vv_u[i]
                    kmem02.addmemory(me)

            datas.append(data01[i])
            datas.append(data02[i])

            Pmem.addmemory(datas)
        else:

            datas = []

            z_nexs_2 = nexs[2].reshape(1, 4, 15)
            z_nexs_0 = nexs[0].reshape(1, 4, 15)
            z_d01 = data01[i]['obs'].reshape(6, 4, 15)

            # combine_state01 = np.concatenate((tr01, z_nexs_2), axis=0)
            combine_state01 = np.concatenate((z_nexs_2, z_nexs_0), axis=0)
            combine_state01 = np.concatenate((combine_state01, z_d01), axis=0)

            data01[i]['val'] = vv_d[i]
            data01[i]['combine'] = combine_state01

            if kicker_datas01[i] != 0:
                for m in range(len(kicker_datas01[i]['k_action'])):
                    me = {}
                    me['k_state'] = kicker_datas01[i]['k_state'][m].squeeze()
                    me['k_action'] = kicker_datas01[i]['k_action'][m]
                    me['k_seq'] = kicker_datas01[i]['k_seq'][m].squeeze()
                    me['k_prob'] = kicker_datas01[i]['k_prob'][m]
                    me['k_reward'] = vv_d[i]
                    kmem01.addmemory(me)

            datas.append(data01[i])
            datas.append(data02[i - 1])
            Pmem.addmemory(datas)


def score_count(trajectories,payoff):
    score = payoff
    buff = []
    l_0 = len(trajectories[0])
    l_1 = len(trajectories[1])
    l_2 = len(trajectories[2])

    for i in range(1,l_0,2):
        buff.append(trajectories[0][i])

    for i in range(1,l_1,2):
        buff.append(trajectories[1][i])

    for i in range(1,l_2,2):
        buff.append(trajectories[2][i])


    for item in buff:
        if (item >=27457 and item <=27470):
            score = score * 2
    return score