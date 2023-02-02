import numpy as np
import rlcard
from itertools import chain
"""
from op_agent.random_agent import RandomAgent
from op_agent.rhcp_shang_agent import RhcpShangAgent
from op_agent.zeta_dou import ZetaAgent
from op_agent.min_agent import MinAgent
"""
from net_learner.zeta_agent import Tagent
from game_model.auxiliary_means.trajectories_formate import data_store,data_store_pp01
from multiprocessing import shared_memory
from game_model.auxiliary_means.mem_class import Memory, Memseq, RMemory
from game_model.auxiliary_means.catalogue import create_catalogue, create_csv, add_data_csv
from game_model.auxiliary_means.trajectories_formate import state_formate, score_count, state_formate_matrix
from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION
from net_learner.zeta_model import PNet,PKNet
from optim_equ.my_optim import SharedAdam
from game_model.action_map.action_map import action_to_ab_id, kdata_ana
#from op_agent.rhcp_shang_agent import RhcpShangAgent
from rlcard.utils.utils import *
import torch
import datetime
import random
import os
import csv
import threading
import time
import multiprocessing
from op_agent.nv_dou import NvAgent
from op_agent.sl_agent import Slagent
from op_agent.dou_zero import ZeroAgent

dd = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11,'2': 12,'B': 13, 'R': 14}

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def zhuan_huan(st, action):
    for item in action:
        item_id = dd[item]

        for i in range(3, -1, -1):
            if st[i][item_id] == 1:
                st[i][item_id] = 0
                break

    return st

def worker(rank, lnet,lnet_d,lnet_u,kicker,kicker_d,kicker_u, rmem, rmem01, rmem02, kmem, kmem01, kmem02,lock,dd=0):
    env = rlcard.make('doudizhu')


    d_l_ag = Tagent(0, lnet, kicker, 309)
    d_d_ag = Tagent(1, lnet_d,kicker_d, 309)
    d_u_ag = Tagent(2, lnet_u,kicker_u, 309)

    d_agents = [d_l_ag, d_d_ag, d_u_ag]
    for mm in range(0, 200000000):
        if mm % 5000==0:
            print("worker:",rank,"noise sample",rmem.get_size(),"::",kmem.get_size(),"::",kmem01.get_size(),"::",kmem02.get_size())

        env.set_agents(d_agents)
        trajectories, payoffs = env.run(is_training=True)

        st00, _ = state_formate_matrix(env.get_state(0),0)
        st01, _ = state_formate_matrix(env.get_state(1), 1)
        st02, _ = state_formate_matrix(env.get_state(2), 2)

        nexs = [st00,st01,st02]

        lock.acquire()
        data_store(d_l_ag.datas, d_d_ag.datas, d_u_ag.datas,nexs,d_l_ag.kicker_datas, d_d_ag.kicker_datas, d_u_ag.kicker_datas, payoffs, 0.9, rmem, rmem01, rmem02, kmem, kmem01, kmem02)
        d_l_ag.update_data()
        d_d_ag.update_data()
        d_u_ag.update_data()
        lock.release()

        #print(mm,"=======",rmem.get_size(),kmem.get_size())


def train(rank, lnet, lnet_d, lnet_u, kicker, kicker_d, kicker_u,  pop, pop_d, pop_u, rmem, rmem01, rmem02, kmem, kmem01, kmem02, lock, dd=0):

    env = rlcard.make('doudizhu')
    kic_access = 'neural'
    train_model = 'all'
    save = 'trainning'
    path = ''
    eval_num = 5000
    every = 5000
    alpha_lan = 0
    alpha_pea = 0

    # rhcp_shang_agent = RhcpShangAgent(309)

    nv_land = NvAgent(309, 'l')
    nv_pea_d = NvAgent(309, 'd')
    nv_pea_u = NvAgent(309, 'u')

    sl_land = ZeroAgent('landlord', 'op_agent/sl/landlord.ckpt')
    sl_pea_d = ZeroAgent('landlord_down', 'op_agent/sl/landlord_down.ckpt')
    sl_pea_u = ZeroAgent('landlord_up', 'op_agent/sl/landlord_up.ckpt')

    zero_land_w = ZeroAgent('landlord', 'op_agent/douzero_WP/landlord.ckpt')
    zero_pea_d_w = ZeroAgent('landlord_down', 'op_agent/douzero_WP/landlord_down.ckpt')
    zero_pea_u_w = ZeroAgent('landlord_up', 'op_agent/douzero_WP/landlord_up.ckpt')

    zero_land_a = ZeroAgent('landlord', 'op_agent/douzero_ADP/landlord.ckpt')
    zero_pea_d_a = ZeroAgent('landlord_down', 'op_agent/douzero_ADP/landlord_down.ckpt')
    zero_pea_u_a = ZeroAgent('landlord_up', 'op_agent/douzero_ADP/landlord_up.ckpt')


    main_batch_size = 512
    kic_batch_size = 128
    
    l_ag = Tagent(0, lnet, kicker, 309)
    d_ag = Tagent(1, lnet_d,kicker_d,309)
    u_ag = Tagent(2, lnet_u,kicker_u,309)

    if  save == 'trainning':
        nowTime = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
        path = "out_put/train/" + nowTime
        os.makedirs(path)
        with open(path + "/performance.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['eposide','wl', 'wd','wu','l_score','p_score','lac_loss', 'lcr_loss', 'l_loss', 'pac_d_loss',
                 'pcr_d_loss', 'p_d_loss', 'pac_u_loss', 'pcr_u_loss', 'p_u_loss', 'k_loss', 'k_d_loss', 'k_u_loss'])

    lac_loss, lcr_loss, l_loss  = 0, 0, 0
    pac_d_loss, pcr_d_loss, p_d_loss = 0, 0, 0
    pac_u_loss, pcr_u_loss, p_u_loss = 0, 0, 0
    k_loss, k_d_loss, k_u_loss = 0,0,0

    max_land = 0
    max_peas = 0
    print("starting trainning==========")
    for mm in range(0, 200000000):
        #time.sleep(0.3)
        l_ag.update_data(kic_access)
        d_ag.update_data(kic_access)
        u_ag.update_data(kic_access)

        if mm % every == 0:

            l_scores = 0
            p_socres = 0

            lnet.remove_noise()
            lnet_d.remove_noise()
            lnet_u.remove_noise()

            kicker.remove_noise()
            kicker_d.remove_noise()
            kicker_u.remove_noise()

            env.set_agents([l_ag,sl_pea_d,sl_pea_u])
            w_l = 0
            w_d = 0
            w_u = 0

            for i in range(eval_num):
                trajectories, payoffs = env.run(is_training=False)
                w_l = w_l + payoffs[0]

                if payoffs[0] == 0:
                    l_scores = l_scores + score_count(trajectories, -2)
                else:
                    l_scores = l_scores + score_count(trajectories, 2)

                print("land::::^^^^^^::::::::::^^^^^^^:::::",i)

            env.set_agents([sl_land,d_ag,u_ag])

            for i in range(eval_num):
                trajectories, payoffs = env.run(is_training=False)

                if len(trajectories[1]) == len(trajectories[2]):
                    w_u = w_u + payoffs[1]
                else:
                    w_d = w_d + payoffs[1]

                if payoffs[1] == 0:
                    p_socres = p_socres + score_count(trajectories, -1)
                else:
                    p_socres = p_socres + score_count(trajectories, 1)

                print("pea::::^^^^^^::::::::::^^^^^^^:::::", i)

            if w_l > max_land:
                max_land = w_l
                torch.save(lnet.state_dict(), path + "/max_nework.pth")
                torch.save(kicker.state_dict(), path + "/max_knework.pth")

            if (w_d+w_u) > max_peas:
                max_peas = w_d+w_u
                torch.save(lnet_d.state_dict(), path + "/max_nework01.pth")
                torch.save(lnet_u.state_dict(), path + "/max_nework02.pth")
                torch.save(kicker_d.state_dict(), path + "/max_knework01.pth")
                torch.save(kicker_u.state_dict(), path + "/max_knework02.pth")

            torch.save(lnet.state_dict(),  path + "/nework.pth")
            torch.save(lnet_d.state_dict(),path + "/nework01.pth")
            torch.save(lnet_u.state_dict(),path + "/nework02.pth")

            torch.save(kicker.state_dict(),path + "/knework.pth")
            torch.save(kicker_d.state_dict(),path + "/knework01.pth")
            torch.save(kicker_u.state_dict(),path +"/knework02.pth")

            if save == 'trainning':
                with open(path + "/performance.csv", "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([mm / every, w_l / eval_num, w_d / eval_num, w_u / eval_num, l_scores, p_socres, lac_loss, lcr_loss, l_loss,
                         pac_d_loss, pcr_d_loss, p_d_loss, pac_u_loss, pcr_u_loss, p_u_loss, k_loss, k_d_loss,
                         k_u_loss])

            print("testing is complete strart sample the noise==")

            lnet.sample_noise_n()
            lnet_d.sample_noise_n()
            lnet_u.sample_noise_n()

            kicker.sample_noise_n()
            kicker_d.sample_noise_n()
            kicker_u.sample_noise_n()

        if mm % every == 0 or mm == 1:
            m_s = datetime.datetime.now()

        #time.sleep(0.01)
        lock.acquire()  # 获得锁
        bs, bse, ba, bv, bp, com = rmem.sample(main_batch_size)
        bs01, bse01, ba01, bv01, bp01, bs02, bse02, ba02, bv02, bp02, com01, com02 = rmem01.sample_peas(main_batch_size)
        lock.release()  # 解放锁
        """
        obs, ac_seq, ab_legal, st_action, action, prob, st_batch, ac_seq_batch, val, com00 = rmem.sample(batch_size)
        obs01, ac_seq01, ab_legal01, st_action01, action01, prob01, st_batch01, ac_seq_batch01, val01, obs02, ac_seq02, ab_legal02, st_action02, action02, prob02, st_batch02, ac_seq_batch02, val02, com01,com02= rmem01.pea_sample(batch_size)
        #obs02, ac_seq02, ab_legal02, st_action02, action02, prob02, st_batch02, ac_seq_batch02, val02 = rmem02.sample(batch_size)
        """

        lac_loss, lcr_loss, l_loss = lnet.loss_func(bs, bse, ba, bv, bp, com)
        pac_d_loss, pcr_d_loss, p_d_loss = lnet_d.loss_func(bs01, bse01, ba01, bv01, bp01, com01)
        pac_u_loss, pcr_u_loss, p_u_loss = lnet_u.loss_func(bs02, bse02, ba02, bv02, bp02, com02)


        if  kmem.get_size()> 0:
            kbs, kbse, kba, kbv, kbp = kmem.ksample(kic_batch_size)
            k_loss = kicker.loss_func(kbs, kbse, kba, kbv, kbp, alpha_lan)
            z_loss = l_loss + k_loss

            pop.zero_grad()
            z_loss.backward()
            """
            ensure_shared_grads(T_lnet, lnet)
            ensure_shared_grads(T_kicker, kicker)
            """
            pop.step()
            lac_loss, lcr_loss, l_loss = lac_loss.item(), lcr_loss.item(), l_loss.item()
            k_loss = k_loss.item()
        else:
            z_loss = l_loss
            pop.zero_grad()
            z_loss.backward()
            """
            ensure_shared_grads(T_lnet, lnet)
            ensure_shared_grads(T_kicker, kicker)
            """
            pop.step()
            lac_loss, lcr_loss, l_loss = lac_loss.item(), lcr_loss.item(), l_loss.item()

        if kmem01.get_size() > 0 and kmem02.get_size() > 0:

            kbs01, kbse01, kba01, kbv01, kbp01 = kmem01.ksample(kic_batch_size)
            k_d_loss = kicker_d.loss_func(kbs01, kbse01, kba01, kbv01, kbp01, alpha_pea)

            kbs02, kbse02, kba02, kbv02, kbp02 = kmem02.ksample(kic_batch_size)
            k_u_loss = kicker_u.loss_func(kbs02, kbse02, kba02, kbv02, kbp02, alpha_pea)

            z_d_loss = p_d_loss + k_d_loss
            z_u_loss = p_u_loss + k_u_loss

            pop_d.zero_grad()
            z_d_loss.backward()
            """
            ensure_shared_grads(T_lnet_d, lnet_d)
            ensure_shared_grads(T_kicker_d, kicker_d)
            """
            pop_d.step()
            pac_d_loss, pcr_d_loss, p_d_loss = pac_d_loss.item(), pcr_d_loss.item(), p_d_loss.item()

            pop_u.zero_grad()
            z_u_loss.backward()
            """
            ensure_shared_grads(T_lnet_u, lnet_u)
            ensure_shared_grads(T_kicker_u, kicker_u)
            """
            pop_u.step()
            pac_u_loss, pcr_u_loss, p_u_loss = pac_u_loss.item(), pcr_u_loss.item(), p_u_loss.item()

            k_d_loss = k_d_loss.item()
            k_u_loss = k_u_loss.item()
        else:

            z_d_loss = p_d_loss
            z_u_loss = p_u_loss

            pop_d.zero_grad()
            z_d_loss.backward()
            """
            ensure_shared_grads(T_lnet_d, lnet_d)
            ensure_shared_grads(T_kicker_d, kicker_d)
            """
            pop_d.step()
            pac_d_loss, pcr_d_loss, p_d_loss = pac_d_loss.item(), pcr_d_loss.item(), p_d_loss.item()

            pop_u.zero_grad()
            z_u_loss.backward()
            """
            ensure_shared_grads(T_lnet_u, lnet_u)
            ensure_shared_grads(T_kicker_u, kicker_u)
            """
            pop_u.step()
            pac_u_loss, pcr_u_loss, p_u_loss = pac_u_loss.item(), pcr_u_loss.item(), p_u_loss.item()

        m_e = datetime.datetime.now()
        sec = m_e - m_s

        #print("mm is::",mm,":::",rmem.get_size(),kmem.get_size(),rmem01.get_size(),kmem01.get_size(),kmem02.get_size(),'sec:',sec.seconds)
        print("mm is::", mm, ":::",'max_land:',max_land,'max_peas:',max_peas,'sec:', sec.seconds)