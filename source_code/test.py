# coding=utf-8
import rlcard
import numpy as np
import datetime
import time
from op_agent.random_agent import RandomAgent
from op_agent.min_agent import MinAgent
from op_agent.rhcp_shang_agent import RhcpShangAgent
# from op_agent.zeta_dou import ZetaAgent
from op_agent.zeta_dou_com import ZetaAgentCom
from op_agent.nv_dou import NvAgent
from op_agent.sl_agent import Slagent
from op_agent.dou_zero import ZeroAgent
from game_model.auxiliary_means.trajectories_formate import score_count
import sys
import csv
evaluate_num = 10000

if __name__ == "__main__":
    m_s = datetime.datetime.now()
    random_agent = RandomAgent(309)
    min_agent = MinAgent(309)
    """
    zeta_land = ZetaAgent(309,'l')
    zeta_pea_d = ZetaAgent(309,'d')
    zeta_pea_u = ZetaAgent(309,'u')
    """

    nn = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))

    cs_path = "out_put/test/"+nn+".csv"

    with open(cs_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["eposide", "label", "l_win_rate", "peas_win_rate", "l_scores","peas_scores"])

    nv_land = NvAgent(309, 'l')
    nv_pea_d = NvAgent(309, 'd')
    nv_pea_u = NvAgent(309, 'u')

    zeta_land_com = ZetaAgentCom(309, 'l')
    zeta_pea_d_com = ZetaAgentCom(309, 'd')
    zeta_pea_u_com = ZetaAgentCom(309, 'u')

    sl_land = ZeroAgent('landlord', 'op_agent/sl/landlord.ckpt')
    sl_pea_d = ZeroAgent('landlord_down', 'op_agent/sl/landlord_down.ckpt')
    sl_pea_u = ZeroAgent('landlord_up', 'op_agent/sl/landlord_up.ckpt')

    zero_land_w = ZeroAgent('landlord', 'op_agent/douzero_WP/landlord.ckpt')
    zero_pea_d_w = ZeroAgent('landlord_down', 'op_agent/douzero_WP/landlord_down.ckpt')
    zero_pea_u_w = ZeroAgent('landlord_up', 'op_agent/douzero_WP/landlord_up.ckpt')

    zero_land_a = ZeroAgent('landlord', 'op_agent/douzero_ADP/landlord.ckpt')
    zero_pea_d_a = ZeroAgent('landlord_down', 'op_agent/douzero_ADP/landlord_down.ckpt')
    zero_pea_u_a = ZeroAgent('landlord_up', 'op_agent/douzero_ADP/landlord_up.ckpt')

    rhcp_shang_agent = RhcpShangAgent(309)

    eval_env = rlcard.make('doudizhu')
    rhcp_shang_agent.num = 0

    rrs = []

    agent_lans = [zeta_land_com, nv_pea_d, nv_pea_u]
    agent_peas = [nv_land, zeta_pea_d_com, zeta_pea_u_com]

    label = 'Zeta vs nv'

    for mm in range(20):
        reward = [0, 0, 0]
        l_scores = 0
        l_win_rate = 0

        eval_env.set_agents(agent_lans)

        for i in range(1, evaluate_num + 1):
            rhcp_shang_agent.num = 0
            nv_land.update_model()
            nv_pea_d.update_model()
            nv_pea_u.update_model()

            guiji, pay_offs = eval_env.run(is_training=False)
            reward = reward + pay_offs
            if pay_offs[0] == 0:
                l_scores = l_scores + score_count(guiji, -2)
            else:
                l_scores = l_scores + score_count(guiji, 2)

            print("the reward is ===", i, "****===", reward)

        l_win_rate = reward[0]
        print("mm::",mm,"地主总收益为::", reward[0],l_scores)

        reward = [0, 0, 0]
        p_socres = 0
        pea_win_rate = 0

        eval_env.set_agents(agent_peas)
        for i in range(1, evaluate_num + 1):
            rhcp_shang_agent.num = 0
            nv_land.update_model()
            nv_pea_d.update_model()
            nv_pea_u.update_model()

            guiji, pay_offs = eval_env.run(is_training=False)
            reward = reward + pay_offs

            if pay_offs[1] == 0:
                p_socres = p_socres + score_count(guiji, -1)
            else:
                p_socres = p_socres + score_count(guiji, 1)

            print("the reward is ===", i, "****===", reward)

        pea_win_rate = reward[1]

        with open(cs_path, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [mm,label,l_win_rate,pea_win_rate,l_scores,p_socres*2]
            csv_write.writerow(data_row)
        print("mm::",mm,"农民总收益为::", reward[1],p_socres)



    m_e = datetime.datetime.now()
    sec = m_e - m_s
    print("执行完成,耗时", sec.seconds)
