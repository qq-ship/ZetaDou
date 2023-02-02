#coding=utf-8
import rlcard
import numpy as np
import datetime
from op_agent.random_agent import RandomAgent
from op_agent.min_agent  import MinAgent
from op_agent.rhcp_shang_agent import RhcpShangAgent

#from op_agent.zeta_dou import ZetaAgent
from op_agent.zeta_dou_com import ZetaAgentCom
from op_agent.nv_dou import NvAgent
from op_agent.sl_agent import Slagent
from op_agent.dou_zero import ZeroAgent

import sys
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

    nv_land = NvAgent(309,'l')
    nv_pea_d = NvAgent(309, 'd')
    nv_pea_u = NvAgent(309, 'u')

    zeta_land_com = ZetaAgentCom(309,'l')
    zeta_pea_d_com = ZetaAgentCom(309,'d')
    zeta_pea_u_com = ZetaAgentCom(309,'u')


    sl_land = ZeroAgent('landlord','op_agent/sl/landlord.ckpt')
    sl_pea_d = ZeroAgent('landlord_down','op_agent/sl/landlord_down.ckpt')
    sl_pea_u = ZeroAgent('landlord_up','op_agent/sl/landlord_up.ckpt')

    zero_land_w = ZeroAgent('landlord', 'op_agent/douzero_WP/landlord.ckpt')
    zero_pea_d_w = ZeroAgent('landlord_down', 'op_agent/douzero_WP/landlord_down.ckpt')
    zero_pea_u_w = ZeroAgent('landlord_up', 'op_agent/douzero_WP/landlord_up.ckpt')

    zero_land_a = ZeroAgent('landlord', 'op_agent/douzero_ADP/landlord.ckpt')
    zero_pea_d_a = ZeroAgent('landlord_down', 'op_agent/douzero_ADP/landlord_down.ckpt')
    zero_pea_u_a = ZeroAgent('landlord_up', 'op_agent/douzero_ADP/landlord_up.ckpt')


    rhcp_shang_agent = RhcpShangAgent(309)
    
    eval_env = rlcard.make('doudizhu')
    rhcp_shang_agent.num = 0
    reward = [0,0,0]
    eval_env.set_agents([rhcp_shang_agent,sl_pea_d,sl_pea_u])

    for i in range(1, evaluate_num + 1):
        rhcp_shang_agent.num = 0
        """
        zeta_land.update_model()
        zeta_pea_d.update_model()
        zeta_pea_u.update_model()

        zeta_land_com.update_model()
        zeta_pea_d_com.update_model()
        zeta_pea_u_com.update_model()
        """

        guiji, pay_off = eval_env.run(is_training=False)
        reward = reward + pay_off
        print("the reward is ===", i, "****===", reward)



    m_e = datetime.datetime.now()
    sec = m_e - m_s
    print("执行完成,耗时", sec.seconds)
