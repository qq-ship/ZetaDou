import  datetime
import os
from logger.logger import Logger
def create_catalogue():
    nowTime = str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    path = "out_put/logs/zeta/" + nowTime
    os.makedirs(path)

    path01 = "out_put/nets/zeta/" + nowTime
    os.makedirs(path01)
    return path,path01

def create_csv(path):
    log_path = path + '/' + 'log.txt'
    csv_path = path + '/' + 'performance.csv'
    logger = Logger(xlabel='epsoide', ylabel='landlord', zlabel='farmernext', mlabel='farmerup', lslabel='ls', ps1label='ps1', ps2label='ps2', l_a_loss='allo', l_c_loss='cllo', l_s_loss='sllo',p1_a_loss='ap1lo', p1_c_loss='cp1lo', p1_s_loss='sp1lo', p2_a_loss='ap2lo', p2_c_loss='cp2lo', p2_s_loss='sp2lo',k_l_loss='kicker_lan_loss', k_d_loss='kicker_down_loss', k_u_loss='kicker_up_loss',acc_0='acc0', acc_1='acc1', acc_2='acc2', acc_t='acct', legend='a3c on Dou Dizhu', log_path=log_path, csv_path=csv_path)
    return logger

def add_data_csv(logger,episode,eval_every_eposide,reward,reward01,reward02,evaluate_num,landlord_scores,persant_d_scores,persant_u_scores,nallo,ncllo,nsllo,nap1lo,ncp1lo,nsp1lo,nap2lo,ncp2lo,nsp2lo,k_l_loss,k_d_loss,k_u_loss,acc,acc01,acc02,acct):
    logger.add_point(x=episode / eval_every_eposide, y=float(reward) / evaluate_num,z=float(reward01) / evaluate_num, m=float(reward02) / evaluate_num,ls=landlord_scores, ps1=persant_d_scores, ps2=persant_u_scores, allo=nallo,cllo=ncllo, sllo=nsllo, ap1lo=nap1lo, cp1lo=ncp1lo, sp1lo=nsp1lo, ap2lo=nap2lo, cp2lo=ncp2lo, sp2lo=nsp2lo,k_l_loss=k_l_loss,k_d_loss=k_d_loss,k_u_loss=k_u_loss,acc0=acc, acc1=acc01, acc2=acc02, acct=acct)