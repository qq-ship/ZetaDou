# coding=utf-8
import torch
import torch.nn as nn
from net_learner.zeta_model import Net,KNet
import numpy as np
class Llearner(nn.Module):
    def __init__(self, action_num):
        super(Llearner, self).__init__()

        self.lnet = Net(action_num).cuda()
        self.kicker = KNet(28).cuda()

        self.pop = torch.optim.Adam(self.lnet.parameters(),lr = 0.0001)
        self.kpop = torch.optim.Adam(self.kicker.parameters(), lr=0.0001)


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def v_wrap(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def sample_noise(self,score):
        self.lnet.sample_noise(score,0)

    def remove_noise(self):
        self.lnet.remove_noise()


    def learn_op(self,data):

        buffers_obs,buffers_obs_seq,buffers_actions,buffer_rewards,buffer_pr= [],[],[],[],[]

        for da in data:
            buffers_obs.append(da['obs'])
            buffer_rewards.append(da['reward'])
            buffers_actions.append(da['ac_ab'])
            buffers_obs_seq.append(da['action_seq'])
            buffer_pr.append(da['pr'])

        #print("====++++=======",buffers_obs)
        """
        bs = self.v_wrap(np.array(buffers_obs))
        br = self.v_wrap(np.array(buffer_rewards)).view(len(buffer_rewards))
        ba = self.v_wrap(np.array(buffers_actions)).view(len(buffers_actions),1)
        bp = self.v_wrap(np.array(buffer_pr))
        bse = buffers_obs_seq
        """
        bs = buffers_obs
        bse = buffers_obs_seq
        ba = buffers_actions
        br = buffer_rewards
        bp = buffer_pr


        actor_loss,critic_loss,loss = self.lnet.loss_func(bs,bse,ba,br,bp)

        self.pop.zero_grad()
        loss.backward(retain_graph=True)
        self.pop.step()


        #return actor_loss.mean().detach().cpu().numpy(),critic_loss.mean().detach().cpu().numpy(),loss.detach().cpu().numpy(),acc
        return actor_loss.item(),critic_loss.item(), loss.item(),0

    def learn_kic(self,kdata):
        kicker_obs,kicker_actions,kicker_rewards = [],[],[]
        k_loss = 0
        if len(kdata) > 0:
            for da in kdata:
                kicker_obs.append(da['k_state'])
                kicker_rewards.append(da['k_reward'])
                kicker_actions.append(da['k_action'])

            kbr = self.v_wrap(np.array(kicker_rewards)).view(len(kicker_rewards), 1)
            kba = self.v_wrap(np.array(kicker_actions)).view(len(kicker_actions), 1)
            kprob = self.kicker.forward(kicker_obs)

            kvalue = self.kicker_critic.forward(kicker_obs)
            kvalue = (kprob * kvalue).sum(1).view(kvalue.size(0), 1)
            kcritic_loss = 0.5 * (kbr - kvalue).pow(2)
            kadvantage = (kbr - kvalue).detach()

            klog_pob = kprob.log()
            klog_pob = klog_pob.gather(dim=1, index=kba.long().view(kba.size(0), 1))

            """
            kentroy = -(kprob * kprob.log()).sum(1)
            kentroy = kentroy.view(kentroy.size(0), -1)
            """

            k_loss_tmp = -kadvantage * klog_pob

            k_loss = (k_loss_tmp + kcritic_loss).mean()

            self.kpop.zero_grad()
            self.kcpop.zero_grad()
            k_loss.backward()
            self.kpop.step()
            self.kcpop.step()
            k_loss = k_loss.detach().cpu().numpy()
            #print("00==",k_loss)

        return k_loss


class Plearner(nn.Module):
    def __init__(self, action_num):
        super(Plearner, self).__init__()

        self.lnet_d = Net(action_num).cuda()
        self.lnet_u = Net(action_num).cuda()

        self.kicker_d = KNet(28).cuda()
        self.kicker_u = KNet(28).cuda()

        self.pop_d = torch.optim.Adam(self.lnet_d.parameters(), lr = 0.0001)
        self.pop_u = torch.optim.Adam(self.lnet_u.parameters(), lr = 0.0001)

        self.kpop_d = torch.optim.Adam(self.kicker_d.parameters(), lr=0.0001)
        self.kpop_u = torch.optim.Adam(self.kicker_u.parameters(), lr=0.0001)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def v_wrap(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def sample_noise(self,score):
        self.lnet_d.sample_noise(score,1)
        self.lnet_u.sample_noise(score,2)

    def remove_noise(self):
        self.lnet_d.remove_noise()
        self.lnet_u.remove_noise()


    def learn_op(self, data,data01):
        buffers_obs_d, buffers_actions_d, buffer_rewards_d,buffers_obs_u, buffers_actions_u, buffer_rewards_u,buffer_obs_com_d,buffer_obs_com_u = [], [], [], [], [],[],[],[]
        buffers_obs_seq_d,buffers_obs_seq_u = [],[]
        buffer_pr_d,buffer_pr_u = [],[]

        """
                    buffers_obs.append(da['obs'])
            buffer_rewards.append(da['reward'])
            buffers_actions.append(da['ac_ab'])
            buffers_obs_seq.append(da['action_seq'])
            buffer_pr.append(da['pr'])
        """

        for da in data:
            buffers_obs_d.append(da['obs'])
            buffers_actions_d.append(da['ac_ab'])
            buffer_rewards_d.append(da['reward'])
            buffers_obs_seq_d.append(da['action_seq'])
            buffer_pr_d.append(da['pr'])
            """
            buffer_obs_com_d.append(id00+da['obs_com'])
            """
        for da in data01:
            buffers_obs_u.append(da['obs'])
            buffers_actions_u.append(da['ac_ab'])
            buffer_rewards_u.append(da['reward'])
            buffers_obs_seq_u.append(da['action_seq'])
            buffer_pr_u.append(da['pr'])
            """
            buffer_obs_com_u.append(id01+da['obs_com'])
            """
        """
        ba_d = self.v_wrap(np.array(buffers_actions_d)).view(len(buffers_actions_d), 1)
        br_d = self.v_wrap(np.array(buffer_rewards_d)).view(len(buffer_rewards_d))
        bp_d = self.v_wrap(np.array(buffer_pr_d))
        bse_d =  buffers_obs_seq_d

        ba_u = self.v_wrap(np.array(buffers_actions_u)).view(len(buffers_actions_u), 1)
        br_u = self.v_wrap(np.array(buffer_rewards_u)).view(len(buffer_rewards_u))
        bp_u = self.v_wrap(np.array(buffer_pr_u))
        bse_u = buffers_obs_seq_u
        """

        bs_d = buffers_obs_d
        bse_d = buffers_obs_seq_d
        ba_d = buffers_actions_d
        br_d = buffer_rewards_d
        bp_d = buffer_pr_d

        bs_u = buffers_obs_u
        bse_u = buffers_obs_seq_u
        ba_u = buffers_actions_u
        br_u = buffer_rewards_u
        bp_u = buffer_pr_u

        actor_loss_d,critic_loss_d,loss_d = self.lnet_d.loss_func(bs_d,bse_d,ba_d, br_d, bp_d)
        actor_loss_u,critic_loss_u,loss_u = self.lnet_u.loss_func(bs_u,bse_u,ba_u, br_u, bp_u)


        self.pop_d.zero_grad()
        loss_d.backward(retain_graph=True)
        self.pop_d.step()

        self.pop_u.zero_grad()
        loss_u.backward(retain_graph=True)
        self.pop_u.step()

        return actor_loss_d.item(),critic_loss_d.item(),loss_d.item(),0, actor_loss_u.item(),critic_loss_u.item(),loss_u.item(), 0


    def learn_kic(self,ddata,udata):

        d_loss = 0
        u_loss = 0

        kicker_d_obs, kicker_d_actions, kicker_d_rewards = [], [], []
        kicker_u_obs, kicker_u_actions, kicker_u_rewards = [], [], []

        if len(ddata) > 0:
            for da in ddata:
                kicker_d_obs.append(da['k_state'])
                kicker_d_rewards.append(da['k_reward'])
                kicker_d_actions.append(da['k_action'])

            dbr = self.v_wrap(np.array(kicker_d_rewards)).view(len(kicker_d_rewards), 1)
            dba = self.v_wrap(np.array(kicker_d_actions)).view(len(kicker_d_actions), 1)
            dprob = self.kicker_d.forward(kicker_d_obs)

            dvalue = self.kicker_critic_d.forward(kicker_d_obs)
            dvalue = (dprob * dvalue).sum(1).view(dvalue.size(0), 1)
            dcritic_loss = 0.5 * (dbr - dvalue).pow(2)
            dadvantage = (dbr - dvalue).detach()

            dlog_pob = dprob.log()
            dlog_pob = dlog_pob.gather(dim=1, index=dba.long().view(dba.size(0), -1))

            """
            dentroy = -(dprob * dprob.log()).sum(1)
            dentroy = dentroy.view(dentroy.size(0), -1)
            """

            d_loss_tmp = -dadvantage * dlog_pob
            d_loss = (d_loss_tmp + dcritic_loss).mean()
            # d_loss = d_loss_tmp.mean()
            #print("01==", d_loss.detach().cpu().numpy())

            self.kpop_d.zero_grad()
            self.k_cpop_d.zero_grad()
            d_loss.backward(retain_graph=True)
            self.kpop_d.step()
            self.k_cpop_d.step()
            d_loss = d_loss.detach().cpu().numpy()

        if len(udata) > 0:

            for da in udata:
                kicker_u_obs.append(da['k_state'])
                kicker_u_rewards.append(da['k_reward'])
                kicker_u_actions.append(da['k_action'])

            ubr = self.v_wrap(np.array(kicker_u_rewards)).view(len(kicker_u_rewards), 1)
            uba = self.v_wrap(np.array(kicker_u_actions)).view(len(kicker_u_actions), 1)
            uprob = self.kicker_u.forward(kicker_u_obs)

            uvalue = self.kicker_critic_u.forward(kicker_u_obs)
            uvalue = (uprob * uvalue).sum(1).view(uvalue.size(0), 1)
            ucritic_loss = 0.5 * (ubr - uvalue).pow(2)
            uadvantage = (ubr - uvalue).detach()

            ulog_pob = uprob.log()
            ulog_pob = ulog_pob.gather(dim=1, index=uba.long().view(uba.size(0), -1))


            u_loss_tmp = -uadvantage*ulog_pob
            u_loss = (u_loss_tmp + ucritic_loss).mean()
            # u_loss = u_loss_tmp.mean()
            #print("02==", u_loss.detach().cpu().numpy())

            self.kpop_u.zero_grad()
            self.k_cpop_u.zero_grad()
            u_loss.backward(retain_graph=True)
            self.kpop_u.step()
            self.k_cpop_u.step()
            u_loss = u_loss.detach().cpu().numpy()
        return d_loss,u_loss
