#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init, Parameter
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable
from game_model.action_map.action_map import action_to_ab_id, r_decode_action, ID_SPACE, n_decode_action
from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION
from rlcard.utils.utils import remove_illegal

import math

class noisypool(object):
    def __init__(self,size):
        self.noisy_weigth = []
        self.noisy_bias = []
        self.score = []
        self.size = size
    def addmemory(self,mem_weight,mem_bias,score):
        if len(self.noisy_weigth)>self.size:
            index = np.argmin(np.array(self.score))
            self.noisy_weigth.pop(index)
            self.noisy_bias.pop(index)
            self.score.pop(index)

        self.noisy_weigth.append(mem_weight)
        self.noisy_bias.append(mem_bias)
        self.score.append(score)

    def sample(self,num):
        score_array = np.array(self.score)
        score_array = score_array/score_array.sum()
        index = np.random.choice(len(self.score),p = score_array)
        return self.noisy_weigth[index], self.noisy_bias[index]

class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.051, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.sigma_init = sigma_init
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features).to(self.device))  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features).to(self.device))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features).to(self.device))
    self.register_buffer('epsilon_bias', torch.zeros(out_features).to(self.device))
    self.reset_parameters()

    self.npool = noisypool(20)
    self.tmp_epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.tmp_epsilon_bias = torch.randn(self.out_features)


  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.constant(self.sigma_weight, self.sigma_init)
      init.constant(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight).cuda(), self.bias + self.sigma_bias * Variable(self.epsilon_bias).cuda())

  def sample_noise(self,score,player_id):
    self.ran = random.random()
    #print("====",self.ran,"====:","====",len(self.npool.score),"====:")
    self.npool.addmemory(self.tmp_epsilon_weight,self.tmp_epsilon_bias,score)
    if self.ran < 0.4 or player_id == 0:
        #print("player_id:",player_id,"zhi xing random")
        self.epsilon_weight = torch.randn(self.out_features,self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)
    else:
        #print("player_id:", player_id, "zhi xing pool")
        if len(self.npool.score) > 0:
            self.epsilon_weight, self.epsilon_bias = self.npool.sample(1)
        else:
            self.epsilon_weight = torch.randn(self.out_features, self.in_features)
            self.epsilon_bias = torch.randn(self.out_features)

    self.tmp_epsilon_weight = self.epsilon_weight
    self.tmp_epsilon_bias = self.epsilon_bias

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)



class Mish(torch.nn.Module):
    __name__ = 'Mish'

    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))

class Net(nn.Module):
    def __init__(self, action_num=2, state_shape=None, mlp_layers=None):
        super(Net, self).__init__()

        self.action_num = action_num
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.mish = Mish()

        con01 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con01.weight)
        torch.nn.init.constant(con01.bias, 0.1)

        con02 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con02.weight)
        torch.nn.init.constant(con02.bias, 0.1)

        con03 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con03.weight)
        torch.nn.init.constant(con03.bias, 0.1)

        con04 = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(con04.weight)
        torch.nn.init.constant(con04.bias, 0.1)

        con05 = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(con05.weight)
        torch.nn.init.constant(con05.bias, 0.1)

        con06 = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(con06.weight)
        torch.nn.init.constant(con06.bias, 0.1)

        self.conv = nn.Sequential(
            con01,
            nn.BatchNorm2d(64),
            nn.Tanh(),
            con02,
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            con03,
            nn.BatchNorm2d(64),
            nn.Tanh(),
            con04,
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            con05,
            nn.BatchNorm2d(64),
            nn.Tanh(),
            con06,
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.fc = NoisyLinear(2240,self.action_num, bias=True)
        self.fc1 = NoisyLinear(2240,309, bias=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss = nn.CrossEntropyLoss()
        self.distribution = torch.distributions.Categorical

    def forward(self, s):
        s = s.view(s.size(0),1,30,15)

        s = self.conv(s)
        s = s.view(s.size(0),-1)
        tmp = self.fc(s)

        action_logits = F.softmax(tmp)
        state_values = self.fc1(s)

        return action_logits,state_values

    def v_wrap(self,np_array, dtype=np.float32):

        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def sample_noise(self,score,player_id):
        self.fc.sample_noise(score,player_id)
        self.fc1.sample_noise(score,player_id)

    def remove_noise(self):
        self.fc.remove_noise()
        self.fc1.remove_noise()

    def t_choose_action(self, state, legal_action):
        self.eval()
        legal_action = sorted(legal_action)
        prob,value = self.forward(self.v_wrap(state))
        prob = prob + 1e-10
        return prob

    def choose_action(self, s, legal_actions_id, rule_actions,model='test'):
        kicker_data = 0
        tmp_s = s
        s = self.v_wrap(np.array(s))

        probs, _ = self.forward(s)
        probs = probs.detach().cpu().numpy()[0]
        tmp_probs = probs
        legal_actions = [ID_2_ACTION[item] for item in legal_actions_id]
        legal_actions_ab = [action_to_ab_id(item) for item in legal_actions]
        legal_actions_ab = list(set(legal_actions_ab))
        probs = remove_illegal(probs, legal_actions_ab)

        if model is 'train':

            action = np.random.choice(len(probs), p=probs)
        else:
            action = np.argmax(probs)

        action_str = ID_SPACE[action]

        if '*' in action_str:
            action_str = r_decode_action(action, legal_actions, rule_actions)

        action = ACTION_2_ID[action_str]

        return action, tmp_probs, kicker_data

    def loss_func(self, s, a,value_target,bp):
        self.train()
        bp = bp.squeeze()
        prob, value = self.forward(s)

        value = prob * value
        value = value.sum(1)
        value = value.squeeze()
        td_error = value_target - value
        critic_loss = 0.5 * td_error.pow(2)

        # td_error = (td_error - td_error.mean())/( td_error.std() + 1e-10)

        rho = (prob / bp).detach()
        rho_a = a.view(a.size(0), 1)
        rho_action = torch.gather(rho, 1, rho_a.long())

        prob = prob + 1e-10
        m = self.distribution(prob)
        log_pob = m.log_prob(a)

        # td_error = torch.clamp(td_error, min=0.0)
        # td_error = td_error + 1e-10

        actor_loss_tmp = -torch.clamp(rho_action, max=2.0).detach() * log_pob * td_error.detach()
        # actor_loss_tmp = -torch.clamp(rho_action, max=1.5).detach()  * td_error.detach()
        rho_correction = torch.clamp(1 - 2.0 / rho_action, min=0.).detach()
        tmp_td_error = td_error.view(td_error.size(0), 1)
        actor_loss_tmp -= (rho_correction * log_pob * tmp_td_error.detach())
        # actor_loss_tmp -= (rho_correction  * tmp_td_error.detach())

        entroy = -(prob.log() * prob).sum(1)
        exp_v = actor_loss_tmp

        actor_loss = exp_v - 0.01 * entroy

        loss = (critic_loss + actor_loss).mean()

        return loss