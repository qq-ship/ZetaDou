# coding=utf-8
import torch
from torch.nn import init, Parameter
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from game_model.action_map.action_map import action_to_ab_id, r_decode_action, ID_SPACE, n_decode_action
from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION
from rlcard.utils.utils import remove_illegal
import math
from torch.autograd import Variable


class noisypool(object):
    def __init__(self, size):
        self.noisy_weigth = []
        self.noisy_bias = []
        self.score = []
        self.size = size

    def addmemory(self, mem_weight, mem_bias, score):
        if len(self.noisy_weigth) > self.size:
            index = np.argmin(np.array(self.score))
            self.noisy_weigth.pop(index)
            self.noisy_bias.pop(index)
            self.score.pop(index)

        self.noisy_weigth.append(mem_weight)
        self.noisy_bias.append(mem_bias)
        self.score.append(score)

    def sample(self, num):
        score_array = np.array(self.score)
        score_array = score_array / score_array.sum()
        index = np.random.choice(len(self.score), p=score_array)
        return self.noisy_weigth[index], self.noisy_bias[index]
    def clear(self):
        self.noisy_weigth = []
        self.noisy_bias = []
        self.score = []


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, device_id = 0, sigma_init=0.095, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        print("*****************::::::::::",device_id)
        self.device_id = int(device_id)
        self.device = 'cuda:'+str(int(device_id)) if torch.cuda.is_available() else 'cpu'
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
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight).cuda(self.device_id),
                        self.bias + self.sigma_bias * Variable(self.epsilon_bias).cuda(self.device_id))

    def sample_noise(self, score, player_id):

        self.ran = random.random()
        # print("====",self.ran,"====:","====",len(self.npool.score),"====:")
        self.npool.addmemory(self.tmp_epsilon_weight, self.tmp_epsilon_bias, score)
        if self.ran < 0.4:
            # print("player_id:",player_id,"zhi xing random")
            self.epsilon_weight = torch.randn(self.out_features, self.in_features)
            self.epsilon_bias = torch.randn(self.out_features)
        else:
            # print("player_id:", player_id, "zhi xing pool")
            if len(self.npool.score) > 0:
                self.epsilon_weight, self.epsilon_bias = self.npool.sample(1)
            else:
                self.epsilon_weight = torch.randn(self.out_features, self.in_features)
                self.epsilon_bias = torch.randn(self.out_features)

        self.tmp_epsilon_weight = self.epsilon_weight
        self.tmp_epsilon_bias = self.epsilon_bias
        # print("-=-=-=-::", self.epsilon_weight[0],"=======",self.sigma_init)

    def sample_noise_n(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class Net(nn.Module):
    def __init__(self, action_num=100, noise=0.095):
        super(Net, self).__init__()

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
            # nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            con02,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            con03,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            con04,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            con05,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            con06,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.action_num = action_num
        self.distribution = torch.distributions.Categorical
        self.lstm = nn.LSTM(180, 256, batch_first=True)
        self.fc = NoisyLinear(2176, 309, sigma_init=noise, bias=True)
        self.fc1 = NoisyLinear(2176, 309, sigma_init=0.0, bias=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, s, seq_input):
        action_values = 0
        s = s.view(s.size(0), 1, 24, 15)
        seq, (h_n, _) = self.lstm(seq_input)
        seq = seq[:, -1, :]
        s = self.conv(s)
        s = s.view(s.size(0), -1)
        s_com = torch.cat([s, seq], dim=-1)
        action_values = self.fc1(s_com)
        #print("s_com",s_com.size())
        action_logits = F.softmax(self.fc(s_com))
        action_logits = action_logits + 1e-10

        return action_logits, action_values

    def v_wrap(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def sample_noise(self, score, player_id):
        self.fc.sample_noise(score, player_id)

    def sample_noise_n(self):

        self.fc.sample_noise_n()

    def remove_noise(self):
        self.fc.remove_noise()

    def choose_action(self, s, seq, legal_actions_id, rule_actions, kicker='', model='train', k_model='rule'):
        kicker_data = 0
        tmp_s = s
        tmp_seq = seq
        s = self.v_wrap(np.array(s))
        seq = self.v_wrap(np.array(seq))

        probs, _ = self.forward(s, seq)
        probs = probs.detach().cpu().numpy()[0]
        tmp_probs = probs
        legal_actions = [ID_2_ACTION[item] for item in legal_actions_id]
        legal_actions_ab = [action_to_ab_id(item) for item in legal_actions]
        legal_actions_ab = list(set(legal_actions_ab))
        probs = remove_illegal(probs, legal_actions_ab)

        if model is 'train':
            # action = np.argmax(probs)
            action = np.random.choice(len(probs), p=probs)
        else:
            action = np.argmax(probs)

        action_str = ID_SPACE[action]

        if '*' in action_str:
            if k_model is 'rule':
                action_str = r_decode_action(action, legal_actions, rule_actions)
            elif k_model is 'neural':
                action_str, kicker_data = n_decode_action(tmp_s, tmp_seq, action, legal_actions, rule_actions, kicker,
                                                          model)
            else:
                action_str = r_decode_action(action, legal_actions, rule_actions)

        action = ACTION_2_ID[action_str]

        return action, tmp_probs, kicker_data

    def loss_func(self, s, seq, a, value_target, bp):
        self.train()

        s = self.v_wrap(np.array(s))
        seq = self.v_wrap(np.array(seq))
        a = self.v_wrap(np.array(a))
        value_target = self.v_wrap(np.array(value_target))
        bp = self.v_wrap(np.array(bp))

        bp = bp.squeeze()
        prob, value = self.forward(s, seq)

        value = prob * value
        value = value.sum(1)
        value = value.squeeze()
        td_error = value_target - value
        critic_loss = 0.5 * td_error.pow(2)

        critic_loss = critic_loss * 5

        rho = (prob / bp).detach()
        rho_a = a.view(a.size(0), 1)
        # m = self.distribution(rho)
        # ratio = m.log_prob(rho_a)

        rho_action = torch.gather(rho, 1, rho_a.long())
        m = self.distribution(prob)
        log_pob = m.log_prob(a)
        #************************************************************************************
        actor_loss_tmp = -torch.clamp(rho_action, max=1.5).detach() * log_pob * td_error.detach()
        rho_correction = torch.clamp(1 - 1.5 / rho_action, min=0.).detach()
        # ************************************************************************************
        tmp_td_error = td_error.view(td_error.size(0), 1)
        actor_loss_tmp -= (rho_correction * log_pob * tmp_td_error.detach())

        # actor_loss_tmp = -torch.min(ratio *  td_error.detach(),torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * td_error.detach())

        entroy = -(prob.log() * prob).sum(1)
        exp_v = actor_loss_tmp

        actor_loss = exp_v - 0.01 * entroy

        loss = (actor_loss + critic_loss).mean()

        return actor_loss.mean(), critic_loss.mean(), loss

class PNet(nn.Module):
    def __init__(self,action_num=100,device_id =0,noise=0.119):
        super(PNet, self).__init__()


        con01 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con01.weight)
        torch.nn.init.constant(con01.bias, 0.1)

        con02 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con02.weight)
        torch.nn.init.constant(con02.bias, 0.1)

        con03 = nn.Conv2d(32, 32,kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con03.weight)
        torch.nn.init.constant(con03.bias, 0.1)

        con04 = nn.Conv2d(32, 32,kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(con04.weight)
        torch.nn.init.constant(con04.bias, 0.1)

        con05 = nn.Conv2d(32, 32,kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con05.weight)
        torch.nn.init.constant(con05.bias, 0.1)

        con06 = nn.Conv2d(32, 32,kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(con06.weight)
        torch.nn.init.constant(con06.bias, 0.1)

        Acon01 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(Acon01.weight)
        torch.nn.init.constant(Acon01.bias, 0.1)

        Acon02 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(Acon02.weight)
        torch.nn.init.constant(Acon02.bias, 0.1)

        Acon03 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(Acon03.weight)
        torch.nn.init.constant(Acon03.bias, 0.1)

        Acon04 = nn.Conv2d(32, 32, kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(Acon04.weight)
        torch.nn.init.constant(Acon04.bias, 0.1)

        Acon05 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(Acon05.weight)
        torch.nn.init.constant(Acon05.bias, 0.1)

        Acon06 = nn.Conv2d(32, 32, kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(Acon06.weight)
        torch.nn.init.constant(Acon06.bias, 0.1)


        self.conv = nn.Sequential(
            con01,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            con02,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            con03,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            con04,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            con05,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            con06,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.conv01 = nn.Sequential(
            Acon01,
            nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Tanh(),
            Acon02,
            nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Tanh(),
            Acon03,
            nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Acon04,
            nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Tanh(),
            Acon05,
            nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Tanh(),
            Acon06,
            nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.action_num = action_num
        self.noise = noise
        self.distribution = torch.distributions.Categorical
        self.lstm = nn.LSTM(180, 256, batch_first=True)

        self.device_id = device_id
        self.device = 'cuda:' + str(device_id) if torch.cuda.is_available() else 'cpu'

        self.fc = NoisyLinear(2816, 309, self.device_id, sigma_init=self.noise, bias=True)
        self.fc1 = NoisyLinear(3328,309, self.device_id, sigma_init=0.0, bias=True)

        self.gpu_id = torch.cuda.current_device()


    def forward(self, s, seq_input, s01=None):

        action_values = 0
        if s01 is None:
            s = s.view(s.size(0),1,24,15)
            seq, (h_n, _) = self.lstm(seq_input)
            seq = seq[:, -1, :]
            s = self.conv(s)
            s = s.view(s.size(0), -1)
            s_com = torch.cat([s, seq], dim=-1)
            #print("s_com",s_com.size())
            action_logits = F.softmax(self.fc(s_com))
            action_logits = action_logits + 1e-10
        else:
            s = s.view(s.size(0),1,24,15)
            #print("====****=======",s01.size())
            s01 = s01.view(s01.size(0),1,32,15)

            seq, (h_n, _) = self.lstm(seq_input)
            seq = seq[:, -1, :]
            s = self.conv(s)
            s = s.view(s.size(0), -1)
            s_com = torch.cat([s, seq], dim=-1)
            action_logits = F.softmax(self.fc(s_com))
            action_logits = action_logits + 1e-10

            s01 = self.conv01(s01)
            s01 = s01.view(s01.size(0), -1)
            s_com01 = torch.cat([s01, seq], dim=-1)
            #print("s_com01===========",s_com01.size())
            action_values = self.fc1(s_com01)

        return action_logits, action_values

    def v_wrap(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def sample_noise(self, score, player_id):
        print("========= Net sample noise")
        self.fc.sample_noise(score, player_id)
        self.fc1.sample_noise(score, player_id)

    def sample_noise_n(self):
        self.fc.sample_noise_n()
        self.fc1.sample_noise_n()

    def remove_noise(self):
        self.fc.remove_noise()
        self.fc1.remove_noise()

    def choose_action(self, s, seq, legal_actions_id, rule_actions, kicker='', model='train', k_model='rule'):
        kicker_data = 0
        tmp_s = s
        tmp_seq = seq
        s = self.v_wrap(np.array(s))
        seq = self.v_wrap(np.array(seq))

        probs, _ = self.forward(s, seq)
        probs = probs.detach().cpu().numpy()[0]
        tmp_probs = probs
        legal_actions = [ID_2_ACTION[item] for item in legal_actions_id]
        legal_actions_ab = [action_to_ab_id(item) for item in legal_actions]
        legal_actions_ab = list(set(legal_actions_ab))
        probs = remove_illegal(probs, legal_actions_ab)

        if model is 'train':
            # action = np.argmax(probs)
            action = np.random.choice(len(probs), p=probs)
        else:
            action = np.argmax(probs)

        action_str = ID_SPACE[action]

        if '*' in action_str:
            if k_model is 'rule':
                action_str = r_decode_action(action, legal_actions, rule_actions)
            elif k_model is 'neural':
                action_str, kicker_data = n_decode_action(tmp_s, tmp_seq, action, legal_actions, rule_actions, kicker,
                                                          model)
            else:
                action_str = r_decode_action(action, legal_actions, rule_actions)

        action = ACTION_2_ID[action_str]

        return action, tmp_probs, kicker_data

    def loss_func(self, s,seq,a,value_target,bp,com=None):
        self.train()

        s = self.v_wrap(np.array(s))
        seq = self.v_wrap(np.array(seq))
        a = self.v_wrap(np.array(a))
        value_target = self.v_wrap(np.array(value_target))
        bp = self.v_wrap(np.array(bp))
        com = self.v_wrap(np.array(com))

        bp = bp.squeeze()
        prob, value = self.forward(s,seq,com)

        value = prob * value
        value = value.sum(1)
        value = value.squeeze()
        td_error = value_target - value
        critic_loss = 0.5 * td_error.pow(2)

        m = self.distribution(prob)
        log_pob = m.log_prob(a)


        rho = (prob / bp).detach()
        rho_a = a.view(a.size(0), 1)

        rho_action = torch.gather(rho, 1, rho_a.long())
        # ************************************************************************************
        actor_loss_tmp = -torch.clamp(rho_action, max=1.5).detach() * log_pob * td_error.detach()
        rho_correction = torch.clamp(1 - 1.5 / rho_action, min=0.).detach()
        # ************************************************************************************
        tmp_td_error = td_error.view(td_error.size(0), 1)
        actor_loss_tmp -= (rho_correction * log_pob * tmp_td_error.detach())

        #actor_loss_tmp = -torch.min(ratio *  td_error.detach(),torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * td_error.detach())
        exp_v = actor_loss_tmp
        entroy = -(prob.log() * prob).sum(1)

        actor_loss = exp_v - 0.01 * entroy

        loss = (actor_loss + critic_loss).mean()

        return actor_loss.mean(), critic_loss.mean(), loss

class PKNet(nn.Module):
    def __init__(self, action_num=100, device_id =0, noise=0.119):
        super(PKNet, self).__init__()

        con01 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con01.weight)
        torch.nn.init.constant(con01.bias, 0.1)

        con02 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con02.weight)
        torch.nn.init.constant(con02.bias, 0.1)

        con03 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con03.weight)
        torch.nn.init.constant(con03.bias, 0.1)

        con04 = nn.Conv2d(32, 32, kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(con04.weight)
        torch.nn.init.constant(con04.bias, 0.1)

        con05 = nn.Conv2d(32, 32,kernel_size=3, stride=1, padding=2)
        torch.nn.init.xavier_uniform(con05.weight)
        torch.nn.init.constant(con05.bias, 0.1)

        con06 = nn.Conv2d(32, 32,kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(con06.weight)
        torch.nn.init.constant(con06.bias, 0.1)

        self.conv = nn.Sequential(
            con01,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            con02,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            con03,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            con04,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            con05,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            con06,
            # nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.action_num = action_num
        self.distribution = torch.distributions.Categorical
        self.lstm = nn.LSTM(180, 256, batch_first=True)
        self.device = 'cuda:'+str(device_id) if torch.cuda.is_available() else 'cpu'
        self.device_id = device_id
        self.fc = NoisyLinear(3328, 28, device_id = self.device_id, sigma_init=0.085, bias=True)
        self.fc1 = NoisyLinear(3328,28, device_id = self.device_id, sigma_init=0.0, bias=True)

    def forward(self, s, seq):
        s = s.view(s.size(0), 1,32,15)
        seq, (h_n, _) = self.lstm(seq)
        seq = seq[:, -1, :]
        #print("s is ==",s.size())
        s = self.conv(s)
        s = s.view(s.size(0), -1)
        s_com = torch.cat([s, seq], dim=-1)
        #print("K==============",s_com.size())
        action_logits = F.softmax(self.fc(s_com))
        action_values = self.fc1(s_com)
        action_logits = action_logits + 1e-10

        return action_logits, action_values

    def v_wrap(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def sample_noise(self, score, player_id):
        print("========= KNet sample noise")
        self.fc.sample_noise(score, player_id)
        self.fc1.sample_noise(score, player_id)

    def sample_noise_n(self):
        self.fc.sample_noise_n()
        self.fc1.sample_noise_n()

    def remove_noise(self):
        self.fc.remove_noise()
        self.fc1.remove_noise()

    def choose_action(self, s, seq, legal_actions_id, model):
        s = self.v_wrap(np.array(s))
        seq = self.v_wrap(np.array(seq))

        probs, _ = self.forward(s, seq)
        probs = probs.detach().cpu().numpy()[0]
        tmp_prob = probs

        probs = remove_illegal(probs, legal_actions_id)
        if model is 'train':
            action = np.random.choice(len(probs), p=probs)
            # action = np.argmax(probs)
        else:
            action = np.argmax(probs)
        return action, tmp_prob

    def loss_func(self, s, seq, a, value_target, bp, alpha):
        self.train()

        s = self.v_wrap(np.array(s))
        seq = self.v_wrap(np.array(seq))
        a = self.v_wrap(np.array(a))
        value_target = self.v_wrap(np.array(value_target))
        bp = self.v_wrap(np.array(bp))

        prob, value = self.forward(s, seq)

        value = prob * value
        value = value.sum(1)
        value = value.squeeze()
        td_error = value_target - value
        critic_loss = 0.5 * td_error.pow(2)

        m = self.distribution(prob)
        log_pob = m.log_prob(a)

        rho = (prob / bp).detach()
        rho_a = a.view(a.size(0), 1)

        rho_action = torch.gather(rho, 1, rho_a.long())
        # ************************************************************************************
        rho_correction = torch.clamp(rho_action, max=1.5).detach() + torch.clamp(1 - 1.5/ rho_action, min=0.).detach()
        # ************************************************************************************
        actor_loss_tmp = -(rho_correction * log_pob * td_error.detach())

        entroy = -(prob.log() * prob).sum(1)
        exp_v = actor_loss_tmp

        actor_loss = exp_v - 0.001 * entroy

        loss = (actor_loss + critic_loss).mean()

        return loss