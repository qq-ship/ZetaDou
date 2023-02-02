# coding=utf-8
import torch
from torch.nn import init, Parameter
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from game_model.action_map.action_map import action_to_ab_id,r_decode_action,ID_SPACE,n_decode_action
from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION
from rlcard.utils.utils import remove_illegal
import math
from torch.autograd import Variable

class RhcpNet(nn.Module):
    def __init__(self,action_num=100):
        super(RhcpNet, self).__init__()

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

        con07 = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(con07.weight)
        torch.nn.init.constant(con07.bias, 0.1)

        con08 = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        torch.nn.init.xavier_uniform(con08.weight)
        torch.nn.init.constant(con08.bias, 0.1)


        self.conv = nn.Sequential(
            con01,
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            con02,
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            con03,
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            con04,
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            con05,
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            con06,
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            con07,
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            con08,
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.action_num = action_num
        self.distribution = torch.distributions.Categorical
        self.fc  = nn.Linear(1792,309)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_func_method = nn.CrossEntropyLoss()

    def forward(self,s):
        s = s.view(s.size(0), 1,52,15)
        s = self.conv(s)
        s = s.view(s.size(0), -1)
        action_logits = F.softmax(self.fc(s))

        action_logits = action_logits + 1e-10

        return action_logits

    def v_wrap(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def choose_action(self,s,legal_actions_id,rule_actions):
        kicker_data = 0
        s = self.v_wrap(np.array(s))

        probs= self.forward(s)
        probs = probs.detach().cpu().numpy()[0]
        tmp_probs = probs
        legal_actions = [ID_2_ACTION[item] for item in legal_actions_id]
        legal_actions_ab = [action_to_ab_id(item) for item in legal_actions]
        legal_actions_ab = list(set(legal_actions_ab))
        probs = remove_illegal(probs,legal_actions_ab)

        action = np.argmax(probs)

        action_str = ID_SPACE[action]

        if '*' in action_str:
            action_str = r_decode_action(action, legal_actions, rule_actions)

        action = ACTION_2_ID[action_str]

        return action,tmp_probs,kicker_data

    def cc_action(self, s):
        s = self.v_wrap(np.array(s))
        prob = self.forward(s).detach().cpu().numpy()
        aa = np.argmax(prob,axis=1)
        return aa

    def loss_func(self, s,act):
        self.train()

        s = self.v_wrap(np.array(s))
        act = self.v_wrap(np.array(act)).long ()

        prob = self.forward(s)
        loss = self.loss_func_method(prob, act)

        return loss