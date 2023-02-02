import numpy as np
import random

class Memory(object):
    def __init__(self,size):
        self.memory = []
        self.size = size
    def addmemory(self,mem):
        if len(self.memory)>self.size:
            self.memory.pop(0)
        self.memory.append(mem)

    def sample(self,num):
        obs = []
        act = []
        if len(self.memory) <= num:
            rs = self.memory
            for ts in rs:
                obs.append(ts['obs'])
                act.append(ts['action'])

            return obs,act
        else:
            rs = random.sample(self.memory, num)
            for ts in rs:
                obs.append(ts['obs'])
                act.append(ts['action'])
            return obs,act

class Memseq(object):
    def __init__(self,size):
        self.memory = []
        self.size = size
        tmp = np.zeros([5,15])
        tmp[0][:] = 1
        round = np.array([tmp.reshape(75),tmp.reshape(75),tmp.reshape(75)]).reshape(225)

        for i in range(self.size):
            self.memory .append(round)

    def addmemory(self,mem):
        self.memory.pop()
        self.memory.insert(0,mem)

    def getmemory(self):
        return self.memory

    def initmem(self):
        self.memory = []
        tmp = np.zeros([5, 15])
        tmp[0][:] = 1
        round = np.array([tmp.reshape(75), tmp.reshape(75), tmp.reshape(75)]).reshape(225)

        for i in range(self.size):
            self.memory.append(round)
"""
class RMemory(object):
    def __init__(self, size):
        self.memory = []
        self.size = size

    def addmemory(self, mem):
        if len(self.memory) > self.size:
            self.memory.pop(0)
        self.memory.append(mem)

    def sample(self, num):

        obs = []
        ac_seq = []
        ab_legal = []
        st_action = []
        action = []
        prob = []
        st_batch = []
        ac_seq_batch = []
        val = []
        com = []

        if len(self.memory) <= num:
            rs = self.memory
            random.shuffle(rs)
            for ts in rs:
                obs.append(ts['obs'])
                ac_seq.append(ts['ac_seq'])
                ab_legal.append('ab_legal')
                st_action.append(ts['st_action'])
                action.append(ts['action'])
                prob.append(ts['prob'])
                st_batch.append(ts['st_batch'])
                ac_seq_batch.append(ts['ac_seq_batch'])
                val.append(ts['val'])
                com.append(ts['com'])

            return obs,ac_seq,ab_legal,st_action,action,prob,st_batch,ac_seq_batch,val,com
        else:

            rs = random.sample(self.memory, num)

            for ts in rs:
                obs.append(ts['obs'])
                ac_seq.append(ts['ac_seq'])
                ab_legal.append('ab_legal')
                st_action.append(ts['st_action'])
                action.append(ts['action'])
                prob.append(ts['prob'])
                st_batch.append(ts['st_batch'])
                ac_seq_batch.append(ts['ac_seq_batch'])
                val.append(ts['val'])
                com.append(ts['com'])

            return obs,ac_seq,ab_legal,st_action,action,prob,st_batch,ac_seq_batch,val,com

    def ksample(self, num):
        obs = []
        seq = []
        act = []
        val = []
        prob = []

        if len(self.memory) <= num:
            rs = self.memory
            for ts in rs:
                obs.append(ts['k_state'])
                act.append(ts['k_action'])
                seq.append(ts['k_seq'])
                val.append(ts['k_reward'])
                prob.append(ts['k_prob'])
            return obs, seq, act, val, prob
        else:
            rs = random.sample(self.memory, num)
            for ts in rs:
                obs.append(ts['k_state'])
                act.append(ts['k_action'])
                seq.append(ts['k_seq'])
                val.append(ts['k_reward'])
                prob.append(ts['k_prob'])

            return obs, seq, act, val, prob

    def get_size(self):
        return len(self.memory)

    def clear(self):
        self.memory = []

    def pea_sample(self, num):
        obs = []
        ac_seq = []
        ab_legal = []
        st_action = []
        action = []
        prob = []
        st_batch = []
        ac_seq_batch = []
        val = []
        com01 = []

        obs01 = []
        ac_seq01 = []
        ab_legal01 = []
        st_action01 = []
        action01 = []
        prob01 = []
        st_batch01 = []
        ac_seq_batch01 = []
        val01 = []
        com02 = []

        if len(self.memory) <= num:
            rs = self.memory
            random.shuffle(rs)
            for ts in rs:
                obs.append(ts[0]['obs'])
                ac_seq.append(ts[0]['ac_seq'])
                ab_legal.append(ts[0]['ab_legal'])
                st_action.append(ts[0]['st_action'])
                action.append(ts[0]['action'])
                prob.append(ts[0]['prob'])
                st_batch.append(ts[0]['st_batch'])
                ac_seq_batch.append(ts[0]['ac_seq_batch'])
                val.append(ts[0]['val'])
                com01.append(ts[0]['com'])

                obs01.append(ts[1]['obs'])
                ac_seq01.append(ts[1]['ac_seq'])
                ab_legal01.append(ts[1]['ab_legal'])
                st_action01.append(ts[1]['st_action'])
                action01.append(ts[1]['action'])
                prob01.append(ts[1]['prob'])
                st_batch01.append(ts[1]['st_batch'])
                ac_seq_batch01.append(ts[1]['ac_seq_batch'])
                val01.append(ts[1]['val'])
                com02.append(ts[1]['com'])

            return obs, ac_seq, ab_legal, st_action, action, prob, st_batch, ac_seq_batch, val,obs01, ac_seq01, ab_legal01, st_action01, action01, prob01, st_batch01, ac_seq_batch01, val01,com01,com02
        else:

            rs = random.sample(self.memory, num)

            for ts in rs:
                obs.append(ts[0]['obs'])
                ac_seq.append(ts[0]['ac_seq'])
                ab_legal.append(ts[0]['ab_legal'])
                st_action.append(ts[0]['st_action'])
                action.append(ts[0]['action'])
                prob.append(ts[0]['prob'])
                st_batch.append(ts[0]['st_batch'])
                ac_seq_batch.append(ts[0]['ac_seq_batch'])
                val.append(ts[0]['val'])
                com01.append(ts[0]['com'])

                obs01.append(ts[1]['obs'])
                ac_seq01.append(ts[1]['ac_seq'])
                ab_legal01.append(ts[1]['ab_legal'])
                st_action01.append(ts[1]['st_action'])
                action01.append(ts[1]['action'])
                prob01.append(ts[1]['prob'])
                st_batch01.append(ts[1]['st_batch'])
                ac_seq_batch01.append(ts[1]['ac_seq_batch'])
                val01.append(ts[1]['val'])
                com02.append(ts[1]['com'])

            return obs, ac_seq, ab_legal, st_action, action, prob, st_batch, ac_seq_batch, val, obs01, ac_seq01, ab_legal01, st_action01, action01, prob01, st_batch01, ac_seq_batch01, val01,com01,com02
"""


class RMemory(object):
    def __init__(self, size):
        self.memory = []
        self.size = size

    def addmemory(self, mem):
        if len(self.memory) > self.size:
            self.memory.pop(0)
        self.memory.append(mem)

    def res_sampling(self, lis, k):
        result = lis[:k]  # 前k个元素就不模拟数据流了，直接切片读取了
        i = 0
        while True:
            try:
                ele = lis[k + i]  # 尝试读取下一个数据，如果读取失败，说明数据流结束
            except:
                return result  # 数据流结束，返回结果
            if random.randint(0, k + i) < k:
                result[random.randint(0, k - 1)] = ele  # 如果满足替换条件进行替换
            i += 1
        return result

    def sample_res(self, num):

        obs = []
        seq = []
        act = []
        val = []
        prob = []

        if len(self.memory) <= num:
            rs = self.memory
            random.shuffle(rs)
            for ts in rs:
                obs.append(ts['obs'])
                seq.append(ts['seq'])
                act.append(ts['action'])
                val.append(ts['val'])
                prob.append(ts['prob'])

            return obs, seq, act, val, prob
        else:

            rs = self.res_sampling(self.memory, num)
            random.shuffle(rs)
            for ts in rs:
                obs.append(ts['obs'])
                seq.append(ts['seq'])
                act.append(ts['action'])
                val.append(ts['val'])
                prob.append(ts['prob'])

            return obs, seq, act, val, prob

    def sample(self, num):

        obs = []
        seq = []
        act = []
        val = []
        prob = []
        com = []

        if len(self.memory) <= num:
            rs = self.memory
            random.shuffle(rs)
            for ts in rs:
                obs.append(ts['obs'])
                seq.append(ts['seq'])
                act.append(ts['action'])
                val.append(ts['val'])
                prob.append(ts['prob'])
                com.append(ts['combine'])

            return obs, seq, act, val, prob, com
        else:

            rs = random.sample(self.memory, num)
            random.shuffle(rs)
            for ts in rs:
                obs.append(ts['obs'])
                seq.append(ts['seq'])
                act.append(ts['action'])
                val.append(ts['val'])
                prob.append(ts['prob'])
                com.append(ts['combine'])

            return obs, seq, act, val, prob, com

    def sample_peas(self, num):

        obs01 = []
        seq01 = []
        act01 = []
        val01 = []
        prob01 = []
        combine01 = []

        obs02 = []
        seq02 = []
        act02 = []
        val02 = []
        prob02 = []
        combine02 = []

        kdatas01 = []
        kdatas02 = []

        if len(self.memory) <= num:
            rs = self.memory
            random.shuffle(rs)

            for ts in rs:
                obs01.append(ts[0]['obs'])
                seq01.append(ts[0]['seq'])
                act01.append(ts[0]['action'])
                val01.append(ts[0]['val'])
                prob01.append(ts[0]['prob'])
                combine01.append(ts[0]['combine'])

                obs02.append(ts[1]['obs'])
                seq02.append(ts[1]['seq'])
                act02.append(ts[1]['action'])
                val02.append(ts[1]['val'])
                prob02.append(ts[1]['prob'])
                combine02.append(ts[1]['combine'])

            return obs01, seq01, act01, val01, prob01, obs02, seq02, act02, val02, prob02, combine01, combine02
        else:
            rs = random.sample(self.memory, num)
            random.shuffle(rs)

            for ts in rs:
                obs01.append(ts[0]['obs'])
                seq01.append(ts[0]['seq'])
                act01.append(ts[0]['action'])
                val01.append(ts[0]['val'])
                prob01.append(ts[0]['prob'])
                combine01.append(ts[0]['combine'])

                obs02.append(ts[1]['obs'])
                seq02.append(ts[1]['seq'])
                act02.append(ts[1]['action'])
                val02.append(ts[1]['val'])
                prob02.append(ts[1]['prob'])
                combine02.append(ts[1]['combine'])

            return obs01, seq01, act01, val01, prob01, obs02, seq02, act02, val02, prob02, combine01, combine02

    def ksample(self, num):
        obs = []
        seq = []
        act = []
        val = []
        prob = []

        if len(self.memory) <= num:
            rs = self.memory
            for ts in rs:
                obs.append(ts['k_state'])
                act.append(ts['k_action'])
                seq.append(ts['k_seq'])
                val.append(ts['k_reward'])
                prob.append(ts['k_prob'])
            return obs, seq, act, val, prob
        else:
            rs = random.sample(self.memory, num)
            for ts in rs:
                obs.append(ts['k_state'])
                act.append(ts['k_action'])
                seq.append(ts['k_seq'])
                val.append(ts['k_reward'])
                prob.append(ts['k_prob'])

            return obs, seq, act, val, prob

    def clear(self):
        self.memory = []

    def get_size(self):
        return len(self.memory)