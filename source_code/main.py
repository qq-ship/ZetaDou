from __future__ import print_function

import argparse
import os
import torch
from net_learner.zeta_model import PNet,PKNet
import multiprocessing as mp

from multiprocessing import shared_memory,Manager
from optim_equ.my_optim import SharedAdam
from multiprocessing.managers import BaseManager
from game_model.zeta_doudizhu_game import train,worker

from game_model.auxiliary_means.mem_class import Memory, Memseq, RMemory
import time
#torch.cuda.set_device(1)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    processes = []
    lock = mp.Lock()

    manager = BaseManager()
    manager.register('Rmem',RMemory)  # 第一个参数为类型id，通常和第二个参数传入的类的类名相同，直观并且便于阅读
    manager.start()

    rmem = manager.Rmem(800000) #80万
    rmem01 = manager.Rmem(800000)
    rmem02 = manager.Rmem(800000)

    kmem = manager.Rmem(50000)
    kmem01 = manager.Rmem(50000)
    kmem02 = manager.Rmem(50000)

    dd = 0
    lnet = PNet(309, device_id=dd, noise=0.017).cuda(dd)  # 0.085
    lnet_d = PNet(309, device_id=dd, noise=0.017).cuda(dd)  # 0.085
    lnet_u = PNet(309, device_id=dd, noise=0.017).cuda(dd)  # 0.085

    kicker = PKNet(28, device_id=dd, noise=0.085).cuda(dd)  # 0.085
    kicker_d = PKNet(28, device_id=dd, noise=0.085).cuda(dd)  # 0.085
    kicker_u = PKNet(28, device_id=dd, noise=0.085).cuda(dd)  # 0.085

    if os.path.exists(r'game_model/model/nework.pth'):
        print('load the model game_model/model/nework.pth')
        lnet.load_state_dict(torch.load(r'game_model/model/nework.pth'))
    else:
        print('create nework.pth')

    if os.path.exists(r'game_model/model/nework01.pth'):
        print('load the model game_model/model/nework01.pth')
        lnet_d.load_state_dict(torch.load(r'game_model/model/nework01.pth'))
    else:
        print('create nework01.pth')

    if os.path.exists(r'game_model/model/nework02.pth'):
        print('load the model game_model/model/nework02.pth')
        lnet_u.load_state_dict(torch.load(r'game_model/model/nework02.pth'))
    else:
        print('create nework02.pth')

    print("starting the knet#############################################")

    if os.path.exists(r'game_model/model/knework.pth'):
        print('load the model game_model/model/knework.pth')
        kicker.load_state_dict(torch.load(r'game_model/model/knework.pth'))
    else:
        print('create knework.pth')

    if os.path.exists(r'game_model/model/knework01.pth'):
        print('load the model game_model/model/knework01.pth')
        kicker_d.load_state_dict(torch.load(r'game_model/model/knework01.pth'))
    else:
        print('create knework01.pth')

    if os.path.exists(r'game_model/model/knework02.pth'):
        print('load the model game_model/model/knework02.pth')
        kicker_u.load_state_dict(torch.load(r'game_model/model/knework02.pth'))
    else:
        print('create knework02.pth')

    lnet.share_memory()
    lnet_d.share_memory()
    lnet_u.share_memory()

    kicker.share_memory()
    kicker_d.share_memory()
    kicker_u.share_memory()

    pop = SharedAdam([{'params': lnet.parameters()}, {'params': kicker.parameters()}], 0.0001)
    pop_d = SharedAdam([{'params': lnet_d.parameters()}, {'params': kicker_d.parameters()}], 0.0001)
    pop_u = SharedAdam([{'params': lnet_u.parameters()}, {'params': kicker_u.parameters()}], 0.0001)

    pop.share_memory()
    pop_d.share_memory()
    pop_u.share_memory()
    
    for rank in range(0,3):
        p = mp.Process(target=worker, args=(rank, lnet,lnet_d,lnet_u,kicker,kicker_d,kicker_u, rmem, rmem01, rmem02, kmem, kmem01, kmem02, lock, 0))
        p.start()
        processes.append(p)

    train(rank, lnet, lnet_d, lnet_u, kicker, kicker_d, kicker_u, pop, pop_d, pop_u, rmem, rmem01, rmem02, kmem, kmem01,kmem02, lock, 0)
    """
    p = mp.Process(target=train,args=(lnet,lnet_d,lnet_u,kicker,kicker_d,kicker_u, pop, pop_d,pop_u,rmem, rmem01, rmem02, kmem, kmem01, kmem02,))
    p.start()
    processes.append(p)
    """
    for p in processes:
        p.join()

