import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gym
import normflow_policy

from tianshou.env import SubprocVectorEnv

from tianshou.data import Collector, ReplayBuffer, Batch
from tianshou.utils.net.common import Net
from tianshou.policy.imitation.base import ImitationPolicy

from normflow_policy.normflow_ds import NormalizingFlowDynamicalSystem, NormalizingFlowDynamicalSystemActor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='UJICharHandwriting-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args

def prepare_dataset(env):
    pos_traj = env.tar_traj
    vel_traj = (pos_traj[1:, :] - pos_traj[:-1, :])/env.dt
    act_traj = (vel_traj[1:,:] - vel_traj[:-1,:])/env.dt/env.m

    #construct batch, need dummy info entry
    return Batch(act=act_traj, obs=np.concatenate((pos_traj[:-2, :], vel_traj[:-1, :]), axis=1), info=np.zeros(act_traj.shape))

def test_bc(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.max_action = env.action_space.high[0]

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    # net = Net(args.layer_num, args.state_shape, device=args.device)
    net = NormalizingFlowDynamicalSystem(dim=np.prod(args.state_shape)//2, n_flows=args.layer_num, device=args.device)
    actor = NormalizingFlowDynamicalSystemActor(
        net, args.action_shape,
        args.max_action, args.device
    ).to(args.device)

    optim = torch.optim.Adam(
        net.parameters(), lr=args.lr)

    policy = ImitationPolicy(actor, optim, mode='continuous')

    dataset = prepare_dataset(env)

    # log
    log_path = os.path.join(args.logdir, args.task, 'bc')
    writer = SummaryWriter(log_path)

    for epoch in range(1, 1 + args.epoch):
        for batch in dataset.split(size=args.batch_size, shuffle=True):
            result = policy.learn(batch)
            print(result)
        
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        collector = Collector(policy, env, preprocess_fn=None)
        result = collector.collect(n_step=100, render=args.render)
        print('Final reward: {0}, length: {1}'.format(result["rew"], result["len"]))
        collector.close()


if __name__ == '__main__':
    test_bc()