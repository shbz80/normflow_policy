import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gym
import normflow_policy

from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.policy.dist import DiagGaussian

from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

from normflow_policy.normflow_ds import NormalizingFlowDynamicalSystem, NormalizingFlowDynamicalSystemActorProb, NormalizingFlowDynamicalSystemPPO

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='UJICharHandwriting-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--collect-per-step', type=int, default=1)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.01)
    parser.add_argument('--policy', type=str, default='ppo')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    args = parser.parse_known_args()[0]
    return args

if __name__ == '__main__':
    args=get_args()
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.max_action = env.action_space.high[0]

    if args.policy == 'ppo':
        net = Net(args.layer_num, args.state_shape, device=args.device)
        actor = ActorProb(
            net, args.action_shape,
            args.max_action, args.device
        ).to(args.device)
        critic = Critic(Net(
            args.layer_num, args.state_shape, device=args.device
        ), device=args.device).to(args.device)
        optim = torch.optim.Adam(list(
            actor.parameters()) + list(critic.parameters()), lr=args.lr)
        dist = DiagGaussian
        policy = PPOPolicy(
            actor, critic, optim, dist, args.gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            reward_normalization=args.rew_norm,
            value_clip=args.value_clip,
            gae_lambda=args.gae_lambda)
        log_path = os.path.join(args.logdir, args.task, 'ppo')

    else:
        nf_net = NormalizingFlowDynamicalSystem(dim=np.prod(args.state_shape)//2, n_flows=args.layer_num, device=args.device)
        actor = NormalizingFlowDynamicalSystemActorProb(
            nf_net, args.action_shape,
            args.max_action, args.device
        ).to(args.device)

        critic = Critic(Net(
            args.layer_num, args.state_shape, device=args.device
        ), device=args.device).to(args.device)

        optim = torch.optim.Adam(list(
            actor.parameters()) + list(critic.parameters()), lr=args.lr)

        policy = NormalizingFlowDynamicalSystemPPO(
            actor, critic, optim, args.gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            reward_normalization=args.rew_norm,
            value_clip=args.value_clip,
            gae_lambda=args.gae_lambda)
        log_path = os.path.join(args.logdir, args.task, 'normflow_ds_ppo')

    policy_path = os.path.join(log_path, 'policy.pth')
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()

    collector = Collector(policy, env, preprocess_fn=None)
    result = collector.collect(n_step=100, render=args.render)
    print('Final reward: {0}, length: {1}'.format(result["rew"], result["len"]))
    collector.close()