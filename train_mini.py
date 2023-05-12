import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from datetime import datetime
import uuid


class Actor(nn.Module):

    def __init__(self, nS, nA):
        super(Actor, self).__init__()
        self.out = nn.Sequential(
            nn.Linear(nS, 50),
            nn.Tanh(),
            nn.Linear(50, nA)
        )

    def forward(self, x):
        x = self.out(x)
        # normalized sigmoid
        x = F.sigmoid(x)
        x = x/torch.sum(x, dim=-1, keepdim=True)
        # logprob
        x = torch.log(x)
        # x = F.log_softmax(x, dim=-1)
        return x


class Critic(nn.Module):

    def __init__(self, nS, c=11):
        super(Critic, self).__init__()

        self.c = c

        self.common = nn.Sequential(
            nn.Linear(nS, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
        )
        self.critic = nn.Linear(50, c**2)

    def forward(self, x):
        x = self.common(x)
        x = self.critic(x)
        x = F.softmax(x, dim=1)
        x = x.view(-1, self.c, self.c)
        return x


def utility_mul(rewards):
    u = rewards[:,0]*rewards[:,1]
    return u.view(-1,1)


class AccruedWrapper(gym.Wrapper):

    def reset(self):
        obs = super(AccruedWrapper, self).reset()
        self.accrued = np.zeros_like(self.reward_space.low)
        obs = np.concatenate((obs, self.accrued), axis=-1)

        return obs

    def step(self, action):
        obs, r, d, info = super(AccruedWrapper, self).step(action)
        self.accrued += r*0.1
        obs = np.concatenate((obs, self.accrued), axis=-1)
        return obs, r, d, info


if __name__ == '__main__':
    from agents.moreinforce import MOReinforce
    from agents.mocac import MOCAC
    from policies.policy import Categorical
    from memory.memory import EpisodeMemory, Memory
    from wrappers.one_hot import OneHotEnv
    from wrappers.terminal import TerminalEnv
    from log.plotter import Plotter
    import argparse
    import envs.minirandom

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gamma', default=1.00, type=float)
    parser.add_argument('--timesteps', default=20000, type=int)
    parser.add_argument('--acc', default=False, action='store_true')
    parser.add_argument('--algo', default='moreinforce', type=str)
    args = parser.parse_args()
    print(args)

    gamma = args.gamma; nS = 5; nA = 3
    acc = args.acc
    env = gym.make('MiniRandom-v0')
    env = OneHotEnv(env, nS)
    if acc:
        env = AccruedWrapper(env)
    # env = TerminalEnv(env , utility)
    actor = Actor(nS+acc*2, nA)
    utility_function = utility_mul
    # utility_function = utility

    logdir = f'runs/minirandom/{args.algo}/rplus/acc_{acc}/gamma_{gamma}/lr_{args.lr}/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    if args.algo == 'moreinforce':
        agent = MOReinforce(
            env,
            Categorical(),
            EpisodeMemory(),
            actor,
            utility=utility_function,
            gamma=gamma,
            lr=args.lr,
            logdir=logdir,
        )

    else:
        e_coef = 0.01
        n_steps_update = 1
        c = 7
        critic = Critic(nS+acc*2, c=c)
        agent = MOCAC(
            env,
            Categorical(),
            Memory(),
            actor,
            critic,
            gamma=gamma,
            lr=args.lr,
            logdir=logdir,
            e_coef=e_coef,
            n_steps_update=n_steps_update,
            # scheduler='linear',
            # scheduler_steps=args.timesteps//n_steps_update,
            clip_grad_norm=50,
            c=c,
            v_min=[0, 0.],
            v_max=[7, 7.],
            utility=utility_function
        )

    agent.train(timesteps=args.timesteps) #, eval_freq=0.1)
    # Plotter(logdir)
