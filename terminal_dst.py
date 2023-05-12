import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from datetime import datetime
import uuid


class Actor(nn.Module):

    def __init__(self, nS):
        super(Actor, self).__init__()
        self.out = nn.Sequential(
            nn.Linear(nS,50),
            nn.Tanh(),
            nn.Linear(50,4)
        )
        # self.out = nn.Linear(7,2)

    def forward(self, x):
        x = self.out(x)
        x = F.log_softmax(x, dim=-1)
        return x


class Critic(nn.Module):

    def __init__(self, nS):
        super(Critic, self).__init__()

        self.common = nn.Sequential(
            nn.Linear(nS, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
        )
        self.critic = nn.Linear(50, 1)

    def forward(self, x):
        x = self.common(x)
        x = self.critic(x)
        return x


def all_returns(env, gamma):
    returns = torch.empty(0,2)
    for k, v in env.unwrapped._treasures().items():
        steps = sum(k)
        r = torch.tensor([[v*gamma**steps, sum([-1*gamma**i for i in range(steps)])]])
        returns = torch.cat((returns, r), dim=0)
    return returns


def make_weighted_sum(env, gamma, weights, normalize=False):
    min_u, max_u = 0., 1.
    if normalize:
        returns = all_returns(env, gamma)
        values = torch.sum(returns*weights, dim=-1)
        min_u, max_u = values.min(), values.max()
    def utility(values):
        values = torch.tensor(values)
        v = torch.sum(values*weights, dim=-1, keepdim=True)
        v = (v-min_u)/(max_u-min_u)
        return v
    return utility


if __name__ == '__main__':
    from agents.a2c import A2C
    from policies.policy import Categorical
    from memory.memory import Memory
    from gym.wrappers import TimeLimit
    from wrappers.one_hot import OneHotEnv
    from wrappers.terminal import TerminalEnv
    from envs.dst import DeepSeaTreasureEnv
    from log.plotter import Plotter
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--w', default=0., type=float)
    parser.add_argument('--timesteps', default=50000, type=int)
    args = parser.parse_args()
    print(args)

    gamma = args.gamma
    w = args.w
    n_steps_update = 10
    e_coef = 0.1
    env = DeepSeaTreasureEnv()
    env = TimeLimit(env, 100)
    env = OneHotEnv(env, env.nS)
    actor = Actor(env.nS)
    critic = Critic(env.nS)
    utility_function = make_weighted_sum(env, gamma, torch.tensor([[w, 1.-w]]), False)
    env = TerminalEnv(env, utility_function, gamma=gamma)

    logdir = f'runs/deep_sea_treasure/terminal/gamma_{gamma}/w_{w}/lr_{args.lr}/e_coef_{e_coef}/n_steps_update_{n_steps_update}/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    agent = A2C(
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
    )

    agent.train(timesteps=args.timesteps, eval_freq=0.1)
    Plotter(logdir)
