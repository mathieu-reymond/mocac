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


def utility(values):
    debt = 45; deadline = 10; penalty = -10
    ut = F.softplus(values[:,0]-debt)
    # everything lower than deadline yields 0, otherwise, additional steps are squared
    uf = -(values[:,1].abs()-deadline).clamp(0)**2
    uf[uf.nonzero()] += penalty
    return (ut+uf).view(-1,1)

def stepwise_utility(values, utility_function, gamma):
    utilities = []
    for v in values:
        steps = -int(v[1])
        utility = utility_function(torch.tensor([[v[0], -1.]]))
        for _ in range(steps-1):
            utility = utility*gamma + utility_function(torch.tensor([[0., -1.]]))
        utilities.append(utility.item())
    return torch.tensor(utilities).view(-1, 1)


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
        v = torch.sum(values*weights, dim=-1, keepdim=True)
        v = (v-min_u)/(max_u-min_u)
        return v
    return utility


if __name__ == '__main__':
    from agents.mocac import MOCAC
    from policies.policy import Categorical
    from memory.memory import Memory
    from gym.wrappers import TimeLimit
    from wrappers.one_hot import OneHotEnv
    from envs.dst import DeepSeaTreasureEnv
    from log.plotter import Plotter
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--w', default=0., type=float)
    parser.add_argument('--timesteps', default=1000000, type=int)
    args = parser.parse_args()
    print(args)

    c = 11
    gamma = args.gamma
    w = args.w
    n_steps_update = 10
    e_coef = 0.1
    env = DeepSeaTreasureEnv()
    env = TimeLimit(env, 100)
    env = OneHotEnv(env, env.nS)
    actor = Actor(env.nS)
    critic = Critic(env.nS, c=c)
    utility_function = utility
    # utility_function = make_weighted_sum(env, gamma, torch.tensor([[w, 1.-w]]), True)

    logdir = f'runs/deep_sea_treasure/gamma_{gamma}/w_{w}/lr_{args.lr}/e_coef_{e_coef}/n_steps_update_{n_steps_update}/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    agent = MOCAC(
        env,
        Categorical(),
        Memory(),
        actor,
        critic,
        gamma=gamma,
        lr=args.lr,
        utility=utility_function,
        logdir=logdir,
        c=c,
        e_coef=e_coef,
        n_steps_update=n_steps_update,
        v_min=[0., -20.1],
        v_max=[100.1, 0.1],
    )

    agent.train(timesteps=args.timesteps, eval_freq=np.inf)
    breakpoint()
    Plotter(logdir)

    returns = all_returns(env, gamma)
    print(utility_function(returns))
    print(stepwise_utility(returns, utility_function, gamma))
