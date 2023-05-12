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


def utility(values):
    values = torch.from_numpy(values)
    debt = 45; deadline = 10; penalty = -10
    ut = F.softplus(values[:,0]-debt)
    # everything lower than deadline yields 0, otherwise, additional steps are squared
    uf = -(values[:,1].abs()-deadline).clamp(0)**2
    uf[uf.nonzero()] += penalty
    return (ut+uf).view(-1,1).numpy().flatten()


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


class NormalizedEnv(gym.RewardWrapper):

    def __init__(self, env, weights, gamma):
        super(NormalizedEnv, self).__init__(env)
        returns = all_returns(DeepSeaTreasureEnv(), gamma)
        values = torch.sum(returns*weights, dim=-1)
        self.min_u = values.min().item()
        self.max_u = values.max().item()
    
    def reward(self, rew):
        breakpoint()
        return (rew-self.min_u)/(self.max_u-self.min_u)


class TimestepEnv(gym.RewardWrapper):

    def __init__(self, env, utility):
        super(TimestepEnv, self).__init__(env)
        self.utility = utility

    def reward(self, rew):
        rew = self.utility(rew.astype(np.float32).reshape(1, -1)).reshape(-1)
        return rew


if __name__ == '__main__':
    from agents.a2c import A2C
    from policies.policy import Categorical
    from memory.memory import Memory
    from gym.wrappers import TimeLimit
    from wrappers.one_hot import OneHotEnv
    from wrappers.weighted_sum import WeightedSum
    from wrappers.terminal import TerminalEnv
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
    normalize = False
    env = DeepSeaTreasureEnv()
    env = TimeLimit(env, 100)
    env = OneHotEnv(env, env.nS)
    env = TerminalEnv(env , utility)
    # env = TerminalEnv(env , lambda x, w=w: torch.sum(torch.tensor([[w, 1.-w]])*x, dim=1, keepdims=True))
    # env = TimestepEnv(env, utility)
    # env = WeightedSum(env, np.array([w, 1.-w]))
    if normalize:
        env = NormalizedEnv(env, torch.tensor([[w, 1.-w]]), gamma)
    actor = Actor(env.nS)
    critic = Critic(env.nS)

    logdir = f'runs/deep_sea_treasure/a2c/gamma_{gamma}/w_{w}/lr_{args.lr}/e_coef_{e_coef}/n_steps_update_{n_steps_update}/'
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

    returns = all_returns(env, gamma)
