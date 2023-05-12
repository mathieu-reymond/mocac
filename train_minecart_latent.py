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
            nn.Linear(nS,20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20,nA)
        )
        # self.out = nn.Linear(7,2)

    def forward(self, x):
        x = self.out(x)
        # x = x.reshape(-1, 21, 1)
        x = F.log_softmax(x, dim=1)
        return x


class Critic(nn.Module):

    def __init__(self, nS, c=11, nO=2):
        super(Critic, self).__init__()
        self.c = c
        self.nO = nO

        self.common = nn.Sequential(
            nn.Linear(nS, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
        )
        self.critic = nn.Linear(20, c**self.nO)

    def forward(self, x):
        x = self.common(x)
        x = self.critic(x)
        x = F.softmax(x, dim=1)
        x = x.view(-1, *([self.c]*self.nO))
        return x


class TimestepEnv(gym.RewardWrapper):

    def __init__(self, env, utility):
        super(TimestepEnv, self).__init__(env)
        self.utility = utility

    def reward(self, rew):
        rew = self.utility(rew.astype(np.float32).reshape(1, -1)).reshape(-1)
        return rew

class RewardArray(gym.RewardWrapper):

    def reward(self, rew):
        return np.array([rew], dtype=np.float32)


class OneOre(gym.RewardWrapper):

    def __init__(self, *args, **kwargs):
        super(OneOre, self).__init__(*args, **kwargs)
        self.reward_space = gym.spaces.Box(low=self.reward_space.low[1:], high=self.reward_space.high[1:])

    def reward(self, rew):
        return rew[1:]


def utility_contract_2d(values):
    ores, fuel = values[:,0], values[:,1]
    target = 0.7; contract_price = 5.; market_price = 7.; compensation = 2.
    penalty = ores < target
    sales = ores.clamp(max=target)*contract_price + (ores-target).clamp(min=0)*market_price + - compensation*penalty
    return (sales + fuel/20.).view(-1, 1)


if __name__ == '__main__':
    from agents.mocac import MOCAC
    from policies.policy import Categorical, Normal
    from memory.memory import Memory
    from gym.wrappers import TimeLimit
    from wrappers.one_hot import OneHotEnv
    from wrappers.weighted_sum import WeightedSum
    from wrappers.terminal import TerminalEnv
    from log.plotter import Plotter
    import argparse
    import envs.minecart

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--gamma', default=1.00, type=float)
    parser.add_argument('--e-coef', default=0.1, type=float)
    parser.add_argument('--n-steps-update', default=200, type=int)
    parser.add_argument('--clip-grad-norm', default=50, type=float)
    parser.add_argument('--c', default=11, type=int)
    parser.add_argument('--timesteps', default=20000000, type=int)
    args = parser.parse_args()
    print(args)

    c = args.c
    gamma = args.gamma
    n_steps_update = args.n_steps_update
    e_coef = args.e_coef
    clip_grad_norm = args.clip_grad_norm
    env = gym.make('MinecartDeterministic-v0')
    env = TimeLimit(env, 1000)
    env = OneOre(env)
    
    nS = env.observation_space.shape[0]
    actor = Actor(nS, env.action_space.n)
    critic = Critic(nS, c=c, nO=np.prod(env.reward_space.shape))

    logdir = f'runs/minecart_contract_latent/mocac/c_{c}/gamma_{gamma}/lr_{args.lr}/e_coef_{e_coef}/n_steps_update_{n_steps_update}/clip_grad_norm_{clip_grad_norm}/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

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
        clip_grad_norm=clip_grad_norm,
        c=c,
        v_min=[0, -4.],
        v_max=[1.5, 0.],
        utility=utility_contract_2d
    )

    agent.train(timesteps=args.timesteps) #, eval_freq=0.1)
