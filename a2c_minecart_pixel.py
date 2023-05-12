import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from datetime import datetime
import uuid


class Flatten(nn.Module):

    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class Actor(nn.Module):

    def __init__(self, nS, nA):
        super(Actor, self).__init__()

        self.nS = nS
        self.nA = nA

        self.common = nn.Sequential(
            nn.Conv2d(nS[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64, 20),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, nA),
        )
        def ortho(m, gain):
            if hasattr(m, 'weight'):
                nn.init.orthogonal_(m.weight, gain=gain)
        self.common.apply(lambda m: ortho(m, np.sqrt(2)))
        self.actor.apply(lambda m: ortho(m, 0.01))

    def forward(self, state):
        x = self.common(state)
        x = self.actor(x)
        x = F.log_softmax(x, dim=1)

        return x


class Critic(nn.Module):

    def __init__(self, actor, device='cpu'):
        super(Critic, self).__init__()

        self.device = device

        self.common = actor.common
        self.critic = nn.Sequential(
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        def ortho(m, gain):
            if hasattr(m, 'weight'):
                nn.init.orthogonal_(m.weight, gain=gain)
        self.critic.apply(lambda m: ortho(m, np.sqrt(2)))

    def forward(self, state):

        x = self.common(state)
        x = self.critic(x)
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
    values = torch.from_numpy(values)
    ores, fuel = values[:,0], values[:,1]
    target = 0.7; contract_price = 5.; market_price = 7.; compensation = 2.
    penalty = ores < target
    sales = ores.clamp(max=target)*contract_price + (ores-target).clamp(min=0)*market_price + - compensation*penalty
    return (sales + fuel/20.).view(-1,).numpy()


if __name__ == '__main__':
    from agents.a2c import A2C
    from policies.policy import Categorical, Normal
    from memory.memory import Memory
    from gym.wrappers import TimeLimit
    from wrappers.one_hot import OneHotEnv
    from wrappers.weighted_sum import WeightedSum
    from wrappers.terminal import TerminalEnv
    from wrappers.atari import Rescale42x42, NormalizedEnv
    from wrappers.minecart_pixel import PixelMinecart
    from wrappers.history import History
    import argparse
    import os
    import envs.minecart

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--gamma', default=1.00, type=float)
    parser.add_argument('--e-coef', default=0.1, type=float)
    parser.add_argument('--n-steps-update', default=200, type=int)
    parser.add_argument('--clip-grad-norm', default=50, type=float)
    parser.add_argument('--timesteps', default=20000000, type=int)
    args = parser.parse_args()
    print(args)

    device='cpu'

    gamma = args.gamma
    n_steps_update = args.n_steps_update
    e_coef = args.e_coef
    clip_grad_norm = args.clip_grad_norm
    env = gym.make('MinecartDeterministic-v0')
    env = TimeLimit(env, 1000)
    env = OneOre(env)

    env = PixelMinecart(env)
    env = Rescale42x42(env)
    env = NormalizedEnv(env)
    env = History(env, history=2)
    # env = DiscretizeEnv(env)
    env = TerminalEnv(env , utility_contract_2d)
    # env = TimestepEnv(env, utility)
    # env = WeightedSum(env, np.array([0., 0.99, 0.01]))
    nS = env.observation_space.shape
    actor = Actor(nS, env.action_space.n).to(device)
    critic = Critic(actor).to(device)

    logdir = f'runs/minecart_contract_pixel/head_20-20/history_2/terminal_a2c/gamma_{gamma}/lr_{args.lr}/e_coef_{e_coef}/n_steps_update_{n_steps_update}/clip_grad_norm_{clip_grad_norm}'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    agent = A2C(
        env,
        Categorical(),
        Memory(device=device),
        actor,
        critic,
        gamma=gamma,
        lr=args.lr,
        logdir=logdir,
        e_coef=e_coef,
        n_steps_update=n_steps_update,
        # scheduler='linear',
        # scheduler_steps=args.timesteps//n_steps_update,
        clip_grad_norm=clip_grad_norm
    )

    agent.train(timesteps=args.timesteps)
    # from log.plotter import Plotter
    # Plotter(logdir)
