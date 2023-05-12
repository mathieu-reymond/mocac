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


class OneOre(gym.RewardWrapper):

    def __init__(self, *args, **kwargs):
        super(OneOre, self).__init__(*args, **kwargs)
        self.reward_space = gym.spaces.Box(low=self.reward_space.low[1:], high=self.reward_space.high[1:])

    def reward(self, rew):
        return rew[1:]


def utility_contract_2d(values):
    # values = torch.from_numpy(values)
    ores, fuel = values[:,0], values[:,1]
    target = 0.7; contract_price = 5.; market_price = 7.; compensation = 2.
    penalty = ores < target
    sales = ores.clamp(max=target)*contract_price + (ores-target).clamp(min=0)*market_price + - compensation*penalty
    # return (sales + fuel/20.).view(-1).numpy()
    return (sales + fuel/20.).view(-1, 1)


if __name__ == '__main__':
    from agents.moreinforce import MOReinforce
    from policies.policy import Categorical
    from memory.memory import EpisodeMemory
    from gym.wrappers import TimeLimit
    from wrappers.terminal import TerminalEnv
    from log.plotter import Plotter
    import argparse
    import envs.minecart

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--gamma', default=1.00, type=float)
    parser.add_argument('--clip-grad-norm', default=50, type=float)
    parser.add_argument('--timesteps', default=20000000, type=int)
    args = parser.parse_args()
    print(args)

    gamma = args.gamma
    clip_grad_norm = args.clip_grad_norm
    env = gym.make('MinecartDeterministic-v0')
    env = TimeLimit(env, 1000)
    env = OneOre(env)
    nS = env.observation_space.shape[0]
    actor = Actor(nS, env.action_space.n)

    logdir = f'runs/minecart_contract_latent/moreinforce/gamma_{gamma}/lr_{args.lr}/clip_grad_norm_{clip_grad_norm}'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    agent = MOReinforce(
        env,
        Categorical(),
        EpisodeMemory(),
        actor,
        utility=utility_contract_2d,
        gamma=gamma,
        lr=args.lr,
        logdir=logdir,
        clip_grad_norm=clip_grad_norm
    )

    agent.train(timesteps=args.timesteps) #, eval_freq=0.1)
    # Plotter(logdir)
