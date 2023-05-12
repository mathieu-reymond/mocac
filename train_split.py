import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, tunnel):
        super(Actor, self).__init__()
        self.tunnel = tunnel
        self.out = nn.Sequential(
            nn.Linear(2*self.tunnel+6,2*self.tunnel),
            nn.Tanh(),
            nn.Linear(2*self.tunnel,2)
        )
        # self.out = nn.Linear(7,2)

    def forward(self, x):
        x = x.flatten().long()
        x = F.one_hot(x, num_classes=2*self.tunnel+6).float()
        x = self.out(x)
        x = F.log_softmax(x, dim=-1)
        return x


class C_Critic(nn.Module):

    def __init__(self, tunnel, c=11):
        super(C_Critic, self).__init__()

        self.c = c
        self.tunnel = tunnel

        self.common = nn.Sequential(
            nn.Linear(2*self.tunnel+6, 50),
            nn.Tanh()
        )
        self.critic = nn.Linear(50, c**2)

    def forward(self, x):
        x = x.flatten().long()
        x = F.one_hot(x, num_classes=2*self.tunnel+6).float()
        x = self.common(x)
        x = self.critic(x)
        x = F.softmax(x, dim=1)
        x = x.view(-1, self.c, self.c)
        return x


class Critic(nn.Module):

    def __init__(self, tunnel):
        super(Critic, self).__init__()

        self.tunnel = tunnel

        self.common = nn.Sequential(
            nn.Linear(2*self.tunnel+6, 50),
            nn.Tanh()
        )
        self.critic = nn.Linear(50, 2)

    def forward(self, x):
        x = x.flatten().long()
        x = F.one_hot(x, num_classes=2*self.tunnel+6).float()
        x = self.common(x)
        x = self.critic(x)
        return x


def utility_mul(rewards):
    u = rewards[:,0]*rewards[:,1]
    return u.view(-1,1)


if __name__ == '__main__':
    from policies.policy import Categorical
    from memory.memory import Memory, EpisodeMemory
    from agents.mocac import MOCAC
    from agents.moac import MOAC
    from agents.moreinforce import MOReinforce
    from envs.split import SplitEnv
    from log.plotter import Plotter
    from datetime import datetime
    import uuid
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--algo', default='moreinforce', type=str)
    args = parser.parse_args()
    print(args)

    tunnel = 10
    c=11
    env = SplitEnv(tunnel)
    logdir = f'runs/split_env/tunnel_{tunnel}/{args.algo}/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    if args.algo == 'mocac':
        actor = Actor(tunnel)
        critic = C_Critic(tunnel,c=c)
        agent = MOCAC(
            env,
            Categorical(),
            Memory(),
            actor,
            critic,
            lr=1e-3,
            utility=utility_mul,
            logdir=logdir,
            c=c,
            n_steps_update=1,
            v_min=[-1.1, -1.1],
            v_max=[10.1, 10.1],
            v_coef=1.
        )
    elif args.algo == 'moac':
        actor = Actor(tunnel)
        critic = Critic(tunnel)
        agent = MOAC(
            env,
            Categorical(),
            Memory(),
            actor,
            critic,
            lr=1e-3,
            utility=utility_mul,
            logdir=logdir,
            n_steps_update=1,
            v_coef=1.,
        )
    else:
        actor = Actor(tunnel)
        agent = MOReinforce(
            env,
            Categorical(),
            EpisodeMemory(),
            actor,
            utility=utility_mul,
            gamma=1.,
            lr=1e-3,
            logdir=logdir,
        )

    agent.train(episodes=500, eval_freq=0.1)
    agent.logger.flush()
    Plotter(logdir)
