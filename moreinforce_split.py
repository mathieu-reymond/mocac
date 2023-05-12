import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, tunnel):
        super(Actor, self).__init__()
        self.tunnel = tunnel
        self.out = nn.Sequential(
            nn.Linear(2*self.tunnel+6+2,2*self.tunnel),
            nn.Tanh(),
            nn.Linear(2*self.tunnel,2)
        )
        # self.out = nn.Linear(7,2)

    def forward(self, x):
        x, return_ = x[:,0].flatten().long(), x[:,1:]
        x = F.one_hot(x, num_classes=2*self.tunnel+6).float()
        x = torch.cat((x, return_), 1)
        x = self.out(x)
        x = F.log_softmax(x, dim=-1)
        return x


def utility_mul(rewards):
    u = rewards[:,0]*rewards[:,1]
    return u.view(-1,1)


if __name__ == '__main__':
    from policies.policy import Categorical
    from memory.memory import EpisodeMemory
    from agents.moreinforce import MOReinforce
    from envs.split import SplitEnv
    from log.plotter import Plotter

    tunnel = 10
    env = SplitEnv(tunnel)
    actor = Actor(tunnel)
    logdir = f'runs/split_env/tunnel_{tunnel}'

    agent = MOReinforce(
        env,
        Categorical(),
        EpisodeMemory(),
        actor,
        lr=1e-3,
        utility=utility_mul,
        logdir=logdir,
    )

    agent.train(episodes=500) #, eval_freq=0.1)
    Plotter(logdir)