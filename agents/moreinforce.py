import numpy as np
import torch
from agents.agent import NNAgent, Transition
from wrappers.tensor import TensorWrapper


class MOReinforce(NNAgent):

    def __init__(self, env,
                 policy=None,
                 memory=None,
                 actor=None,
                 gamma=1.,
                 lr=1e-3,
                 utility=None,
                 **nn_kwargs):
        optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        super(MOReinforce, self).__init__(optimizer=optimizer, **nn_kwargs)
        # tensor wrapper to make batches of steps, and convert np.arrays to tensors
        env = TensorWrapper(env)
        self.env = env
        self.actor = actor
        self.policy = policy
        self.memory = memory

        self.gamma = gamma
        self.utility = utility
        self.accrued = torch.tensor([]).view(0, 2)

    def start(self, log=None):
        obs = self.env.reset()
        return {'observation': obs,
                'terminal': False}

    def step(self, previous, log=None):
        with torch.no_grad():
            actor_out = self.actor(previous['observation'])
            action = self.policy(actor_out, log=log)
        next_obs, reward, terminal, _ = self.env.step(action)

        gamma = torch.tensor([self.gamma**log.episode_step])

        if log.episode_step == 0:
            self.accrued = torch.cat((self.accrued[:-1], torch.zeros_like(reward), reward), dim=0)
        else:
            self.accrued = torch.cat((self.accrued, self.accrued[-1] + gamma*reward), dim=0)
        
        t = Transition(observation=previous['observation'],
                       action=action,
                       reward=reward,
                       next_observation=next_obs,
                       terminal=terminal,
                       gamma=gamma)

        self.memory.add(t, log.episode)
        return {'observation': next_obs,
                'action': action,
                'reward': reward,
                'terminal': terminal}

    def end(self, log=None, writer=None):
        episode = self.memory.get_episode()
        # rewards to discounted returns
        returns = episode.reward.clone()
        for i in range(len(returns)-1, 0, -1):
            returns[i-1] += self.gamma * returns[i]
        # keep final episodic return and ask for its utility, repeat for every timestep
        # utility = self.utility(returns)
        utility = self.utility(returns[0:1])
        utility = utility.repeat(len(returns), 1)
            
        # reinforce loss is NLLLoss
        actor_out = self.actor(episode.observation)
        log_prob = self.policy.log_prob(episode.action, actor_out)
        loss = -log_prob*utility
        self.optimizer_step(loss)

        self.accrued = self.accrued[-1:]

    def evalstep(self, previous, log=None):
        actor_out = self.actor(previous['observation'])
        action = self.policy(actor_out, log=log)
        next_obs, reward, terminal, info = self.env.step(action)
        return {'observation': next_obs,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'env_info': info}

    def state_dict(self):
        return {'actor': self.actor.state_dict(),
                'policy': self.policy,
                'memory': self.memory}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])
        self.policy = sd['policy']
        self.memory = sd['memory']
