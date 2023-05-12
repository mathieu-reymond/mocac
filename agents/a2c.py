import numpy as np
import torch
from itertools import chain
from agents.agent import NNAgent, Transition
from wrappers.tensor import TensorWrapper


class A2C(NNAgent):

    def __init__(self, env,
                 policy=None,
                 memory=None,
                 actor=None,
                 critic=None,
                 gamma=1.,
                 n_steps_update=1,
                 lr=1e-3,
                 v_coef=0.5,
                 e_coef=0.01,
                 **nn_kwargs):

        params = chain(actor.parameters(), critic.parameters())
        # if actor and critic share layers, only insert them once in the optimizer
        # do not use a set as it is unordered
        params = list(dict.fromkeys(params))
        optimizer = torch.optim.Adam(params, lr=lr)
        super(A2C, self).__init__(optimizer=optimizer, **nn_kwargs)
        # tensor wrapper to make batches of steps, and convert np.arrays to tensors
        env = TensorWrapper(env)
        self.env = env
        self.actor = actor
        self.critic = critic
        self.policy = policy
        self.memory = memory

        self.gamma = gamma
        self.n_steps_update = n_steps_update
        self.v_coef = v_coef
        self.e_coef = e_coef

    def start(self, log=None):
        obs = self.env.reset()
        return {'observation': obs,
                'terminal': False}

    def step(self, previous, log=None):
        with torch.no_grad():
            actor_out = self.actor(previous['observation'])
            action = self.policy(actor_out, log=log)
        # observations, actions are in batch, and tensor types for nn
        next_obs, reward, terminal, _ = self.env.step(action)

        t = Transition(observation=previous['observation'],
                       action=action,
                       reward=reward,
                       next_observation=next_obs,
                       terminal=terminal)

        self.memory.add(t)

        # update every n steps
        # if ((log.episode_step + 1) % self.n_steps_update == 0) or terminal:
        #     last = (log.episode_step%self.n_steps_update)+1 if terminal else self.n_steps_update
        #     batch = self.memory.last(last)
        if (log.total_steps + 1) % self.n_steps_update == 0:
            batch = self.memory.last(self.n_steps_update)
            # get V(s) of all batch, need V(s') of last sample for n-step return
            obs = torch.cat((batch.observation, batch.next_observation[-1:]), 0)
            v_s_ns = self.critic(obs)
            # detach() value of next state, to not propagate gradients from returns
            v_s, v_ns = v_s_ns[:-1], v_s_ns[-1:].detach()
            # rewards to discounted returns
            returns = batch.reward.clone()
            non_terminal = torch.logical_not(batch.terminal)
            # add v of next-state if not terminal
            returns[-1] += self.gamma * v_ns[-1] * non_terminal[-1]
            for i in range(len(returns)-1, 0, -1):
                # if episode ended in last n-steps, do not add to return
                returns[i-1] += self.gamma * returns[i] * non_terminal[i-1]

            advantage = returns - v_s
            # get nllloss from actor
            actor_out = self.actor(batch.observation)
            log_prob = self.policy.log_prob(batch.action, actor_out)
            # don't propagate gradients from advantage
            actor_loss = -log_prob*advantage.detach()
            # entropy
            entropy = -torch.exp(log_prob) * log_prob
            # mseloss for critic
            critic_loss = advantage.pow(2)

            loss = actor_loss + self.v_coef*critic_loss - self.e_coef*entropy
            self.optimizer_step(loss)

        return {'observation': next_obs,
                'action': action,
                'reward': reward,
                'terminal': terminal}


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
                'critic': self.critic.state_dict(),
                'policy': self.policy,
                'memory': self.memory}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])
        self.critic.load_state_dict(sd['critic'])
        self.policy = sd['policy']
        self.memory = sd['memory']
