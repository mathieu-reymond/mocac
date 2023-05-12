import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain, product
from agents.agent import NNAgent, Transition
from wrappers.tensor import TensorWrapper
import gym
from pathlib import Path
from collections.abc import Iterable


class MOCAC(NNAgent):

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
                 c=51,
                 v_min=-10.,
                 v_max=10,
                 utility=None,
                 **nn_kwargs):

        if isinstance(lr, Iterable):
            raise ValueError('unsupported for shared layers')
        params = chain(actor.parameters(), critic.parameters())
        # # if actor and critic share layers, only insert them once in the optimizer
        # # do not use a set as it is unordered
        params = list(dict.fromkeys(params))
        optimizer = torch.optim.Adam(params, lr=lr)
        # if not isinstance(lr, Iterable):
        #     lr = [lr, lr]
        # optimizer = torch.optim.Adam([
        #     {'params': actor.parameters(), 'lr': lr[0]},
        #     {'params': critic.parameters(), 'lr': lr[1]}])
        super(MOCAC, self).__init__(optimizer=optimizer, **nn_kwargs)
        self.device = memory.device
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
        self.utility = utility

        self.n_objectives = np.prod(self.env.reward_space.shape)

        self.c = c
        if not isinstance(v_min, Iterable):
            v_min = [v_min]*self.n_objectives
        if not isinstance(v_max, Iterable):
            v_max = [v_max]*self.n_objectives  
        self.v_min = torch.tensor(v_min).to(self.device)
        self.v_max = torch.tensor(v_max).to(self.device)
        assert len(self.v_min) == self.n_objectives and len(self.v_max) == self.n_objectives, \
            'VMIN, VMAX should have a value for each objective'
        # [nO]
        self.d_z = (self.v_max-self.v_min)/(self.c-1.)
        # [C51 nO]
        self.z = torch.arange(c)[:,None].to(self.device)*self.d_z+self.v_min
        # get the utility of every possible V-value (meshgrid over objectives: *[nO Batch])
        r_ = torch.stack(torch.meshgrid(*self.z.T), dim=-1)
        self.r_z = r_.unsqueeze(0) # [1 C51 .. c51 nO]
        self.u_z = self.utility(r_.view(-1,self.n_objectives)).view(1, *([self.c]*self.n_objectives)) # constant

        self.accrued = torch.tensor([]).view(0, self.n_objectives)

    def start(self, log=None):
        obs = self.env.reset()
        # TODO TEST incorporate accrued reward into state
        # obs = torch.cat((obs, torch.zeros((1, 2))), 1)
        return {'observation': obs,
                'terminal': False}

    def step(self, previous, log=None):
        with torch.no_grad():
            actor_out = self.actor(previous['observation'].to(self.device))
            action = self.policy(actor_out, log=log)
        # observations, actions are in batch, and tensor types for nn
        next_obs, reward, terminal, _ = self.env.step(action)

        gamma = torch.tensor([self.gamma**log.episode_step])

        if log.episode_step == 0:
            self.accrued = torch.cat((self.accrued[:-1], torch.zeros_like(reward), reward), dim=0)
        else:
            self.accrued = torch.cat((self.accrued, self.accrued[-1] + gamma*reward), dim=0)

        # TODO TEST incorporate accrued reward into state
        # next_obs = torch.cat((next_obs, self.accrued[-1:]), 1)

        t = Transition(observation=previous['observation'],
                       action=action,
                       reward=reward,
                       next_observation=next_obs,
                       terminal=terminal,
                       gamma=gamma)

        self.memory.add(t)

        # update every n steps
        # if ((log.episode_step + 1) % self.n_steps_update == 0) or terminal:
        #     last = (log.episode_step%self.n_steps_update)+1 if terminal else self.n_steps_update
        #     batch = self.memory.last(last)
        if (log.total_steps + 1) % self.n_steps_update == 0:
            batch = self.memory.last(self.n_steps_update)
            with torch.no_grad():
                # get probs of V(ns) of all batch
                p_ns = self.critic(batch.next_observation)
                # need for every category of the V-distribution the return
                non_terminal = torch.logical_not(batch.terminal).unsqueeze(1)
                s_ = batch.reward.shape
                # [Batch C51 nO]
                returns = batch.reward.unsqueeze(1).expand(s_[0], self.c, s_[1]).clone()
                # [C51 nO] + gamma*[C51 nO]*[1 1] -> [C51 nO]
                returns[-1] += self.gamma * self.z * non_terminal[-1]
                for i in range(len(returns)-1, 0, -1):
                    # if episode ended in last n-steps, do not add to return
                    returns[i-1] += self.gamma * returns[i] * non_terminal[i-1]
                # clamp every objective separately, depending on its own vmax and vmin: [Batch C51 nO]
                tz = torch.stack([returns[...,o].clamp(min=self.v_min[o], max=self.v_max[o]) for o in range(len(self.v_min))], dim=-1)
                b = (tz - self.v_min)/self.d_z
                l = torch.floor(b).long()
                # change b to not be exactly on border of categories
                b = torch.where(b != l, b, b + self.d_z/100)
                b = b.clamp(min=0, max=self.c-1)
                b = torch.where(b != l, b, b - self.d_z/100)
                b = b.clamp(min=0, max=self.c-1)
                u = torch.ceil(b).long()
                m = torch.zeros_like(p_ns)
                i_s = torch.arange(len(returns))
                # for each objective, for each category, get lower and upper neighbour
                for c_i in product(range(self.c), repeat=self.n_objectives):
                    b_i = [b[i_s, c_i[j], j] for j in range(self.n_objectives)] # b_i..k
                    l_i = [l[i_s, c_i[j], j] for j in range(self.n_objectives)]
                    u_i = [u[i_s, c_i[j], j] for j in range(self.n_objectives)]
                    # (b - floor(b))
                    nl_i = [(b_j-l_j) for b_j, l_j in zip(b_i, l_i)]
                    # (ceil(b) - b)
                    nu_i = [(u_j-b_j) for b_j, u_j in zip(b_i, u_i)]
                    lower_or_upper_i = [l_i, u_i]
                    lower_or_upper_p = [nu_i, nl_i]
                    current_i = (i_s,) + c_i
                    # for each combination of lower, upper neighbour, update probabilities
                    for n_i in product(range(2), repeat=self.n_objectives):
                        # tuple (Batch, neighbour[0], ..., neighbour[n])
                        neighbour_i = (i_s,) + tuple(lower_or_upper_i[j][i] for i,j in enumerate(n_i))
                        neighbour_p = [lower_or_upper_p[j][i] for i,j in enumerate(n_i)]
                        m[neighbour_i] += p_ns[current_i]*torch.stack(neighbour_p).prod(dim=0)
            # update V-probabilities using nll
            p_s = self.critic(batch.observation)
            objective_dims = tuple(range(1,len(p_s.shape)))
            critic_loss = -torch.sum(m*torch.log(p_s), dim=objective_dims).unsqueeze(-1)
            
            with torch.no_grad():
                # expand accrued from [Batch nO] to [Batch 1 .. 1 nO]
                accrued = self.accrued[:-1].view(len(returns), *(1,)*self.n_objectives, self.n_objectives).to(self.device)
                gamma = batch.gamma.view(len(returns), *(1,)*(self.n_objectives+1))
                # shift back discounted return: accrued + gamma^t*R_t
                accrued_v = accrued + gamma*self.r_z
                u_v_s = self.utility(accrued_v.view(-1, self.n_objectives)).view_as(p_s)
                # expected utility for current state [Batch C51 .. C51]*[Batch C51 .. C51] -> [Batch]
                u_v_s = torch.sum(u_v_s*p_s, dim=objective_dims)
                # get all combinations of n0,n1,... (so C51 goes to c51**nO)
                o_n = torch.meshgrid(*[torch.arange(self.c) for _ in range(self.n_objectives)])
                # [Batch C51 .. C51 nO]
                r_z = torch.stack(tuple(returns[:,o_i,i] for i, o_i in enumerate(o_n)),dim=-1)
                accrued_r = accrued + gamma*r_z
                # compute the utility for all these returns [Batch C51 .. C51]
                u_r_s = self.utility(accrued_r.view(-1, self.n_objectives)).view_as(p_s)
                # expected utility using n-step returns: [Batch]
                u_r_s = torch.sum(u_r_s*p_ns[-1].unsqueeze(0), dim=objective_dims)
                advantage = u_r_s - u_v_s

            actor_out = self.actor(batch.observation)
            # [Batch 1]
            log_prob = self.policy.log_prob(batch.action, actor_out)
            actor_loss = -log_prob*advantage.detach().unsqueeze(-1)
            # entropy
            entropy = -torch.exp(log_prob) * log_prob

            loss = actor_loss + self.v_coef*critic_loss - self.e_coef*entropy
            self.optimizer_step(loss)

            # manage accrued
            self.accrued = self.accrued[-1:]

        return {'observation': next_obs,
                'action': action,
                'reward': reward,
                'terminal': terminal}


    def evalstep(self, previous, log=None):
        with torch.no_grad():
            actor_out = self.actor(previous['observation'])
            action = self.policy(actor_out, log=log)
            next_obs, reward, terminal, info = self.env.step(action)

            p_s = self.critic(previous['observation'])
        self.logger.put('eval/p_v', p_s[0].numpy(), log.total_steps, 'histogram')
        # self.logger.put('eval/p_a', torch.exp(actor_out)[0].numpy(), log.total_steps, 'histogram')
        return {'observation': next_obs,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'env_info': info}

    def state_dict(self):
        sd = super(MOCAC, self).state_dict()
        sd.update({'actor': self.actor.state_dict(),
                   'critic': self.critic.state_dict(),
                   'policy': self.policy,
                   'memory': self.memory})
        return sd

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])
        self.critic.load_state_dict(sd['critic'])
        self.policy = sd['policy']
