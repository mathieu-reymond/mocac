from dataclasses import dataclass, astuple
import random
import torch
import numpy as np


class BatchTransition(object):

    def __init__(self, observation, action, reward, next_observation, terminal, gamma=1., device='cpu'):
        # ensure that there is no tensors (Batch,), but always have a second dim
        if len(action.shape) == 1: action = action[:, None]
        if len(reward.shape) == 1: reward = reward[:, None]
        if len(terminal.shape) == 1: terminal = terminal[:, None]
        self.observation = observation.to(device)
        self.action = action.to(device)
        self.reward = reward.to(device)
        self.next_observation = next_observation.to(device)
        self.terminal = terminal.to(device)
        self.gamma = gamma.to(device)


class Memory(object):

    def __init__(self, size=100000, device='cpu'):

        self.size = size
        self.memory = []
        self.current = 0
        self.device = device

    def add(self, transition):
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.current] = astuple(transition)
        self.current = (self.current + 1) % self.size

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        # TODO if multisized batch, will mix different envs steps together
        # use torch.stack(i).transpose(1,0).reshape(-1, *i.shape[1:])
        batch = BatchTransition(*[torch.cat(i) for i in zip(*batch)], device=self.device)
        return batch

    def last(self, batch_size):
        assert len(self.memory) >= batch_size, 'not enough samples in memory'
        s_i, e_i = self.current - batch_size, self.current
        # if s_i < 0, part of batch is at end of list, wrap around
        if s_i < 0:
            batch = self.memory[s_i:] + self.memory[:e_i]
        else:
            batch = self.memory[s_i:e_i]
        # TODO if multisized batch, will mix different envs steps together
        # use torch.stack(i).transpose(1,0).reshape(-1, *i.shape[1:])
        batch = BatchTransition(*[torch.cat(i) for i in zip(*batch)], device=self.device)
        return batch


class EpisodeMemory(Memory):

    def __init__(self, size=100000, device='cpu'):
        super(EpisodeMemory, self).__init__(size=size)
        self.episode = 0
        self.e_start = 0
        self.device = device

    def add(self, transition, episode=0):
        # keep track of current episode, and its first transition
        if episode > self.episode:
            self.episode = episode
            self.e_start = self.current
        elif self.e_start == self.current:
            raise MemoryError('current episode is longer than max buffer size')
        super(EpisodeMemory, self).add(transition)

    def get_episode(self):
        # part of episode is at the end, wrap around
        if self.e_start > self.current:
            batch = self.memory[self.e_start:] + self.memory[:self.current]
        else:
            batch = self.memory[self.e_start:self.current]
        # TODO if multisized batch, will mix different envs steps together
        # use torch.stack(i).transpose(1,0).reshape(-1, *i.shape[1:])
        batch = BatchTransition(*[torch.cat(i) for i in zip(*batch)], device=self.device)
        return batch