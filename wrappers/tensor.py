import gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):

    def __init__(self, env):
        super(TensorWrapper, self).__init__(env)

    def reset(self):
        obs = super(TensorWrapper, self).reset()
        # convert to torch.tensor, needs to be float32 for nn
        obs = torch.from_numpy(obs.astype(np.float32))
        # make 1-sized batch
        obs = obs.unsqueeze(0)
        return obs

    def step(self, action):
        # squeeze to remove 1-sized batch
        action = action.detach().squeeze(0).cpu().numpy()
        # if discrete action, take it from 1-sized array
        if action.dtype != np.float32 and len(action) == 1: action = action[0]
        obs, reward, terminal, info = super(TensorWrapper, self).step(action)
        # convert to torch.tensor, needs to be float32 for nn
        obs = torch.from_numpy(obs.astype(np.float32))
        # # add trailing dimension to have [Batch, 1] tensors
        # reward = torch.from_numpy(np.array(reward, dtype=np.float32)).unsqueeze(-1)
        # assume multiple objectives
        reward = torch.from_numpy(np.array(reward, dtype=np.float32))
        terminal = torch.from_numpy(np.array(terminal)).unsqueeze(-1)
        # make 1-sized batch
        obs, reward, terminal = map(lambda x: x.unsqueeze(0), [obs, reward, terminal])
        return obs, reward, terminal, info