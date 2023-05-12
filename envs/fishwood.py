import gym
import numpy as np

from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register

import random


class FishWood(gym.Env):
    FISH = 0
    WOOD = 1

    def __init__(self, fishproba, woodproba):
        self._fishproba = fishproba
        self._woodproba = woodproba

        self.action_space = spaces.Discrete(2)              # 2 actions, go fish and go wood
        self.observation_space = spaces.Discrete(2)         # 2 states, fishing and in the woods
        # 2 objectives, amount of fish and amount of wood
        self.reward_space = spaces.Box(np.zeros(2), np.ones(2)*200)

        self.seed()
        self.reset()

    def reset(self):
        """ Reset the environment and return the initial state number
        """
        # Pick an initial state at random
        self._state = self.WOOD
        self._timestep = 0

        return self._state

    def step(self, action):
        # Obtain a resource from the current state
        rewards = np.zeros((2,))

        if self._state == self.WOOD and random.random() < self._woodproba:
            rewards[self.WOOD] = 1.0
        elif self._state == self.FISH and random.random() < self._fishproba:
            rewards[self.FISH] = 1.0

        # Execute the action
        self._state = action
        self._timestep += 1
        
        return self._state, rewards, self._timestep == 13, {}


register(
    id='FishWood-v0',
    entry_point='envs.fishwood:FishWood',
    reward_threshold=0.0,
    kwargs={'fishproba': 0.25, 'woodproba': 0.65}
)
