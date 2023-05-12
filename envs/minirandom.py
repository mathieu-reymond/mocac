from gym.core import Env
from gym.spaces import MultiDiscrete, Discrete, Box
import numpy as np
from gym.envs.registration import register


class MiniRandom(Env):

    def __init__(self):
        super(MiniRandom, self).__init__()

        transitions = np.arange(5)+1
        transitions = transitions.repeat(3).reshape(-1, 3)
        transitions[1,:] = np.arange(3)+2
        self.transitions = transitions

        self.rewards = {
            1: [(0, 0), (0, 5), (5, 0)],
            2: [(5, 0)],
            3: [(0, 5)],
            4: [(2, 2)]
        }
 
        self.observation_space = Discrete(5)
        self.action_space = Discrete(3)
        self.reward_space = Box(np.zeros(2), np.ones(2)*5)

    def reset(self):
        self.state = np.array([0])
        return self.state

    def step(self, action):
        s = self.state[0]
        ns = self.transitions[s,action]
        self.state = np.array([ns])
        if ns not in self.rewards:
            r = np.zeros(2)
        else:
            choices = self.rewards[ns]
            r = choices[np.random.randint(len(choices))]
            r = np.array(r)
        done = ns > 1
        return self.state, r, done, {}
        

register(
    id='MiniRandom-v0',
    entry_point='envs.minirandom:MiniRandom',
    reward_threshold=0.0,
)