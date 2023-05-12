import numpy as np
import gym
from gym.spaces import Box


class SplitEnv(gym.Env):
    """
        Make a tunnel-like env, with 2 paths:
        start -> (action 0) -> tunnel -> (action 0) -> terminal with reward [0, 10]
                                      -> (action 1) -> terminal with reward [10, 0]
              -> (action 1) -> tunnel -> terminal with reward [3, 3]
        The utility is r0*r1, so [3, 3] should be best, but simply using the Q-values,
        the expected return (per objective) of the other tunnel will be better [5, 5],
        which leads to the wrong policy
    """

    def __init__(self, tunnel=10):
        super(SplitEnv, self).__init__()
        transitions = np.arange(6+2*tunnel) + 1
        transitions = transitions.repeat(2).reshape(-1, 2)
        # s_0, a_1 goes to other tunnel
        transitions[0, 1] = 4 + tunnel
        # end of tunnel 1, a_1 goes to other terminal state
        transitions[tunnel+1, 1] += 1
        rewards = np.zeros_like(transitions)
        rewards[tunnel+2] = np.array([0, 10])
        rewards[tunnel+3] = np.array([10, 0])
        rewards[-1] = np.array([3, 3])
        self.transitions = transitions
        self.rewards = rewards

        self.reward_space = Box(np.zeros(2), np.ones(2)*10)

    def reset(self):
        self.state = np.array([0])
        return self.state

    def step(self, action):
        s = self.state[0]
        ns = self.transitions[s,action]
        self.state = np.array([ns])
        rew = self.rewards[ns]
        done = np.any(rew != 0)
        return self.state, rew, done, {}