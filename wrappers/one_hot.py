import gym
import numpy as np


class OneHotEnv(gym.ObservationWrapper):
    """
    convert a state from index value to one-hot encoding
    """
    def __init__(self, env, n_states):
        super(OneHotEnv, self).__init__(env)
        self.n_states = n_states


    def observation(self, state):
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[state] = 1

        return one_hot