import numpy as np
import gym


class TerminalEnv(gym.Wrapper):
    def __init__(self, env, utility, gamma=1.):
        super(TerminalEnv, self).__init__(env)
        self.utility = utility
        self.gamma = gamma

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self._accrued = 0.
        return state

    def step(self, ac):
        state, r, d, i = self.env.step(ac)
        self._accrued += r
        r = np.array([0.])

        if d:
            r = self.utility(np.array([self._accrued]))[0]
            r = np.array([r])

        return state, r, d, i
