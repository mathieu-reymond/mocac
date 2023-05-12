from gym.envs.toy_text import discrete
import numpy as np
from gym.spaces import Box


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class DeepSeaTreasureEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width=11):

        self.shape = (width+1, width)
        self.start_state_index = 0

        nS = np.prod(self.shape)
        nA = 4

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (0, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(DeepSeaTreasureEnv, self).__init__(nS, nA, P, isd)
        t = self._treasures()
        min_dist = np.sum(list(t.keys()), axis=1).min()
        max_treasure = max(list(t.values()))
        self.reward_space = Box(np.array([0., -np.inf]), np.array([max_treasure, min_dist]))

    def _treasures(self):

        # return {(1, 0): 10,
        #         (2, 1): 20,
        #         (3, 2): 30,
        #         (4, 3): 40,
        #         (4, 4): 50,
        #         (4, 5): 60,
        #         (7, 6): 70,
        #         (7, 7): 80,
        #         (9, 8): 90,
        #         (10, 9): 100}
        # return {(1, 0): 1,
        #         (2, 1): 2,
        #         (3, 2): 3,
        #         (4, 3): 5,
        #         (4, 4): 8,
        #         (4, 5): 16,
        #         (7, 6): 24,
        #         (7, 7): 50,
        #         (9, 8): 74,
        #         (10, 9): 124}
        return {(2, 0): 18,
                (2, 1): 26,
                (2, 2): 31,
                (4, 3): 44,
                (4, 4): 48.2,
                (5, 5): 56,
                (8, 6): 72,
                (8, 7): 76.3,
                (10, 8): 90,
                (11, 9): 100}

    def _unreachable_positions(self):
        u = []
        treasures = self._treasures()
        for p in treasures.keys():
            for i in range(p[0]+1, self.shape[0]):
                u.append((i, p[1]))
        return u

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):

        unreachable = self._unreachable_positions()
        treasures = self._treasures()
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_position = tuple(new_position)
        if new_position in unreachable:
            new_position = tuple(current)
        new_state = np.ravel_multi_index(new_position, self.shape)

        if new_position in treasures:
            reward = [treasures[new_position], -1]
            done = True
        else:
            reward = [0, -1]
            done = False
        return [(1., new_state, np.array(reward), done)]