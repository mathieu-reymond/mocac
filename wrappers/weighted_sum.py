import gym
import numpy as np


class WeightedSum(gym.RewardWrapper):

    def __init__(self, env, weights):
        super(WeightedSum, self).__init__(env)
        self.weights = weights

    def reward(self, rew):
        return self.weights.dot(rew).reshape(1,)