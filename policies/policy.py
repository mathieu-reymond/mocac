import torch
import numpy as np


class Policy(object):

    def __call__(self, args, log=None):
        raise NotImplementedError()

    def log_prob(self, samples, args):
        raise NotImplementedError()


class OrnsteinUhlenbeckActionNoise(object):

    def __init__(self, n_actions, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.theta = theta
        self.mu = torch.zeros(n_actions)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(0, 1, self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros(*self.mu.shape)


class NormalNoise(object):

    def __init__(self, action_dim, sigma=0.2):
        super(NormalNoise, self).__init__()
        self.sigma = sigma
        self.action_dim = action_dim

    def __call__(self):
        return self.sigma*torch.normal(0, 1, (self.action_dim,))

    def reset(self):
        pass


class Normal(Policy):

    def __init__(self, action_high, action_dim, noise=None, **noise_kwargs):
        noises = {
            'ornstein_uhlenbeck': OrnsteinUhlenbeckActionNoise(action_dim, **noise_kwargs),
            'normal': NormalNoise(action_dim, **noise_kwargs)
        }
        if noise is not None:
            noise = noises[noise]
        self.noise = noise
        self.action_high = action_high

    @classmethod
    def _get_mu_std(cls, log_probs):
        # if tuple, deviation was provided, else, assume deterministic
        if type(log_probs) == tuple:
            mu, log_std = log_probs
            std = torch.exp(log_std)
        else:
            mu, std = log_probs, torch.zeros(log_probs.shape)
            log_std = None
        return mu, std, log_std

    def __call__(self, log_probs, log=None):
        if self.noise is not None and log.episode_step == 0:
            self.noise.reset()
        mu, std, _ = Normal._get_mu_std(log_probs)
        # TODO user rsample from torch.distributions.Normal
        actions = torch.normal(mu, std)
        if self.noise is not None:
            actions += self.noise()
        actions.clamp_(-self.action_high, self.action_high)
        return actions

    def log_prob(self, samples, log_probs):
        mu, std, log_std = Normal._get_mu_std(log_probs)
        var = std ** 2
        return -((samples - mu) ** 2) / (2 * var) - log_std - np.log(np.sqrt(2 * np.float32(np.pi)))


class Categorical(Policy):

    def __call__(self, log_probs, log=None):
        probs = torch.exp(log_probs)
        actions = torch.multinomial(probs, 1)
        return actions

    def log_prob(self, samples, log_probs):
        return log_probs.gather(1, samples)


class MultiCategorical(Policy):

    def __call__(self, log_probs, log=None):
        probs = torch.exp(log_probs)
        actions = torch.cat([torch.multinomial(probs[...,i], 1) for i in range(probs.shape[-1])], 1)
        return actions

    def log_prob(self, samples, log_probs):
        return log_probs.gather(1, samples.unsqueeze(1)).squeeze(1)


class EGreedy(Policy):

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, values, log=None):
        actions = torch.multinomial(self.probs(values), 1)
        return actions

    def probs(self, values):
        # get greedy value
        m, _ = torch.max(values, dim=1, keepdims=True)
        # corresponding greedy actions
        p = values == m
        # possible that multiple actions are greedy, scale so that probs add to 1-eps
        p = (1.-self.epsilon)*p/torch.sum(p, dim=1, keepdims=True)
        # add epsilon probability to take random action
        p = p + self.epsilon/p.shape[1]
        return p


class AnnealingEGreedy(EGreedy):

    def __init__(self, n_steps=1, max_epsilon=1., min_epsilon=0.1, decay='linear'):
        self.n_steps = n_steps
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self._range = self.max_epsilon - self.min_epsilon

        def linear(step):
            e = (self.n_steps - step)/self.n_steps*self._range + self.min_epsilon
            return e
        def exp(step):
            tau = self.n_steps/np.e
            e = (np.exp(-step/tau) - np.exp(-np.e)*(step/n_steps))*self._range + self.min_epsilon
            return e
        decays = {'linear': linear,
                  'exp': exp}
        self.decay = decays[decay]
        
        super(AnnealingEGreedy, self).__init__(epsilon=1.)

    def __call__(self, values, log=None):
        e = self.decay(log.total_steps)
        self.epsilon = np.maximum(self.min_epsilon, e)
        return super(AnnealingEGreedy, self).__call__(values, log=log)