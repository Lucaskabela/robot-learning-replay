"""
utils.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines the various datastructures and utility functions
        used in this project
"""
import gym
import numpy as np
import torch
import torch.nn.functional as F


def get_action_dim(env):
    if type(env.action_space) is gym.spaces.Box:
        return env.action_space.shape[0]
    else:
        return env.action_space.n


def guard_q_actions(actions, dim):
    """Guard to convert actions to one-hot for input to Q-network"""
    actions = F.one_hot(actions.long(), dim).float()
    return actions


def get_one_hot_np(tgt, dim):
    """Guard to convert actions to one-hot for input to buffers"""
    res = np.eye(dim)[np.array(tgt).reshape(-1)]
    return res.reshape(list(tgt.shape) + [dim])


# Taken from https://github.com/vaishak2future/sac/blob/master/sac.ipynb
class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


# https://stackoverflow.com/questions/56226133/soft-actor-critic-with-discrete-action-space
# ... for discrete action, GumbelSoftmax distribution
class GumbelSoftmax(torch.distributions.RelaxedOneHotCategorical):
    """
    A differentiable Categorical distribution using reparam trick with Gumbel
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution
    since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete
        Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    """

    def sample(self, sample_shape=torch.Size()):
        """
        Gumbel-softmax sampling. rsample is from RelaxedOneHotCategorical
        """
        device = self.logits.device
        u = torch.empty(
            self.logits.size(), device=device, dtype=self.logits.dtype
        ).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def log_prob(self, value):
        """value is one-hot or relaxed"""
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return -torch.sum(-value * F.log_softmax(self.logits, -1), -1)
