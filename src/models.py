"""
models.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines Neural Network Architecture and other models
        which will be evaluated in this expirement
"""
import torch
import torch.nn as nn

# from os import path
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()
        state_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        num_hidden = 128

        self.l1 = nn.Linear(state_space, num_hidden, bias=False)
        self.l2 = nn.Linear(num_hidden, action_space, bias=False)

        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.reset()

    def reset(self):
        # Episode policy and reward history
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1, nn.Dropout(p=0.5), nn.ReLU(), self.l2, nn.Softmax(dim=-1)
        )
        return model(x)

    def device(self):
        if next(self.parameters()).is_cuda:
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def predict(self, state):
        # Select an action (0 or 1) by running policy model
        # and choosing based on the probabilities in state
        device = self.device()
        state = torch.from_numpy(state).type(torch.FloatTensor).to(device)
        action_probs = self(state)
        distribution = Categorical(action_probs)
        action = distribution.sample()

        # Add log probability of our chosen action to our history
        self.saved_log_probs.append(distribution.log_prob(action))

        return action


class Value(nn.Module):
    def __init__(self, env):
        super(Value, self).__init__()
        state_space = env.observation_space.shape[0]
        num_hidden = 128

        self.l1 = nn.Linear(state_space, num_hidden, bias=False)
        self.l2 = nn.Linear(num_hidden, 1, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.5),
            nn.ReLU(),
            self.l2,
        )
        return model(x)


def save_model(model):
    # if isinstance(model, Planner):
    #     return save(
    #       model.state_dict(), path.join(path.dirname(path.abspath(__file__)),
    #       'planner.th')
    #     )
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    r = None
    # if isinstance(model, Planner):
    #     r = Planner()
    #     r.load_state_dict(load(
    #         path.join(path.dirname(path.abspath(__file__)), 'planner.th'),
    #         map_location=model.device)
    #     )
    return r
