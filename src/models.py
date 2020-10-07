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


class SAC(nn.Module):
    """
    This network is a SAC network from
       > Soft Actor-Critic Algorithms and Applications, Haarnoja et al 2018

    Tricks include dual Q networks, adjusting entropy, and gumbel softmax for
    discrete
    """

    def __init__(self, env, target_entropy_ratio=0.95):
        super(SAC, self).__init__()
        self.policy = Policy(env)
        self.soft_q1 = SoftQNetwork(env)
        self.soft_q2 = SoftQNetwork(env)
        self.tgt_q1 = SoftQNetwork(env).eval()
        self.tgt_q2 = SoftQNetwork(env).eval()

    def _freeze_tgt_networks(self):
        """
        Copy soft q networks into target q networks, and freeze parameters
        for training stability
        """
        q1 = zip(self.tgt_q1.parameters(), self.soft_q1.parameters())
        q2 = zip(self.tgt_q2.parameters(), self.soft_q2.parameters())

        # Copy parameters
        for target_param, param in q1:
            target_param.data.copy_(param.data)
        for target_param, param in q2:
            target_param.data.copy_(param.data)

        # Freeze gradients
        for param in self.tgt_q1.parameters():
            param.requires_grad = False
        for param in self.tgt_q2.parameters():
            param.requires_grad = False


class SoftQNetwork(nn.Module):
    """
    Given an environment with |S| state dim and |A| actions, initialize
    a FFN with 2 hidden layers, and input size |S| + |A|.  Output a single
    Q value
    """

    def __init__(self, env, num_hidden=128, dropout=0.0):
        super(SoftQNetwork, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space + self.action_space, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_hidden)
        self.l3 = nn.Linear(num_hidden, 1)

        self.ffn = nn.Sequential(
            self.l1,
            nn.Dropout(p=dropout),
            nn.ReLU(),
            self.l2,
            nn.Dropout(p=dropout),
            nn.ReLU(),
            self.l3,
        )
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights with xaiver uniform, and
        fill bias with 1 over n
        """
        over_n = 1 / (self.state_space + self.action_space)
        nn.init.xavier_uniform_(self.l1.weight)
        self.l1.bias.data.fill_(over_n)
        nn.init.xavier_uniform_(self.l2.weight)
        self.l2.bias.data.fill_(over_n)
        nn.init.xavier_uniform_(self.l3.weight)
        self.l3.bias.data.fill_(over_n)

    def forward(self, state, action):
        """
        Given the state and action, produce a Q value
        """
        q_in = torch.cat([state, action], 1)
        return self.ffn(q_in).view(-1)

class Actor(nn.Module):
    
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
