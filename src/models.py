"""
models.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines Neural Network Architecture and other models
        which will be evaluated in this expirement
"""
import numpy as np
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from os import path
from torch.distributions import Normal
from utils import GumbelSoftmax, guard_q_actions, get_action_dim


class SAC(nn.Module):
    """
    This network is a SAC network from
       > Soft Actor-Critic Algorithms and Applications, Haarnoja et al 2018

    Tricks include dual Q networks, adjusting entropy, and gumbel softmax for
    discrete
    """

    def __init__(self, env, alpha=.2, gamma=0.99, tau=0.005, at=True, dis=False):
        super(SAC, self).__init__()
        if dis:
            self.actor = DiscreteActor(env)
        else:
            self.actor = Actor(env)
        self.discrete = dis
        self.soft_q1 = SoftQNetwork(env, name="q1")
        self.soft_q2 = SoftQNetwork(env, name="q2")
        self.tgt_q1 = SoftQNetwork(env).eval()
        self.tgt_q2 = SoftQNetwork(env).eval()
        self.gamma = gamma
        self.tau = tau
        
        self.alpha = alpha
        self.adjust_alpha = at
        self.log_alpha = torch.zeros(1, requires_grad=True)
        if at:
            if self.discrete:
                # Need positive entropy < log(action_size) if discrete
                tgt = -np.log((1.0 / env.action_space.n)) * 0.98
                self.target_entropy = tgt.item()
            else:
                tgt = torch.Tensor(env.action_space.shape)
                self.target_entropy = -torch.prod(tgt).item()
            self.alpha = self.log_alpha.detach().exp()

    def to(self, device):
        self.log_alpha = self.log_alpha.to(device)
        return super(Actor, self).to(device)

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            return self.actor.get_action(state)

    def init_opt(self, opt="Adam", lr=3e-4):
        self.q1_opt = optim.Adam(self.soft_q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.soft_q2.parameters(), lr=lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.entropy_opt = optim.Adam([self.log_alpha], lr=lr)

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

    def soft_copy(self):
        q1_params = zip(self.tgt_q1.parameters(), self.soft_q1.parameters())
        q2_params = zip(self.tgt_q2.parameters(), self.soft_q2.parameters())
        for target_param, param in q1_params:
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in q2_params:
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def calc_critic_loss(self, states, actions, rewards, next_states, done):
        with torch.no_grad():
            advantage = self.actor.evaluate(next_states)
            next_actions, next_probs, _, _, _ = advantage
            if self.discrete:
                act_size = self.tgt_q1.action_space
                next_actions = guard_q_actions(next_actions, act_size)
            else:
                next_probs = next_probs.squeeze(1)
            next_q1 = self.tgt_q1(next_states, next_actions)
            next_q2 = self.tgt_q2(next_states, next_actions)
            min_q_next = torch.min(next_q1, next_q2) - self.alpha * next_probs
            target_q_value = rewards + (1 - done) * self.gamma * min_q_next

        p_q1 = self.soft_q1(states, actions)
        p_q2 = self.soft_q2(states, actions)
        q_value_loss1 = F.mse_loss(p_q1, target_q_value.detach())
        q_value_loss2 = F.mse_loss(p_q2, target_q_value.detach())
        return q_value_loss1, q_value_loss2

    def update_critics(self, q1_loss, q2_loss, clip=None):
        self.q1_opt.zero_grad()
        q1_loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(self.soft_q1.parameters(), clip)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(self.soft_q2.parameters(), clip)
        self.q2_opt.step()

    def calc_actor_loss(self, states):
        # Train actor network
        res = self.actor.evaluate(states, reparam=True)
        actions, log_probs, _, _, _ = res
        q1 = self.soft_q1(states, actions)
        q2 = self.soft_q1(states, actions)
        min_q = torch.min(q1, q2)
        policy_loss = (self.alpha * log_probs - min_q).mean()
        return policy_loss, log_probs

    def update_actor(self, actor_loss, clip=None):
        self.actor_opt.zero_grad()
        actor_loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), clip)
        self.actor_opt.step()

    def calc_entropy_tuning_loss(self, log_probs):
        """
        Calculates the loss for the entropy temperature parameter.
        log_probs come from the return value of calculate_actor_loss
        """
        alpha_loss = 0
        if self.adjust_alpha:
            with torch.no_grad():
                inner_prod = (log_probs + self.target_entropy).detach()
            alpha_loss = -(self.log_alpha * inner_prod).mean()
        return alpha_loss

    def update_entropy(self, alpha_loss):
        if self.adjust_alpha:
            self.entropy_opt.zero_grad()
            alpha_loss.backward()
            self.entropy_opt.step()
            self.alpha = self.log_alpha.detach().exp()


    def save(self):
        self.soft_q1.save_model()
        self.soft_q2.save_model()
        self.actor.save_model()

    def load(self):
        self.soft_q1.load_model()
        self.soft_q2.load_model()
        self.actor.load_model()
        self._freeze_tgt_networks()


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def device(self):
        return next(self.parameters()).device

    def save_model(self):
        dir = "ckpt"
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        file_name = "{}.th".format(self.name)
        fn = path.join(dir, file_name)
        return torch.save(
            self.state_dict(),
            path.join(path.dirname(path.abspath(__file__)), fn),
        )

    def load_model(self):
        file_name = "{}.th".format(self.name)
        fn = path.join("ckpt", file_name)
        if not path.exists(fn):
            raise Exception("Missing saved model")       
        self.load_state_dict(
            torch.load(
                path.join(path.dirname(path.abspath(__file__)), fn),
                map_location=self.device(),
            )
        )


class SoftQNetwork(BaseNetwork):
    """
    Given an environment with |S| state dim and |A| actions, initialize
    a FFN with 2 hidden layers, and input size |S| + |A|.  Output a single
    Q value
    """

    def __init__(self, env, hidden=[256, 256], dropout=0.0, name="q1"):
        super(SoftQNetwork, self).__init__()
        self.name = name
        self.state_space = env.observation_space.shape[0]
        self.action_space = get_action_dim(env)
        self.hidden = hidden
        self.l1 = nn.Linear(self.state_space + self.action_space, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3 = nn.Linear(hidden[1], 1)
        self.init_weights()

        self.ffn = nn.Sequential(
            self.l1,
            nn.Dropout(p=dropout),
            nn.ReLU(),
            self.l2,
            nn.Dropout(p=dropout),
            nn.ReLU(),
            self.l3,
        )

    def init_weights(self, init_w=3e-3):
        """
        Initialize weights with uniform
        """
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """
        Given the state and action, produce a Q value
        """
        q_in = torch.cat([state, action], 1)
        return self.ffn(q_in).view(-1)


class Actor(BaseNetwork):
    def __init__(
        self,
        env,
        hidden=[256, 256],
        dropout=0.0,
        log_std_min=-20,
        log_std_max=2,
        name="sac",
    ):
        super(Actor, self).__init__()
        self.name = name
        self.state_space = env.observation_space.shape[0]
        self.action_space = get_action_dim(env)
        if env.action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (env.action_space.high - env.action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (env.action_space.high + env.action_space.low) / 2.)

        self.hidden = hidden
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(self.state_space, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.ffn = nn.Sequential(
            self.l1,
            nn.Dropout(p=dropout),
            nn.ReLU(),
            self.l2,
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )

        self.mean_linear = nn.Linear(hidden[1], self.action_space)
        self.log_std_linear = nn.Linear(hidden[1], self.action_space)
        self.init_weights()

    def init_weights(self, init_w=3e-3):
        """
        Initialize weights with uniform
        """
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = self.ffn(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(
            log_std,
            min=self.log_std_min,
            max=self.log_std_max,
        )

        return mean, log_std

    def evaluate(self, state, reparam=True, epsilon=1e-6):
        """
        Evaluate a state, returning action, log probs,
        mean, log_std, and z, the sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        if reparam:
            z = normal.rsample()
        else:
            z = mean
        x = torch.tanh(z)
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(self.action_scale * (1 - x.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        action = x * self.action_scale + self.action_bias

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        """
        Returns an action given a state
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.action_scale + self.action_bias

        return action.detach().cpu().numpy()[0]

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)

class DiscreteActor(BaseNetwork):
    def __init__(self, env, hidden=[256, 256], dropout=0.0, name="sac_d"):
        super(DiscreteActor, self).__init__()
        self.name = name
        self.state_space = env.observation_space.shape[0]
        # always discrete, so never box
        self.action_space = env.action_space.n
        self.hidden = hidden

        self.l1 = nn.Linear(self.state_space, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3 = nn.Linear(hidden[1], self.action_space)
        self.init_weights()
        self.ffn = nn.Sequential(
            self.l1,
            nn.Dropout(p=dropout),
            nn.ReLU(),
            self.l2,
            nn.Dropout(p=dropout),
            nn.ReLU(),
            self.l3,
            nn.Softmax(dim=-1),
        )

    def init_weights(self, init_w=3e-3):
        """
        Initialize weights with uniform
        """
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        return self.ffn(state)

    def evaluate(self, state, epsilon=1e-6, reparam=False):
        """
        Evaluate a state, returning action, log probs,
        mean, log_std, and z, the sampled action
        """

        action_probs = self.forward(state)
        action_pd = GumbelSoftmax(probs=action_probs, temperature=0.9)
        actions = action_pd.rsample() if reparam else action_pd.sample()
        log_probs = action_pd.log_prob(actions)
        return actions, log_probs, None, None, None

    def get_action(self, state):
        """
        Returns an action given a state
        """
        action_probs = self.forward(state)
        action = torch.distributions.Categorical(probs=action_probs).sample()
        action = action.detach().cpu().numpy()
        return action
