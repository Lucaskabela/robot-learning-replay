"""
replay.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines the various experience replay techniques used
        in this research project
"""
import numpy as np


class ReplayBuffer:
    """
    Implements a simple circular buffer as a replay buffer.
    Experiences are sampled randomly.
    """

    def __init__(self, buf_size, in_shape, action_d):

        self.buf_size = buf_size
        self.num_added = 0

        self.states = np.zeros((self.buf_size, in_shape))
        self.next_states = np.zeros((self.buf_size, in_shape))
        self.actions = np.zeros((self.buf_size, action_d))
        self.rewards = np.zeros(self.buf_size)
        self.dones = np.zeros(self.buf_size, dtype=np.bool)

    def store_transition(self, state, action, state_p, reward, done):
        idx = self.num_added % self.buf_size

        self.states[idx], self.action[idx] = state, action
        self.next_states[idx] = state_p
        self.rewards[idx], self.dones[idx] = reward, done

        self.num_added += 1

    def sample_transitions(self, b_size):
        upper_bound = min(self.buf_size, self.num_added)
        batch_idxes = np.random.choice(upper_bound, b_size)

        states = self.states[batch_idxes]
        actions = self.actions[batch_idxes]
        state_ps = self.next_states[batch_idxes]
        rewards = self.rewards[batch_idxes]
        dones = self.dones[batch_idxes]

        return [states, actions, state_ps, rewards, dones]

    def __len__(self):
        return min(self.buf_size, self.num_added)
