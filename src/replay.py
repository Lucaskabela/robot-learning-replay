"""
replay.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines the various experience replay techniques used
        in this research project
"""
import numpy as np
import random
from utils import SumSegmentTree, MinSegmentTree 

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

    def store(self, state, action, state_p, reward, done):
        idx = self.num_added % self.buf_size

        self.states[idx], self.actions[idx] = state, action
        self.next_states[idx] = state_p
        self.rewards[idx], self.dones[idx] = reward, done

        self.num_added += 1

    def sample(self, b_size=256):
        upper_bound = min(self.buf_size, self.num_added)
        batch_idxes = np.random.choice(upper_bound, b_size)

        states = self.states[batch_idxes]
        actions = self.actions[batch_idxes]
        state_ps = self.next_states[batch_idxes]
        rewards = self.rewards[batch_idxes]
        dones = self.dones[batch_idxes]

        return [states, actions, rewards, state_ps, dones]

    def __len__(self):
        return min(self.buf_size, self.num_added)

class HindsightReplay:
    # Adapted from openAi baselines again
    # env_params requires goal environment
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func

        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
    
    # store the episode
    def store(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]

        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self, b_size=256):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def __len__(self):
        return self.current_size


class PrioritizedReplay(ReplayBuffer):
    # From https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    def __init__(self, buf_size, in_shape, action_d, alpha=.6):
        super(PrioritizedReplay, self).__init__(buf_size, in_shape, action_d)
        self.alpha = alpha
        it_capacity = 1
        while it_capacity < buf_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def store(self, state, action, state_p, reward, done):
        idx = self.num_added % self.buf_size

        super().store(state, action, state_p, reward, done)
        self._it_sum[idx] = self._max_priority ** self.alpha
        self._it_min[idx] = self._max_priority ** self.alpha
       

    def _sample_prop(self, b_size):
        res = []
        upper = min(self.buf_size, self.num_added) - 1
        p_total = self._it_sum.sum(0, upper)
        every_range_len = p_total / b_size
        for i in range(b_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, b_size=256, beta=.4):
        batch_idxes = self._sample_prop(b_size)
        upper = min(self.buf_size, self.num_added) - 1
        weights = np.zeros(b_size)
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * upper) ** (-beta)

        i = 0
        for idx in batch_idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * upper) ** (-beta)
            weights[i] = weight / max_weight
            i += 1
        states = self.states[batch_idxes]
        actions = self.actions[batch_idxes]
        state_ps = self.next_states[batch_idxes]
        rewards = self.rewards[batch_idxes]
        dones = self.dones[batch_idxes]

        return states, actions, rewards, state_ps, dones, weights, np.array(batch_idxes)

    def update_priorities(self, idxes, priorities):
        assert len(priorities.shape) == 1
        assert idxes.shape == priorities.shape 

        idxes = idxes.long().tolist()
        priorities = priorities.tolist()
        for idx, prio in zip(idxes, priorities):
            self._it_sum[idx] = prio ** self.alpha
            self._it_min[idx] = prio ** self.alpha

            self._max_priority = max(self._max_priority, prio)


# TODO: Change this to make use of hindsight expereince replay.  Should
# be simple!
class PHEReplay(HindsightReplay):

    def __init__(self, buf_size, in_shape, action_d, alpha=.6):

        self.buf_size = buf_size
        self.num_added = 0

        self.states = np.zeros((self.buf_size, in_shape))
        self.next_states = np.zeros((self.buf_size, in_shape))
        self.actions = np.zeros((self.buf_size, action_d))
        self.rewards = np.zeros(self.buf_size)
        self.dones = np.zeros(self.buf_size, dtype=np.bool)

    def store(self, state, action, state_p, reward, done):
        idx = self.num_added % self.buf_size

        self.states[idx], self.actions[idx] = state, action
        self.next_states[idx] = state_p
        self.rewards[idx], self.dones[idx] = reward, done

        self.num_added += 1

    def sample(self, b_size=256):
        upper_bound = min(self.buf_size, self.num_added)
        batch_idxes = np.random.choice(upper_bound, b_size)

        states = self.states[batch_idxes]
        actions = self.actions[batch_idxes]
        state_ps = self.next_states[batch_idxes]
        rewards = self.rewards[batch_idxes]
        dones = self.dones[batch_idxes]

        return [states, actions, rewards, state_ps, dones]

    def __len__(self):
        return min(self.buf_size, self.num_added)