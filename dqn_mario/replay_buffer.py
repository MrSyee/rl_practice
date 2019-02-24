"""Replay buffer for DQN"""

import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.idx = 0

    # buffer 길이 체크
    def __len__(self):
        return len(self.buffer)

    # buffer에 sample 추가
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if len(self.buffer) == self.capacity:
            self.buffer[self.idx] = data
            self.idx = (self.idx + 1) % self.capacity
        else:
            self.buffer.append(data)

    # buffer에서 batch_size만큼 뽑기
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in idxs:
            s, a, r, n_s, d = self.buffer[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(n_s, copy=False))
            dones.append(np.array(float(d), copy=False))

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        return states, actions, rewards, next_states, dones
