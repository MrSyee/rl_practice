import random
import numpy as np
from collections import namedtuple
import torch

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class Memory_With_TDError(Memory):
    def __init__(self, capacity):
        self.memory = []
        self.td_memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, next_state, action, reward, mask, td_error):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, next_state, action, reward, mask))
            self.td_memory.append(td_error)
        self.memory[self.position] = Transition(state, next_state, action, reward, mask)
        self.td_memory[self.position] = td_error

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        td_sum = sum(self.td_memory)
        p = [td_error / td_sum for td_error in self.td_memory]
        indexes = np.random.choice(np.arange(len(self.memory)), batch_size, p=p)
        transitions = [self.memory[idx] for idx in indexes]
        # transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)

    def get_td_error(self, state, next_state, action, reward, mask, gamma, net, target_net):
        state = torch.stack([state])
        next_state = torch.stack([next_state])
        action = torch.Tensor([action]).long()
        reward = torch.Tensor([reward])
        mask = torch.Tensor([mask])

        pred = net(state).squeeze(1)
        next_pred = target_net(next_state).squeeze(1)

        one_hot_action = torch.zeros(1, pred.size(-1))
        one_hot_action.scatter_(1, action.unsqueeze(1), 1)
        pred = torch.sum(pred.mul(one_hot_action), dim=1)

        target = reward + mask * gamma * next_pred.max(1)[0]

        td_error = pred - target

        return td_error
