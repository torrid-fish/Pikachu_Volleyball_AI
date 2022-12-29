import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import random

"""
    Memory Device (PER)
"""
class PER:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __getstate__(self):
        # Return the object's state as a dictionary
        return {
            'capacity': self.capacity,
            'tree': self.tree
        }
    
    def __setstate__(self, state):
        # Set the object's state from the dictionary
        self.capacity = state['capacity']
        self.tree = state['tree']

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error.data.item())
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        for i, e in zip(idx, error):
            p = self._get_priority(e)
            self.tree.update(i, p)
        
    def size(self):
        return self.tree.n_entries

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def __getstate__(self):
        # Return the object's state as a dictionary
        return {
            'tree': self.tree,
            'capacity': self.capacity,
            'data': self.data,
            'n_entries': self.n_entries
        }
    
    def __setstate__(self, state):
        # Set the object's state from the dictionary
        self.tree = state['tree']
        self.capacity = state['capacity']
        self.data = state['data']
        self.n_entries = state['n_entries']

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])



"""
    Network structure (D3QN)
"""
class Dueling_D3QN(nn.Module):
    def __init__(self, action_dim):
        super(Dueling_D3QN, self).__init__()

        self.f1 = nn.Linear(17, 128)
        self.f2 = nn.Linear(128, 512)
        self.f3 = nn.Linear(512, 1024)
        

        self.val_hidden = nn.Linear(1024, 512)
        self.adv_hidden = nn.Linear(1024, 512)

        self.val = nn.Linear(512, 1) # State value
        self.adv = nn.Linear(512, action_dim) # Advantage value

        torch.nn.init.normal_(self.f1.weight, 0, 0.02)
        torch.nn.init.normal_(self.f2.weight, 0, 0.02)
        torch.nn.init.normal_(self.f3.weight, 0, 0.02)
        
        torch.nn.init.normal_(self.val_hidden.weight, 0, 0.02)      
        torch.nn.init.normal_(self.val.weight, 0, 0.02)
        torch.nn.init.normal_(self.adv_hidden.weight, 0, 0.02)
        torch.nn.init.normal_(self.adv.weight, 0, 0.02)

    def forward(self, x):
        x = torch.reshape(x, (-1, 17)) 

        x = self.f1(x)
        x = F.relu(x)

        x = self.f2(x)
        x = F.relu(x)
        
        x = self.f3(x)
        x = F.relu(x)

        val_hidden = self.val_hidden(x)
        val_hidden = F.relu(val_hidden)

        adv_hidden = self.adv_hidden(x)
        adv_hidden = F.relu(adv_hidden)

        val = self.val(val_hidden)
        adv = self.adv(adv_hidden)

        adv_ave = torch.mean(adv, axis=1, keepdim=True)

        x = adv + val - adv_ave
        return x

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q)
        return action_index.item()
