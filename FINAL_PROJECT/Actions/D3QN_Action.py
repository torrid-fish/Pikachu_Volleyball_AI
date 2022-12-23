import random
from itertools import count
from tensorboardX import SummaryWriter
import gym
from collections import deque
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gym_pikachu_volleyball

if torch.cuda.is_available():
    device = "cuda:0"
    print("GPU is used.")
else:
    device = "cpu"
    print("CPU is used.")

class Conv2d(nn.Conv2d):
    print("weight!")

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Dueling_D3QN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Dueling_D3QN, self).__init__()
        
        # state_dim = torch.tensor(state_dim).to(device)
        # state_dim = torch.unsqueeze(state_dim, 0)

        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4) 
        self.g1 = nn.GroupNorm(4, 32)
        self.m1 = nn.AvgPool2d(3, stride=2)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) 
        self.g2 = nn.GroupNorm(4, 64)
        self.m2 = nn.MaxPool2d(3, stride=1)
        self.conv3 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) 
        self.g3 = nn.GroupNorm(4, 64)

        self.f1 = nn.Linear(17472, 1024)

        self.val_hidden = nn.Linear(1024, 256)
        self.adv_hidden = nn.Linear(1024, 256)

        self.val = nn.Linear(256, 1) # State value

        self.adv = nn.Linear(256, action_dim) # Advantage value

        torch.nn.init.normal_(self.conv1.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.02)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.02)
        torch.nn.init.normal_(self.val_hidden.weight, 0, 0.02)
        torch.nn.init.normal_(self.val.weight, 0, 0.02)
        torch.nn.init.normal_(self.adv_hidden.weight, 0, 0.02)
        torch.nn.init.normal_(self.adv.weight, 0, 0.02)

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, 304, 432)) 

        x = self.conv1(x)
        x = self.g1(x)
        x = F.relu(x)

        x = self.m1(x)
        
        x = self.conv2(x)
        x = self.g2(x)
        x = F.relu(x)

        x = self.m2(x)

        x = self.conv3(x)
        x = self.g3(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)

        x = self.f1(x)
        x = F.relu(x)

        val_hidden = self.val_hidden(x)
        val_hidden = F.relu(val_hidden)

        adv_hidden = self.adv_hidden(x)
        adv_hidden = F.relu(adv_hidden)

        val = self.val(val_hidden)
        adv = self.adv(adv_hidden)
        
        adv_ave = torch.mean(adv, axis=1, keepdims=True)
        x = adv + val - adv_ave
        #print(x.shape, adv.shape, val.shape, adv_ave.shape)
        return x

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q)
        return action_index.item()

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        return observations, actions, rewards, next_observations, dones

    def size(self):
        return len(self.memory)


class PER:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error.cpu().detach().numpy()) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
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
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
    def size(self):
        return self.tree.n_entries

def d3qn_act(isPlayer2: bool, state: np.ndarray, model: Dueling_D3QN) -> int:
    if not isPlayer2:
        np.flip(state, axis = 1)
    state = torch.as_tensor(state, dtype=torch.float32).to(device)
    model = model.to(device)
    # Get the model
    return model.select_action(state)

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

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
