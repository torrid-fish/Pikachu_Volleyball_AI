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

def mutiple_tuple(nums):
    temp = list(nums)
    product = 1 
    for x in temp:
        product *= x
    return product

class Dueling_D3QN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Dueling_D3QN, self).__init__()

        self.f1 = nn.Linear(mutiple_tuple(state_dim), 512)
        self.f2 = nn.Linear(512, 256)

        self.val_hidden = nn.Linear(256, 128)
        self.adv_hidden = nn.Linear(256, 128)

        self.val = nn.Linear(128, 1) # State value

        self.adv = nn.Linear(128, action_dim) # Advantage value

    def forward(self, x):

        if (x.ndim == 2):
            x = torch.unsqueeze(x, 0)
        
        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
        x = self.f1(x)
        x = F.relu(x)
        
        x = self.f2(x)
        x = F.relu(x)

        val_hidden = self.val_hidden(x)
        val_hidden = F.relu(val_hidden)

        adv_hidden = self.adv_hidden(x)
        adv_hidden = F.relu(adv_hidden)

        val = self.val(val_hidden)

        adv = self.adv(adv_hidden)

        adv_ave = torch.mean(adv)

        x = adv + val - adv_ave
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

def d3qn_act(isPlayer2: bool, state: list, model: Dueling_D3QN) -> int:
    # Get the model
    return model.select_action(torch(state))
