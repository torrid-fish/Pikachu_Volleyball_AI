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
        
        # state_dim = torch.tensor(state_dim).to(device)
        # state_dim = torch.unsqueeze(state_dim, 0)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4) 
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) 
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) 
        # self.bn3 = nn.BatchNorm2d(64)

        self.f1 = nn.Linear(108800, 512)

        self.val_hidden = nn.Linear(512, 128)
        self.adv_hidden = nn.Linear(512, 128)

        self.val = nn.Linear(128, 1) # State value

        self.adv = nn.Linear(128, action_dim) # Advantage value

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, 304, 432)) 

        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        # x = self.bn3(x)
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

        adv_ave = torch.mean(adv)

        x = adv + val - adv_ave
        x = x.squeeze(0) 
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

def d3qn_act(isPlayer2: bool, state: np.ndarray, model: Dueling_D3QN) -> int:
    if not isPlayer2:
        np.flip(state, axis = 1)
    state = torch.as_tensor(state, dtype=torch.float32).to(device)
    model = model.to(device)
    # Get the model
    return model.select_action(state)
