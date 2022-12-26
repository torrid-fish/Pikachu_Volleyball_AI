import random
from collections import deque
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
from gym_pikachu_volleyball.envs.constants import *

# The device we used (either CPU or GPU)
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
    def __init__(self, action_dim):
        super(Dueling_D3QN, self).__init__()

        self.f1 = nn.Linear(17, 512)

        self.val_hidden = nn.Linear(512, 256)
        self.adv_hidden = nn.Linear(512, 256)

        self.val = nn.Linear(256, 1) # State value
        self.adv = nn.Linear(256, action_dim) # Advantage value

        torch.nn.init.normal_(self.val_hidden.weight, 0, 0.02)
        torch.nn.init.normal_(self.val.weight, 0, 0.02)
        torch.nn.init.normal_(self.adv_hidden.weight, 0, 0.02)
        torch.nn.init.normal_(self.adv.weight, 0, 0.02)

    def forward(self, x):
        x = torch.reshape(x, (-1, 17)) 

        x = self.f1(x)
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

def d3qn_act(isPlayer2: bool, state, model: Dueling_D3QN) -> int:
    # Flip state for other side, assume we always let computer at right side (P2)
    if not isPlayer2:
        temp = state
        """
        state = np.array([
            P1.x, P1.y, P1.y_velocity, P1.state, P1.diving_direction, P1.lying_down_duration_left,
            P2.x, P2.y, P2.y_velocity, P2.state, P2.diving_direction, P2.lying_down_duration_left,
            ball.x, ball.y, ball.x_velocity, ball.y_velocity, ball.is_power_hit
        ])
        """
        # P1.x, P1.y = 1 - P2.x, P2.y
        state[0], state[1] = 1 - temp[6], temp[7]
        # P2.x, P2.y = 1 - P1.x, P1.y
        state[6], state[7] = 1 - temp[0], temp[1]
        # swap(state[2:6], state[8:12])
        state[2], state[3], state[4], state[5] = temp[8], temp[9], temp[10], temp[11]
        state[8], state[9], state[10], state[11] = temp[2], temp[3], temp[4], temp[5]
        # ball.x = 1 - ball.x
        state[12] = 1 - temp[12]
        # ball.y = ball.y
        state[13] = temp[13]
        # ball.x_velocity = -ball.x_velocity
        state[14] = -temp[14]
        # ball.y_velocity = ball.y_velocity
        state[15] = temp[15]
        # ball.is_power_hit
        state[16] = temp[16]

    state = torch.as_tensor(state, dtype=torch.float32).to(device)
    model = model.to(device)
    # Get the model
    return model.select_action(state)


