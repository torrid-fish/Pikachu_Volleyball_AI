import random
from collections import deque
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
from main import STATE_MODE
from gym_pikachu_volleyball.envs.constants import *

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

        if STATE_MODE == "gray_scale":

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
        
        elif STATE_MODE == "info_vector":

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
        if STATE_MODE == "gray_scale":
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
        
        elif STATE_MODE == "info_vector":
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

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        self.size = min(self.capacity, self.size + 1)


    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        return observations, actions, rewards, next_observations, dones


def d3qn_act(isPlayer2: bool, state, model: Dueling_D3QN) -> int:
    # Flip state for other side, assume we always let computer at right side (P2)
    if not isPlayer2:
        if STATE_MODE == "gray_scale":
            np.flip(state, axis = 1)
        elif STATE_MODE == "info_vector":
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


class SumTree(object):
    data_pointer = 0
    
    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
    
    
    # Here we define function that will add our priority score in the sumtree leaf and add the experience in data:
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update (tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer = self.data_pointer + 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0
            
    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
        
    # Here build a function to get a leaf from our tree. So we'll build a function to get the leaf_index, priority value of that leaf and experience associated with that leaf index:
    def get_leaf(self, v):
        parent_index = 0

        # the while loop is faster than the method in the reference code
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node

# Now we finished constructing our SumTree object, next we'll build a memory object.
class PER(object):  # stored as ( state, action, reward, next_state ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree 
        self.tree = SumTree(capacity)
        self.size = 0
        self.capacity = capacity
        
    # Next, we define a function to store a new experience in our tree.
    # Each new experience will have a score of max_prority (it will be then improved when we use this exp to train our DDQN).
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.size = min(self.capacity, self.size + 1)

        self.tree.add(max_priority, experience)   # set the max priority for new priority
        
    # Now we create sample function, which will be used to pick batch from our tree memory, which will be used to train our model.
    # - First, we sample a minibatch of n size, the range [0, priority_total] into priority ranges.
    # - Then a value is uniformly sampled from each range.
    # - Then we search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.
    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        minibatch = []

        b_idx = np.empty((n,), dtype=np.int32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            b_idx[i]= index

            minibatch.append([data[0],data[1],data[2],data[3],data[4]])

        return b_idx, minibatch
    
    # Update the priorities on the tree
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)