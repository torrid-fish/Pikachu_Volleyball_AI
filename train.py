from itertools import count
from torchsummary import summary
import torch
from gym_pikachu_volleyball.envs.constants import *
from player import Player
from game import Game
from network import *
import random
import time

# Choose CPU or GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

"""
    Constants
"""
# Input/Output dimension
INPUT_DIM = 17
OUTPUT_DIM = 18

# Reward gamma
GAMMA = 0.99

# Size of slice
BATCH_SIZE = 128

# number of actors in a row
WID_NUM = 2

# Number of times to update target network
UPDATA_TAGESTEP = 10

# Memory size
REPLAY_MEMORY = 8192

# This should be smaller than `REPLAY_MEMORY` to start learning
BEGIN_LEARN_SIZE = 2048

# Learning rate
LEARNING_RATE = 1e-5

# Begin value of epsilon
BEGIN_EPSILON = 0.15

# Smallest epsilon
FINAL_EPSILON = 0.001

# Controlling the epsilon
EXPLORE = 100000

"""
    Train function
"""
def init(P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY, ACTOR_NUM):
    # Config window for APEX
    global WID_NUM
    WID_NUM = min(WID_NUM, ACTOR_NUM)
    HEI_NUM = np.ceil(ACTOR_NUM / WID_NUM)
    SCREEN_SIZE = (1060 * RESOLUTION_RATIO * WID_NUM + (WID_NUM + 1) * 30, 304 * RESOLUTION_RATIO * HEI_NUM + (HEI_NUM + 1) * 30)
    # Create one opponent
    P1 = Player(P1_MODE, False, P1_TAG)

    # Init memory
    try:
        memory = torch.load('./memory/' + P2_TAG + '.pth')
        print('Load previous memory successfully.')
    except FileNotFoundError:
        memory = PER(REPLAY_MEMORY)
        print('Can not find previoous memory, create a new memory.')

    # Randomly choose one as target_network
    try:
        target_network = torch.load('./model/' + P2_TAG + f'_{random.randrange(ACTOR_NUM)}.pth')
        print('Load previous network successfully.')
    except FileNotFoundError:
        target_network = Dueling_D3QN(OUTPUT_DIM)
        print('Can not find previoous network, create a new memory.')

    # Init multiple training gym
    Pikachus, networks, optimizers, epsilons, losses_list = [], [], [], [], []
    for i in range(ACTOR_NUM):
        POS = (i % WID_NUM, i // WID_NUM)
        Pikachu = Game("Train", P1_MODE, P2_MODE, RESOLUTION_RATIO, DISPLAY == 'PYGAME', SCREEN_SIZE, POS)

        try:
            network = torch.load('./model/' + P2_TAG + f'_{i}.pth')
            print('Load previous network successfully.')
        except FileNotFoundError:
            network = Dueling_D3QN(OUTPUT_DIM)
            print('Can not find previoous network, create a new memory.')

        try:
            epsilon = torch.load('./log/' + P2_TAG + f'_{i}.pth')['epsilon']
            pre_result = torch.load('./log/' + P2_TAG + f'_{i}.pth')['pre_result']
            winrts = torch.load('./log/' + P2_TAG + f'_{i}.pth')['winrts']
            losses = torch.load('./log/' + P2_TAG + f'_{i}.pth')['losses']
            print('Load previous log successfully.')
        except FileNotFoundError:
            pre_result = []
            epsilon = BEGIN_EPSILON
            winrts = []
            losses = []
            print('Can not find previous log, create a new log.')

        # Update Game value
        Pikachu.epsilon = epsilon
        Pikachu.losses = losses
        Pikachu.winrts = winrts
        Pikachu.pre_result = pre_result
        Pikachu.tot = len(winrts)

        # Set to train mode
        network.train()
        target_network.train()
        # summary(network)

        # Store to GPU / CPU
        network = network.to(device)
        target_network = target_network.to(device)

        # Gradiant Optimizer ADAM
        optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

        Pikachus += [Pikachu] 
        networks += [network] 
        optimizers += [optimizer] 
        epsilons += [epsilon] 
        losses_list += [losses]

    return networks, target_network, P1, Pikachus, optimizers, memory, epsilons, losses_list

def actor(network, target_network, P1, Pikachu, memory: PER, state, epsilon):
    p = random.random()
    if p < epsilon: # Choose random action
        action = random.randint(0, OUTPUT_DIM-1)
        Pikachu.is_random = True
    else: # Use model to predict next action
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        action = network.select_action(state_tensor)
        Pikachu.is_random = False

    # Update epsilon
    epsilon -= (BEGIN_EPSILON - FINAL_EPSILON) / EXPLORE
    epsilon = max(epsilon, FINAL_EPSILON)
    Pikachu.epsilon = epsilon

    # Interact with environment
    reward, next_state, done = Pikachu.update(P1.get_act(Pikachu.env, state), action)

    # Turn next_state into tensor
    _state = state.reshape(1, INPUT_DIM)
    _state = torch.FloatTensor(state).to(device)
    _next_state = next_state.reshape(1, INPUT_DIM)
    _next_state = torch.FloatTensor(next_state).to(device)

    # Compute td_error = abs(q_value - td_target)
    if done:
        # If the episode is done, the TD target is the reward
        td_target = reward
    else:
        # Otherwise, the TD target is the reward plus the discounted maximum action-value for the next state
        action = network(_next_state).argmax()
        td_target = reward + GAMMA * target_network(_next_state)[0, action]
    # Compute q_value
    q_value = network(_state)[0, action]
    # Return td_error
    td_error =  torch.abs(q_value - td_target)

    data = np.array([state, int(action), reward, next_state, done], dtype=object)
    # Add transition
    memory.add(td_error, data)

    return next_state, epsilon

def learner(memory: PER, network, target_network, Pikachu, optimizer, losses):
    # Sample data from memory
    mini_batch, idxs, is_weight = memory.sample(BATCH_SIZE)
    mini_batch = np.array(mini_batch).transpose()
    # Put parameters to GPU / CPU
    states = torch.FloatTensor(np.vstack(mini_batch[0])).to(device)
    actions = torch.LongTensor(list(mini_batch[1])).to(device)
    rewards = torch.FloatTensor(list(mini_batch[2])).to(device)
    next_states = torch.FloatTensor(np.vstack(mini_batch[3])).to(device)
    dones = torch.LongTensor(mini_batch[4].astype(int)).to(device)

    # [0, 1, 2, ..., BATH-1]       
    indices = np.arange(BATCH_SIZE)

    # Double D3QN: (state, action, reward, next_state)
    #  
    #       reward + GAMMA * QT(next_state, maxarg_a{Q(next_state, a)}) <- Q(state, action)
    #
    # QT : target Q-network
    # Q  : Q-network (learner)
    
    # Q_pred: Q(state, action)
    # Shape: [BATH, 1]
    Q_pred = network(states)[indices, actions]

    # Q_next: QT(next_state, action)
    # Shape: [BATH, n_Action]
    Q_temp = target_network(next_states)

    # QT(next_state, maxarg_a{Q(next_state, a)})
    # Shape: [BATH, 1]
    Q_temp = Q_temp[indices, torch.argmax(network(next_states), dim=1)]
    
    # Q_target: reward + GAMMA * QT(next_state, maxarg_a{Q(next_state, a)})
    # Shape: [BATH, 1]
    Q_temp[dones] = 0
    Q_target = rewards + GAMMA * Q_temp
    
    
    loss = (torch.tensor(is_weight, dtype= torch.float32).to(device) * F.mse_loss(Q_target, Q_pred)).mean()
    # loss = F.mse_loss(Q_target, Q_pred)

    # Update priority in sum tree
    errors = np.abs((Q_target - Q_pred).data.cpu().numpy())
    # Update priority in ReplayBuffer
    memory.update(idxs, errors)

    Pikachu.loss = float(loss)
    losses += [float(loss)]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

def print_info(Pikachus, losses_list):
    i = 0
    for Pikachu, losses in zip(Pikachus, losses_list):
        print(f'== Pikachu {i} ROUND {Pikachu.tot} ==')
        print(f'- Speed: {Pikachu.speed:.2f}')
        curtime = time.gmtime(time.perf_counter() - Pikachu.beg_time)
        print(f'- Time: {curtime.tm_hour:02d}:{curtime.tm_min:02d}:{curtime.tm_sec:02d}')
        print(f'- Win Rate: {Pikachu.prewinrt:.2f}')
        print(f'- Loss: {losses[-1]:.6f}')
        print(f'== Model {i} Data saved! ==\n')
        i += 1

