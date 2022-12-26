from itertools import count
from torchsummary import summary
import torch
from gym_pikachu_volleyball.envs.constants import *
from player import Player
from game import Game
from network import *
import random

# Choose CPU or GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

"""
    Constants
"""
# Input/Output dimension
INPUT_DIM = 17
OUTPUT_DIM = 18

# Reward gamma
GAMMA = 0.8

# Size of slice
BATCH_SIZE = 256

# Number of times to update target network
UPDATA_TAGESTEP = 10

# Memory size
REPLAY_MEMORY = 16384

# This should be smaller than `REPLAY_MEMORY` to start learning
BEGIN_LEARN_SIZE = 512 

# Learning rate
LEARNING_RATE = 1e-7

# Begin value of epsilon
BEGIN_EPSILON = 0.2

# Smallest epsilon
FINAL_EPSILON = 0.0001

# Controlling the epsilon
EXPLORE = 500000

"""
    Train function
"""
def init(P1_MODE, P2_MODE, P1_TAG, P2_TAG):
    # Create one game and opponent
    P1 = Player(P1_MODE, False, P1_TAG)
    Pikachu = Game("Train", P1_MODE, P2_MODE)

    try:
        network = torch.load(P2_TAG)
        target_network = torch.load(P2_TAG)
        memory = torch.load('./memory/' + P2_TAG + '.pth')
        epsilon = torch.load('./log/' + P2_TAG + '.pth')['epsilon']
        pre_result = torch.load('./log/' + P2_TAG + '.pth')['pre_result']
        winrts = torch.load('./log/' + P2_TAG + '.pth')['winrts']
        losses = torch.load('./log/' + P2_TAG + '.pth')['losses']
        print('Load previous data successfully.')
    except FileNotFoundError:
        network = Dueling_D3QN(OUTPUT_DIM)
        target_network = Dueling_D3QN(OUTPUT_DIM)
        pre_result = []
        memory = PER(REPLAY_MEMORY)
        epsilon = BEGIN_EPSILON
        winrts = []
        losses = []
        print('Create new model successfully.')

    # Update Game value
    Pikachu.epsilon = epsilon
    Pikachu.losses = losses
    Pikachu.winrts = winrts
    Pikachu.pre_result = pre_result
    Pikachu.tot = len(winrts)

    # Set to train mode
    network.train()
    target_network.train()
    summary(network)

    # Store to GPU / CPU
    network = network.to(device)
    target_network = target_network.to(device)

    # Gradiant Optimizer ADAM
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    return network, target_network, P1, Pikachu, optimizer, memory, epsilon, winrts, losses

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
    reward, next_state, done = Pikachu.update(P1.get_act(Pikachu.env), action)

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

def train(P1_MODE, P2_MODE, P1_TAG, P2_TAG):
    # Initialize Train Process
    network, target_network, P1, Pikachu, optimizer, memory, epsilon, winrts, losses = init(P1_MODE, P2_MODE, P1_TAG, P2_TAG)
    
    for round in count():
        # Reset environment
        state = Pikachu.reset(True)

        learn_step = 0
        # Keep learning until gain reward
        while True:
            # Actor gain one experience and update state
            state, epsilon = actor(network, target_network, P1, Pikachu, memory, state, epsilon)

            # If experience is enough, start learning
            if memory.size() >= BEGIN_LEARN_SIZE:
                # Learner learn one BATCH_SIZE experience and update network
                learner(memory, network, target_network, Pikachu, optimizer, losses)
            
                # Update target network
                if learn_step % UPDATA_TAGESTEP == 0:
                    target_network.load_state_dict(network.state_dict())

                # Learner learn once
                learn_step += 1
            
            # Keep learning until one game is set, we go to next round.
            if Pikachu.done:
                winrts += [Pikachu.prewinrt]
                torch.cuda.empty_cache()
                break

        # With enough game round, we save trained model.
        if round % UPDATA_TAGESTEP == 0 and round != 0 and memory.size() >= BEGIN_LEARN_SIZE:
            print("Data saved!")
            torch.save(network, './model/' + P2_TAG + '.pth')
            torch.save(memory, './memory/' + P2_TAG + '.pth')
            torch.save({'losses': losses, 'winrts': winrts, 'epsilon': epsilon, 'pre_result': Pikachu.pre_result}, './log/' + P2_TAG + '.pth')