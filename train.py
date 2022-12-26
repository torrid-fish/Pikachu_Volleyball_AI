import numpy as np
import torch
import torch.nn.functional as F
from game import *  
from itertools import count
from torchsummary import summary
from Actions.D3QN_Action import Dueling_D3QN
import threading
from player import Player
from game import Game
import random
import os

## HYPER PARAMETERS ##

# Infomation vector
INPUT_DIM = 17

# Estimated Q_value for each action
OUTPUT_DIM = 18 

# number of actor
ACT_NUM = 4

# number of actors in a row
WID_NUM = 2

# PATH of actors
PATH = [f'./model/D3QN_SMIV_A{i}.pth' for i in range(ACT_NUM)]

# Opponent
Opponent = "Attacker"
P1 = Player(Opponent, False)

# The decease rate of reward
GAMMA = 0.99

# The size of batch we used in learner
MINI_BATCH_LEN = 128

# The size of our memory
REPLAY_MEMORY = 4096

# The begin value of epsilon
BEGIN_EPSILON = 0.2
epsilon = BEGIN_EPSILON

# The final value of epsilon
FINAL_EPSILON = 0.0001 

# The decrease rate of epsilon
EXPLORE = 250000 

# When counter reach this value, update target_network
TARGET_UPDATE_PERIOD = 100

# Learn rate of optimizer
LEARN_RATE = 0.01

# Counter of exprience
exp_cnt = 0

# Resolution ratio of screen
RESOLUTION_RATIO = 0.5

# Whether train with multi-threading mode
MULTI_THEARDING = False

# How often we store q_network
UPDATE_Q_NETWORK_PERIOD = 10

# The device we used (either CPU or GPU)
if torch.cuda.is_available():
    device = "cuda:0"
    print("GPU is used.")
else:
    device = "cpu"
    print("CPU is used.")
######################

## Implement Memory ##

class PER:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

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

######################

## Implement functions ##

def compute_td_targets(rewards: torch.tensor, dones: torch.tensor, q_network, target_network, next_states):
    """
    ## Formula
    - action\n
    `action = argmax_{a}{q_network(next_state, a)}`\n
    action is the best operation that the network can do.
    
    - td_target\n
    `td_target = reward + GAMMA * target_network(next_state, action)`\n
    Then the estimated Q is computed as above.

    ## Return
    A list of td_targets.\n
    - type = `list`, shape = `[MINI_BATCH_LEN]`
    """
    # Compute the TD targets for each transition
    td_targets = []
    for i in range(MINI_BATCH_LEN):
        reward = rewards[i]
        done = dones[i]
        next_state = next_states[i]

        if done:
            # If the episode is done, the TD target is the reward
            td_target = reward
        else:
            # Otherwise, the TD target is the reward plus the discounted maximum action-value for the next state
            action = q_network(next_state).argmax()
            td_target = reward + GAMMA * target_network(next_state)[0, action]

        td_targets.append(td_target)

    return td_targets

def compute_td_error(state: torch.tensor, action, reward, next_state: torch.tensor, done, q_network, target_network):
    """
    Compute td_error for single exp.
    ## Return
    td_error.\n
    - type = float
    """
    # Compute td_targets
    if done:
        # If the episode is done, the TD target is the reward
        td_target = reward
    else:
        # Otherwise, the TD target is the reward plus the discounted maximum action-value for the next state
        action = q_network(next_state).argmax()
        td_target = reward + GAMMA * target_network(next_state)[0, action]

    # Compute q_value
    q_value = q_network(state)[0, action]
    # Return td_error
    return torch.abs(q_value - td_target)

def compute_loss_and_grad(states: torch.tensor, actions: torch.tensor, td_targets: torch.tensor, q_network, optimizer, is_weight):
    """
    This function will use td_targets and q_network(state) to calculate loss.\n
    Here, we use `huber_loss` to calculate loss.\n
    Also, it will utilize our optimizer to calculate gradiant of all vairables on q_network.
    ### Return
    #### loss
    The loss of this learn.
    - type = `float`
    #### grads
    The gardiant of element in the network.
    - type = `list`, shape = `[q_network.paraemters()]`
    """
    # Compute the current action-value estimates
    q_values = q_network(states)
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    # Compute the loss (Important sample weight)
    loss = (torch.tensor(is_weight, dtype= torch.float32).to(device) * F.mse_loss(td_targets, q_selected)).mean()

    # Compute the gradients
    optimizer.zero_grad()
    loss.backward()
    grads = [param.grad for param in q_network.parameters()]

    return loss, grads

#########################

## Impelment learner and actor ##

def learner(replay_buffer: PER, q_network, target_network, optimizer: torch.optim.Adam):
    """
    This function will learn one sample (size = `MINI_BATCH_SIZE`) and update the q_network.
    """
    # Sample a batch of transitions from the replay buffer
    mini_batch, idxs, is_weight = replay_buffer.sample(MINI_BATCH_LEN)

    states, actions, rewards, next_states, dones = [], [], [], [], []
    for batch in mini_batch:
        states += [batch[0].reshape(17).clone().detach()]
        actions += [batch[1]]
        rewards += [batch[2]]
        next_states += [batch[3].reshape(17).clone().detach()]
        dones += [int(batch[4])]

    states = torch.stack(states).to(device)
    actions = torch.LongTensor(list(actions)).to(device)
    rewards = torch.FloatTensor(list(rewards)).to(device)
    next_states = torch.stack(next_states).to(device)
    dones = torch.LongTensor(dones).to(device)
    
    # Compute the TD targets
    td_targets = compute_td_targets(rewards, dones, q_network, target_network, next_states)
    td_targets = torch.stack(td_targets).to(device)

    # Compute the loss and gradients
    loss, grads = compute_loss_and_grad(states, actions, td_targets, q_network, optimizer, is_weight)

    # Update the variables in network
    optimizer.step()

    # Compute q_value
    q_value = q_network(states)[0, actions]

    # Compute td_errors
    td_errors = torch.abs(td_targets - q_value)
    td_errors = td_errors.data.cpu().numpy()
    replay_buffer.update(idxs, td_errors)

    return loss.cpu().detach().numpy()

def actor(replay_buffer: PER, game: Game, q_network, target_network, screen, state):
    global exp_cnt, epsilon

    # Select an action using an epsilon-greedy policy
    if np.random.rand() < epsilon:
        action = np.random.randint(OUTPUT_DIM)
        game.is_random = True
    else:
        action = q_network(state).argmax()
        game.is_random = False

    # Update epsilon
    epsilon -= (BEGIN_EPSILON - FINAL_EPSILON) / EXPLORE
    epsilon = max(epsilon, FINAL_EPSILON)
    game.epsilon = epsilon

    # Take the action and observe the next state, reward, and done flag
    reward, next_state, done = game.update(P1.get_act(game.env), action, screen)

    # Change state, next_state to tensor
    state = state.reshape(1, INPUT_DIM)
    state = state.clone().detach() # MUST ON DEVICE
    next_state = next_state.reshape(1, INPUT_DIM)
    next_state = torch.FloatTensor(next_state).to(device)

    # Compute the TD error
    td_error = compute_td_error(state, action, reward, next_state, done, q_network, target_network)

    # Store the transition in the replay buffer with the TD error as a weight
    replay_buffer.add(td_error, (state, action, reward, next_state, done))

    # Gain one more exp
    exp_cnt += 1

    # Update done and state
    return done, next_state

def call_actor(replay_buffer: PER, game: Game, q_network, target_network, path, screen):
    cnt = 0
    while True:
        cnt += 1
        # Get init state
        state = game.reset(True)
        # Convert state to tensor
        state = state.reshape(1, INPUT_DIM)
        state = torch.FloatTensor(state).to(device)
        done = False
        # Complete one round
        while not done:
            # Update done and state
            done, state = actor(replay_buffer, game, q_network, target_network, screen, state)

        # Save model for enough rounds
        if cnt % UPDATE_Q_NETWORK_PERIOD == 0:
            torch.save(q_network, path)
        
#################################

def train():
    ## Initialize model ##
    # List of q_network
    q_network = []
    # Try to load existed model, otherwise generate a new one
    for i in range(ACT_NUM):
        try:
            q_network += [torch.load(PATH[i])]
            print(f'Q_network {i}: Loaded!')
        except FileNotFoundError:
            q_network += [Dueling_D3QN(OUTPUT_DIM)]
            print(f'Q_network {i}: Created!')

        # Set q network to train mode
        q_network[i].train()

    # Random a q_network as target_network
    target_network = q_network[random.randrange(ACT_NUM)]

    # Set target network to train mode
    target_network.train()

    # Output the summary of the network
    for i in range(ACT_NUM):
        summary(q_network[i])

        # Store to GPU / CPU
        q_network[i] = q_network[i].to(device)

    # Store to GPU / CPU
    target_network = target_network.to(device)
    ######################

    ## Initialize optimizer ##
    optimizer = [torch.optim.Adam(q_network[i].parameters(), lr=LEARN_RATE) for i in range(ACT_NUM)]
    ##########################
    
    ## Initialize Memory ##
    # Shared memory
    memory = PER(REPLAY_MEMORY)
    #######################

    ## Initialize Memory ##
    Pikachu = [Game("Train", Opponent, "D3QN", RESOLUTION_RATIO, i % WID_NUM, i // WID_NUM) for i in range(ACT_NUM)]
    #######################

    ## Parallel Computing learner and actor ##
    # Init pygame screen
    WIDTH = 1060 * RESOLUTION_RATIO
    HEIGHT = 304 * RESOLUTION_RATIO
    HEI_NUM = np.ceil(ACT_NUM / WID_NUM)
    screen = pygame.display.set_mode((WIDTH * WID_NUM + (WID_NUM + 1) * 30, HEIGHT * HEI_NUM + (HEI_NUM + 1) * 30))
    pygame.display.set_caption('Pikachu_Volleyball')

    if MULTI_THEARDING:
        # Use Multi-threading to execute
        ACTOR = []
        # Create multiple actors and start discovering
        for i in range(ACT_NUM):
            ACTOR += [threading.Thread(target = call_actor, args = (memory, Pikachu[i], q_network[i], target_network, PATH[i], screen))]
            ACTOR[i].start()

        cnt = 0
        history_winrt = np.ndarray((0, ACT_NUM))
        history_loss = np.ndarray((0, ACT_NUM))

        while True:
            loss = []

            # Update the window
            pygame.display.flip()
            
            # If the window was closed, end the game
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # If experience is not enough, don't learn
            if exp_cnt < MINI_BATCH_LEN:
                os.system('cls')
                print('Collecting Data ', end='')
                for i in range(exp_cnt % 3 + 1): print('.', end='')
                print('')
                continue

            # Learning for each network
            for i in range(ACT_NUM):
                loss += [learner(memory, q_network[i], target_network, optimizer[i])]
            
            if cnt % 10 == 0:
                target_network = q_network[loss.index(min(loss))]

            cnt += 1
            
            history_winrt = np.append(history_winrt, np.array([[Pikachu[i].prewinrt for i in range(ACT_NUM)]]), axis=0)
            history_loss = np.append(history_loss, np.array([loss]), axis=0)
            np.save("history_winrt.npy", history_winrt)
            np.save("history_loss.npy", history_loss)

            os.system('cls')
            print(f'epoch = {cnt}')
            print('=============================================================')
            for i in range(ACT_NUM):
                print(f'Network{i} loss: {loss[i]:.4f} / pre win rate: {Pikachu[i].prewinrt:.4f} / round: {Pikachu[i].tot}')
                if i != ACT_NUM - 1: print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')      
            print('=============================================================')  
    else:
        # Only use sigle thread to train
        cnt = 0
        history_winrt = np.ndarray((0, ACT_NUM))
        history_loss = np.ndarray((0, ACT_NUM))

        # Initialize dones and states
        dones, states = [], []
        for i in range(ACT_NUM):
            dones += [0]
            state = Pikachu[i].reset(True)
            # Convert state to tensor
            state = state.reshape(1, INPUT_DIM)
            state = torch.FloatTensor(state).to(device)
            states += [state]

        # Keep looping
        while True:
            loss = []

            # Each actor do one action
            for i in range(ACT_NUM):
                # Update dones and states
                dones[i], states[i] = actor(memory, Pikachu[i], q_network[i], target_network, screen, states[i])  
                # Reset game if done
                if dones[i]:
                    state = Pikachu[i].reset(True)
                    state = state.reshape(1, INPUT_DIM)
                    state = torch.FloatTensor(state).to(device)
                    states[i] = state
                    # Save model for enough rounds
                    if Pikachu[i].tot % UPDATE_Q_NETWORK_PERIOD == 0:
                        torch.save(q_network[i], PATH[i])           

            # Update the window (We then make sure all infomation will be drew)
            pygame.display.flip()
            
            # If the window was closed, end the game
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # If experience is not enough, don't learn
            if exp_cnt < MINI_BATCH_LEN:
                os.system('cls')
                print('Collecting Data ', end='')
                for i in range(exp_cnt % 3 + 1): print('.', end='')
                print('')
                continue

            # Learning for each network
            for i in range(ACT_NUM):
                loss += [learner(memory, q_network[i], target_network, optimizer[i])]
                Pikachu[i].loss = loss[i]
            
            if cnt % 10 == 0:
                target_network = q_network[loss.index(min(loss))]

            cnt += 1
            
            history_winrt = np.append(history_winrt, np.array([[Pikachu[i].prewinrt for i in range(ACT_NUM)]]), axis=0)
            history_loss = np.append(history_loss, np.array([loss]), axis=0)
            np.save("history_winrt.npy", history_winrt)
            np.save("history_loss.npy", history_loss)

            os.system('cls')
            print(f'epoch = {cnt}')
            print('=============================================================')
            for i in range(ACT_NUM):
                print(f'Network{i} loss: {loss[i]:.4f} / pre win rate: {Pikachu[i].prewinrt:.4f} / round: {Pikachu[i].tot}')
                if i != ACT_NUM - 1: print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')      
            print('=============================================================')  

    ############################################
