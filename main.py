from game import *  
from Actions.D3QN_Action import *  
from itertools import count
from torchsummary import summary

MODE = "Train"
# MODE = "Play"

# STATE_MODE = "gray_scale" 
STATE_MODE = "info_vector"

PER_ENABLE = True

P1_mode = "Attacker"
P2_mode = "D3QN"


if __name__ == '__main__':
    # Create players and game
    P1 = Player(P1_mode, False)
    P2 = Player(P2_mode, True)
    Pikachu = Game(MODE, P1_mode, P2_mode)

    if MODE == "Play":
        # Get initial state
        state = Pikachu.reset(True)
    
        # Keep update the scene.
        while True:
            reward, state, done = Pikachu.update(P1.get_act(Pikachu.env, state), P2.get_act(Pikachu.env, state))

    elif MODE == "Train":
        ## Train D3QN settings ##
        
        # IO dimension
        n_action = 18 # Output shape (1, 18)
        
        # Load in previous variables
        if STATE_MODE == "gray_scale":
            PATH = './model/D3QN_SMGS.pth' 
        elif STATE_MODE == "info_vector":
            PATH = './model/D3QN_SMIV.pth' 

        # Load or create model
        try:
            network = torch.load(PATH)
            target_network = torch.load(PATH)
        except FileNotFoundError:
            network = Dueling_D3QN(n_action)
            target_network = Dueling_D3QN(n_action)

        # Set to train mode
        network.train()
        target_network.train()
        summary(network)

        # Store to GPU / CPU
        network = network.to(device)
        target_network = target_network.to(device)

        # Gradiant Optimizer ADAM
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        
        # Record variables
        epoch_reward = 0

        # Reward gamma
        GAMMA = 0.99
        # Size of slice
        BATH = 128
        
        # Memory relative
        REPLAY_MEMORY = 4096
        BEGIN_LEARN_SIZE = 256 # This should be smaller than `REPLAY_MEMORY` to start learning

        # Choose which memory used
        if PER_ENABLE:
            memory = PER(REPLAY_MEMORY)
        else:
            memory = Memory(REPLAY_MEMORY)

        # Number of times to update target network
        UPDATA_TAGESTEP = 10
        learn_step = 0

        # Epsilon relatife
        BEGIN_EPSILON = 0.2
        epsilon = BEGIN_EPSILON
        FINAL_EPSILON = 0.0001 # Smallest epsilon
        EXPLORE = 250000 # Controlling the epsilon

        for epoch in count():
            # Reset environment
            state = Pikachu.reset(True)
            episode_reward = 0

            # Keep learning until gain reward

            while True:
                
                p = random.random()
                if p < epsilon: # Choose random action
                    action = random.randint(0, n_action-1)
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

                # Update reward
                episode_reward += reward

                # Add transition
                if PER_ENABLE:
                    memory.store((state, action, reward, next_state, done))
                else:
                    memory.add(state, action, reward, next_state, done)

                # Begin to learn
                if memory.size >= BEGIN_LEARN_SIZE:
                    learn_step += 1

                    # Update target network
                    if learn_step % UPDATA_TAGESTEP == 0:
                        target_network.load_state_dict(network.state_dict())

                    if PER_ENABLE:
                        # Sample data from memory
                        idxs, mini_batch = memory.sample(BATH)
                        mini_batch = np.array(mini_batch, dtype=object).transpose()
                        # Put parameters to GPU / CPU
                        states = torch.FloatTensor(np.vstack(mini_batch[0])).to(device)
                        actions = torch.LongTensor(list(mini_batch[1])).to(device)
                        rewards = torch.FloatTensor(list(mini_batch[2])).to(device)
                        next_states = torch.FloatTensor(np.vstack(mini_batch[3])).to(device)
                        dones = torch.LongTensor(mini_batch[4].astype(int)).to(device)
                    else:
                        # Sample data from memory
                        states, actions, rewards, next_states, dones = memory.sample(BATH)
                        # Put parameters to GPU / CPU
                        states = torch.FloatTensor(np.array(states)).to(device)
                        actions = torch.LongTensor(actions).to(device)
                        rewards = torch.FloatTensor(rewards).to(device)
                        next_states = torch.FloatTensor(np.array(next_states)).to(device)
                        dones = torch.LongTensor(np.array(dones)).to(device)

                    # [0, 1, 2, ..., BATH-1]       
                    indices = np.arange(BATH)

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
                    
                    
                    # loss = F.mse_loss(Q_target, Q_pred)
                    loss = F.huber_loss(Q_target, Q_pred)

                    # Update priority in sum tree
                    if PER_ENABLE:
                        errors = np.abs((Q_target - Q_pred).data.cpu().numpy())
                        # Update priority in ReplayBuffer
                        memory.batch_update(idxs, errors)

                    Pikachu.loss = float(loss)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 

                # Keep learning
                if done:
                    torch.cuda.empty_cache()
                    break

                # Update state
                state = next_state
            
            epoch_reward += episode_reward

            if epoch % UPDATA_TAGESTEP == 0 and epoch != 0 and memory.size >= BEGIN_LEARN_SIZE:
                print(f"{epoch} epoch reward : {epoch_reward / UPDATA_TAGESTEP}")
                print(f"{epoch} epoch loss : {loss}")
                print("Model saved!")
                epoch_reward = 0
                torch.save(network, PATH)