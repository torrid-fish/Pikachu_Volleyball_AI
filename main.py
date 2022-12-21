from game import *  
from Actions.D3QN_Action import *  

mode = "Train"
P1_mode = "Old_AI"
P2_mode = "D3QN"


if __name__ == '__main__':
    # Create players and game
    P1 = Player(P1_mode, False)
    P2 = Player(P2_mode, True)
    Pikachu = Game(mode, P1_mode, P2_mode, False)

    if mode == "Play":
        # Keep update the scene.
        while True:
            reward, state = Pikachu.update(P1.get_act(Pikachu.env), P2.get_act(Pikachu.env))

    elif mode == "Train":
        ## Train D3QN settings ##
        
        # IO dimension
        n_state = Pikachu.env.observation_space.shape[0:2] # Input shape (304, 432)
        n_action = 18 # Output shape (1, 18)

        # Initialize two models
        target_network = Dueling_D3QN(n_state, n_action)
        network = Dueling_D3QN(n_state, n_action)
        
        # Load in previous variables
        PATH = './model/networkdueling.pt' # The path of stored model
        try:
            network.load_state_dict(torch.load(PATH))
            target_network.load_state_dict(torch.load(PATH))
        except FileNotFoundError:
            network = Dueling_D3QN(n_state, n_action)
            target_network.load_state_dict(network.state_dict())

        # Store to GPU / CPU
        network = network.to(device)
        target_network = target_network.to(device)

        # Gradiant Optimizer ADAM
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        
        # Record variables
        epoch_reward = 0

        # Reward gamma
        GAMMA = 1
        # Size of slice
        BATH = 512
        
        # Memory relative
        REPLAY_MEMORY = 20
        BEGIN_LEARN_SIZE = 1024 # This should be smaller than `REPLAY_MEMORY` to start learning
        memory = Memory(REPLAY_MEMORY)

        # Number of times to update target network
        UPDATA_TAGESTEP = 200
        learn_step = 0

        # Epsilon relatife
        epsilon = 0.2
        FINAL_EPSILON = 0.00001 # Smallest epsilon
        EXPLORE = 2000000 # Controlling the epsilon

        for epoch in count():
            # Reset environment
            state = Pikachu.reset()
            episode_reward = 0

            # Keep learning until gain reward
            while True:
                # Normalize data
                state = state / 255
                
                p = random.random()
                if p < epsilon: # Choose random action
                    action = random.randint(0, n_action-1)
                else: # Use model to predict next action
                    state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
                    action = network.select_action(state_tensor)

                # Interact with environment
                reward, next_state, done = Pikachu.update(P1.get_act(Pikachu.env), action)

                # Update reward
                episode_reward += reward
                # Add experience to memory
                memory.add(state, action, reward, next_state, done)

                # Begin to learn
                if memory.size() >= BEGIN_LEARN_SIZE:
                    print("Learn session started")
                    learn_step += 1

                    # Update target network
                    if learn_step % UPDATA_TAGESTEP == 0:
                        target_network.load_state_dict(network.state_dict())

                    # Sample data from memory
                    states, actions, rewards, next_states, dones = memory.sample(BATH)

                    # Put parameters to GPU / CPU
                    states = torch.FloatTensor(np.array(states)).to(device)
                    actions = torch.LongTensor(actions).to(device)
                    rewards = torch.FloatTensor(rewards).to(device)
                    next_states = torch.FloatTensor(np.array(next_states)).to(device)
                    ###dones = torch.FloatTensor(np.array(dones)).to(device)
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
                    
                    # Compute loss
                    # loss = F.mse_loss(Q_target, Q_pred)
                    loss = F.huber_loss(Q_target, Q_pred)
                    Pikachu.the_loss = float(loss)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 

                    # Update epsilon
                    if epsilon > FINAL_EPSILON: 
                        epsilon -= (0.1 - FINAL_EPSILON) / EXPLORE

                # Keep learning
                if done:
                    torch.cuda.empty_cache()
                    break

                # Update state
                state = next_state
            
            epoch_reward += episode_reward

            if epoch % 50 == 0:
                print(f"{epoch/50} epoch reward : {epoch_reward / 50}", epsilon)
                epoch_reward = 0
                torch.save(network.state_dict(), PATH)