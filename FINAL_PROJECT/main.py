from game import *  
from Actions.DQN_Action import *  

mode = "Play"
P1_mode = "Old_AI"
P2_mode = "Old_AI"

if __name__ == '__main__':
    # Create players and game
    P1 = Player(P1_mode, False)
    P2 = Player(P2_mode, True)
    Pikachu = Game(mode, P1_mode, P2_mode, True)

    if mode == "Play":
        # Keep update the scene.
        while True:
            reward, state = Pikachu.update(P1.get_act(Pikachu.env), P2.get_act(Pikachu.env))

    elif mode == "Train":
        # Train DQN
        n_state = Pikachu.env.observation_space.shape
        n_action = 18
        target_network = Dueling_DQN(n_state, n_action).to(device)
        network = Dueling_DQN(n_state, n_action).to(device)
        target_network.load_state_dict(network.state_dict())
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
        r = 0
        c = 0

        GAMMA = 0.99
        BATH = 256
        EXPLORE = 2000000
        REPLAY_MEMORY = 50000
        BEGIN_LEARN_SIZE = 1024
        memory = Memory(REPLAY_MEMORY)
        UPDATA_TAGESTEP = 200
        learn_step = 0
        epsilon = 0.2
        writer = SummaryWriter('logs/dueling_DQN2')
        FINAL_EPSILON = 0.00001

        for epoch in count():
            state = Pikachu.reset()
            episode_reward = 0
            c += 1
            while True:
                state = state / 255
                p = random.random()
                if p < epsilon:
                    action = random.randint(0, n_action-1)
                else:
                    state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
                    action = network.select_action(state_tensor)
                reward, next_state = Pikachu.update(P1.get_act(Pikachu.env), action)
                episode_reward += reward
                memory.add(state, action, reward, next_state, reward != 0) # done = (reward != 0)
                if memory.size() > BEGIN_LEARN_SIZE:
                    print("learning session started")
                    learn_step += 1

                    if learn_step % UPDATA_TAGESTEP:
                        target_network.load_state_dict(network.state_dict())

                    states, actions, rewards, next_states, dones = memory.sample(BATH)

                    # states = torch.tensor(states).to(device)
                    # print(states.dtype)
                    # rewards = torch.tensor(rewards).to(device)
                    # dones = torch.tensor(dones).to(device)
                    # actions = torch.tensor(actions).to(device)
                    # next_states = torch.tensor(next_states).to(device)

                    states = torch.FloatTensor(np.array(states)).to(device)
                    actions = torch.LongTensor(actions).to(device)
                    rewards = torch.FloatTensor(rewards).to(device)
                    next_states = torch.FloatTensor(np.array(next_states)).to(device)
                    dones = torch.FloatTensor(np.array(dones)).to(device)

                    indices = np.arange(BATH)

                    target_Q_next = target_network(next_states)
                    Q_next = network(states)[indices, actions]

                    Q_max_action = torch.argmax(network(states), dim=1)
                    y = rewards + GAMMA*target_Q_next[indices, actions]
                    
                    loss = F.mse_loss(y, Q_next)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    writer.add_scalar('loss', loss.item(), global_step=learn_step)

                    if epsilon > FINAL_EPSILON: #
                        epsilon -= (0.1 - FINAL_EPSILON) / EXPLORE
                if reward != 0:
                    break
                state = next_state
            
            r += episode_reward
            writer.add_scalar('episode reward : ', episode_reward, global_step=epoch)
            if epoch % 50 == 0:
                print(f"{epoch/50} epoch reward : {r / 50}", epsilon)
                r = 0
            if epoch % 10 == 0:
                torch.save(network.state_dict(), 'model/network{}.pt'.format("dueling"))