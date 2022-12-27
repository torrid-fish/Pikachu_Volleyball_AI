from train import *

def play(P1_MODE, P2_MODE, P1_TAG, P2_TAG):
    # Create players and game
    P1 = Player(P1_MODE, False, P1_TAG)
    P2 = Player(P2_MODE, True, P2_TAG)
    Pikachu = Game("Play", P1_MODE, P2_MODE, 2, True)

    # Get initial state
    state = Pikachu.reset(True)

    # Keep update the scene
    while True:
        reward, state, done = Pikachu.update(P1.get_act(Pikachu.env, state), P2.get_act(Pikachu.env, state))

def train(P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY, ACTOR_NUM):
    # Initialize Train Process
    networks, target_network, P1, Pikachus, optimizers, memory, epsilons, losses_list = init(P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY, ACTOR_NUM)
    
    rounds = [0 for i in range(ACTOR_NUM)]
    states = [Pikachus[i].reset(True) for i in range(ACTOR_NUM)]
    
    learn_step = 0
    for round in count():
        # Actor gain one experience and update state
        for i in range(ACTOR_NUM):
            states[i], epsilons[i] = actor(networks[i], target_network, P1, Pikachus[i], memory, states[i], epsilons[i])

        # If experience is enough, start learning
        if memory.size() >= BEGIN_LEARN_SIZE:
            # Learner learn one BATCH_SIZE experience and update network
            for i in range(ACTOR_NUM):
                learner(memory, networks[i], target_network, Pikachus[i], optimizers[i], losses_list[i])
        
            # Update target network
            # Choose the best (highest win rate) model as next target network
            if learn_step % UPDATA_TAGESTEP == 0:
                idx, max_winrt = 0, 0
                for i in range(ACTOR_NUM):
                    if Pikachus[i].prewinrt >= max_winrt:
                        max_winrt = Pikachus[i].prewinrt
                        idx = i
                target_network.load_state_dict(networks[idx].state_dict())

            # Learner learn once
            learn_step += 1
        
        for i in range(ACTOR_NUM):
            # If game ended, we reset the state. 
            if Pikachus[i].done:
                states[i] = Pikachus[i].reset(True)
                rounds[i] += 1

                # With enough game round, we save trained model.
                if rounds[i] % UPDATA_TAGESTEP == 0 and rounds[i] != 0 and memory.size() >= BEGIN_LEARN_SIZE:
                    if DISPLAY == "COMMANDLINE" and i == ACTOR_NUM - 1:
                        print_info(Pikachus, losses_list)
                    if DISPLAY == "PYGAME":
                        print(f'== Model {i} Data saved! ==\n')
                    torch.save(networks[i], './model/' + P2_TAG + f'_{i}.pth')
                    torch.save(memory, './memory/' + P2_TAG + '.pth')
                    torch.save({'losses': losses_list[i], 'winrts': Pikachus[i].winrts, 'epsilon': epsilons[i], 'pre_result': Pikachus[i].pre_result}, './log/' + P2_TAG + f'_{i}.pth')

def validate(P1_MODE, P2_MODE, P1_TAG, P2_TAG):
    # Create players and game
    P1 = Player(P1_MODE, False, P1_TAG)
    P2 = Player(P2_MODE, True, P2_TAG)
    Pikachu = Game("Validate", P1_MODE, P2_MODE, 2, True)

    # Keep update the scene
    for i in range(Pikachu.pre_cal_range):
        # Get initial state
        state = Pikachu.reset(True)

        while True:
            reward, state, done = Pikachu.update(P1.get_act(Pikachu.env, state), P2.get_act(Pikachu.env, state))

            if done:
                if Pikachu.is_player2_win:
                    print(f'Round {i}: P2 WIN!')
                else:
                    print(f'Round {i}: P1 WIN!')
                break

    print('==== Result ====')
    print(f'P2 Win Rate: {Pikachu.prewinrt:.2f}')