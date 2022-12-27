from train import *

def play(P1_MODE, P2_MODE, P1_TAG, P2_TAG):
    # Create players and game
    P1 = Player(P1_MODE, False, P1_TAG)
    P2 = Player(P2_MODE, True, P2_TAG)
    Pikachu = Game("Play", P1_MODE, P2_MODE, 2)

    # Get initial state
    state = Pikachu.reset(True)

    # Keep update the scene
    while True:
        reward, state, done = Pikachu.update(P1.get_act(Pikachu.env, state), P2.get_act(Pikachu.env, state))

def train(P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY):
    # Initialize Train Process
    network, target_network, P1, Pikachu, optimizer, memory, epsilon, losses = init(P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY)
    
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
                if DISPLAY == 'COMMANDLINE':
                    print_info(Pikachu, losses[-1])
                torch.cuda.empty_cache()
                break

        # With enough game round, we save trained model.
        if round % UPDATA_TAGESTEP == 0 and round != 0 and memory.size() >= BEGIN_LEARN_SIZE:
            print('== Data saved! ==\n')
            torch.save(network, './model/' + P2_TAG + '.pth')
            torch.save(memory, './memory/' + P2_TAG + '.pth')
            torch.save({'losses': losses, 'winrts': Pikachu.winrts, 'epsilon': epsilon, 'pre_result': Pikachu.pre_result}, './log/' + P2_TAG + '.pth')

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