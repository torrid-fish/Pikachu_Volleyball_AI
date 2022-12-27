from script import play, train, validate
import os
import torch
import time

"""
You can setting you default mode here.
"""
DEFAULT_PATTERN = ("Train", "Old_AI", "D3QN", "None", "None", 1.3, "PYGAME")

def interactive_initialization():
    game_mode = {'1': 'Play', '2': 'Train', '3': 'Validate'}
    player_mode = {'1': 'Human', '2': 'Old_AI', '3': 'D3QN', '4': 'Attacker'}
    display_mode = {'1': 'PYGAME', '2': 'COMMANDLINE'}
    screen_mode = {'1': 1.3, '2': 0.8}

    if torch.cuda.is_available():
        print('Looks like you can use GPU to speed up! Congrats.')
    else:
        print('Looks like you can only use CPU... Does anything go wrong?')
    time.sleep(2)    

    os.system('cls')
    # Load default pattern
    MODE, P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY = DEFAULT_PATTERN
    print(f'Would you like to use this default pattern?')
    print(f'- MODE: {MODE}\n- P1_MODE: {P1_MODE}\n- P2_MODE: {P2_MODE}\n- P1_TAG: {P1_TAG}\n- P2_TAG: ____\n- RESOLUTION_RATIO: {RESOLUTION_RATIO}\n- DISPLAY: {DISPLAY}\n')
    print(f'- Network Structure:\n')
    NETWORK = f'\
    (1, 17) ⇒ (17, 1024) ⇒ (1024, 512) ⇒ (512, 256) ⇒ (512, 1)   ⇒ (1, 16) \n\
     INPUT      F1           F2        ⇘  HIDDEN_S     STATE    ⇗  OUTPUT    \n\
                                         (512, 256) ⇒ (512, 16)              \n\
                                          HIDDEN_A     ADVANTAGE               '
    print(NETWORK)
    print('\n(1: yes / 2: no)')
    DEFAULT = input()

    # Don't use default pattern.
    if DEFAULT == '2':
        print('Alright, what mode do you want: (1: Play / 2: Train / 3: Validate)')
        MODE = game_mode[input()]

        # P1
        print('Input P1 mode: (1: Human / 2: Old_AI / 3: D3QN / 4: Attacker)')
        P1_MODE = player_mode[input()]

        if P1_MODE == "D3QN":
            print('Input the path to the model of P1: ./model/____.pth (only the underline part)')
            P1_TAG = input()
            print(f'Okay, we will use the model stored at {P1_TAG}.')
        else:
            P1_TAG = 'None'

        print('Input P2 mode: (1: Human / 2: Old_AI / 3: D3QN / 4: Attacker)')
        P2_MODE = player_mode[input()]
        
    print('Input the path to the model of P2: ./model/____.pth (only the underline part)')
    P2_TAG = input()
    print(f'Okay, the model of P2 will be stored at {P2_TAG}.') 

    if MODE == 'Train' and DEFAULT == '2':
        print('By the way, would you like to use pygame to display? (1: use pygame, 2: use command line)')
        DISPLAY = display_mode[input()]
        
        if DISPLAY == 'PYGAME':
            print('Okay, which screen size do you prefer? (1: Medium / 2: Small)')
            RESOLUTION_RATIO = screen_mode[input()]
        
    # Return all infomations    
    return MODE, P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY

if __name__ == '__main__':
    # Initialization
    MODE, P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY = interactive_initialization()

    if MODE == "Play":
        play(P1_MODE, P2_MODE, P1_TAG, P2_TAG)
    
    elif MODE == "Train":
        train(P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY)

    elif MODE == "Validate":
        validate(P1_MODE, P2_MODE, P1_TAG, P2_TAG)