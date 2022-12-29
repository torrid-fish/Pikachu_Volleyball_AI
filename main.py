from script import play, train, validate
import os
import torch
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
You can setting you default mode here.
"""
#                  MODE   , P1_MODE , P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY , ACTOR_NUM
DEFAULT_PATTERN = ("Train", "Old_AI", "D3QN" , "None", "APEX", "Small"         , "PYGAME", 4         )

def interactive_initialization():
    game_mode = {'1': 'Play', '2': 'Train', '3': 'Validate'}
    player_mode = {'1': 'Human', '2': 'Old_AI', '3': 'D3QN', '4': 'Attacker'}
    display_mode = {'1': 'PYGAME', '2': 'COMMANDLINE'}
    screen_mode = {'1': 1.3, '2': 0.8}

    os.system('cls')

    print('=========================================================================================')
    if torch.cuda.is_available():
        print('Looks like you can use GPU to speed up! Congrats.')
    else:
        print('Looks like you can only use CPU... Does anything go wrong?')
    print('=========================================================================================')
    
    # Load default pattern
    MODE, P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY, ACTOR_NUM = DEFAULT_PATTERN
    print(f'Would you like to use this default pattern?')
    print(f'- MODE: {MODE}\n- P1_MODE: {P1_MODE}\n- P2_MODE: {P2_MODE}\n- P1_TAG: {P1_TAG}\n- P2_TAG:{P2_TAG}\n- RESOLUTION_RATIO: {RESOLUTION_RATIO}\n- DISPLAY: {DISPLAY}\n- ACTOR_NUM: {ACTOR_NUM}')
    print(f'- Network Structure:\n')
    NETWORK = f'\
                                       ⇗ (512, 256) ⇒ (512, 1) ⇘          \n\
    (1, 17) ⇒ (17, 1024) ⇒ (1024, 512)     HIDDEN_S     STATE     (1, 16) \n\
     INPUT      F1           F2        ⇘                      ⇗    OUTPUT  \n\
                                         (512, 256) ⇒ (512, 16)            \n\
                                          HIDDEN_A     ADVANTAGE             '
    print(NETWORK)
    print('\n(1: yes / 2: no)')
    DEFAULT = input()

    # Use DEFAULT to train
    if DEFAULT == '1':
        print("Okay, we will start training with DEFAULT settings.")
        if RESOLUTION_RATIO == "Medium": RESOLUTION_RATIO = 1.3
        elif RESOLUTION_RATIO == "Small": RESOLUTION_RATIO = 0.8
        return MODE, P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY, ACTOR_NUM

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

    # P2
    print('Input P2 mode: (1: Human / 2: Old_AI / 3: D3QN / 4: Attacker)')
    P2_MODE = player_mode[input()]

    if MODE == "Train":
        print('Would you like to use APEX? (1: yes, 2: no)')
        APEX = input()
    else:
        APEX = "0"

    if APEX == '1':
        print('OK, then how many player would you like to use?')
        ACTOR_NUM = int(input())
    else:
        ACTOR_NUM = 1

    if P2_MODE == "D3QN":    
        print('Input the path to the model of P2: ./model/____.pth (only the underline part)')
        P2_TAG = input()
    else:
        P2_TAG = "None"

    if MODE == "Train":
        print(f'Okay, the model of P2 will be stored at {P2_TAG}.') 
    else:
        print(f'Okay, we will use the model stored at {P2_TAG}.')

    if MODE == 'Train':
        print('By the way, would you like to use pygame to display? (1: use pygame, 2: use command line)')
        DISPLAY = display_mode[input()]
        
        if DISPLAY == 'PYGAME':
            print('Okay, which screen size do you prefer? (1: Medium / 2: Small)')
            RESOLUTION_RATIO = screen_mode[input()]
        else:
            RESOLUTION_RATIO = 0
    else:
        DISPLAY = "None"

    # Return all infomations    
    return MODE, P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY, ACTOR_NUM

if __name__ == '__main__':
    # Initialization
    MODE, P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY, ACTOR_NUM = interactive_initialization()

    if MODE == "Play":
        play(P1_MODE, P2_MODE, P1_TAG, P2_TAG)
    
    elif MODE == "Train":
        train(P1_MODE, P2_MODE, P1_TAG, P2_TAG, RESOLUTION_RATIO, DISPLAY, ACTOR_NUM)

    elif MODE == "Validate":
        validate(P1_MODE, P2_MODE, P1_TAG, P2_TAG)