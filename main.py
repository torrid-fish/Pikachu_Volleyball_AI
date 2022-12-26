from game import Game
from player import Player
from train import train
import torch
import os
import sys

def interactive_initialization():
    os.system('cls')
    print('Would you like to use this training pattern: "P1 = Old_AI, P2 = D3QN"? (y / n)')
    SHORTCUT = input()
    if SHORTCUT == 'n':
        print('Alright, what mode do you want: (1: Play / 2: Train)')
        MODE = input()
        if MODE == '1': MODE = 'Play'
        if MODE == '2': MODE = 'Train'

        # P1
        print('Input P1 mode: (1: Human / 2: Old_AI / 3: D3QN / 4: Attacker)')
        P1_MODE = input()
        if P1_MODE == '1': P1_MODE = 'Human'
        if P1_MODE == '2': P1_MODE = 'Old_AI'
        if P1_MODE == '3': P1_MODE = 'D3QN'
        if P1_MODE == '4': P1_MODE = 'Attacker'
        if P1_MODE == "D3QN":
            print('Input the path to the model of P1: ./model/____.pth (only the underline part)')
            P1_TAG = input()
            print(f'Okay, we will use the model stored at {P1_TAG}.')
        else:
            P1_TAG = 'None'
        
        # P2
        print('Input P2 mode: (1: Human / 2: Old_AI / 3: D3QN / 4: Attacker)')
        P2_MODE = input()
        if P2_MODE == '1': P2_MODE = 'Human'
        if P2_MODE == '2': P2_MODE = 'Old_AI'
        if P2_MODE == '3': P2_MODE = 'D3QN'
        if P2_MODE == '4': P2_MODE = 'Attacker'
        if P2_MODE == "D3QN":
            print('Input the path to the model of P2: ./model/____.pth (only the underline part)')
            P2_TAG = input()
            print(f'Okay, the model of P1 will be stored at {P2_TAG}.')
        else:
            P2_TAG = 'None'
    
    elif SHORTCUT == 'y':
        MODE, P1_MODE, P2_MODE, P1_TAG = 'Train', 'Old_AI', 'D3QN', 'None'    
        print('Input the path to the model of P2: ./model/____.pth (only the underline part)')
        P2_TAG = input()
        print(f'Okay, the model of P1 will be stored at {P2_TAG}.') 
        
    # Return all infomations    
    return MODE, P1_MODE, P2_MODE, P1_TAG, P2_TAG

def play(P1_MODE, P2_MODE, P1_TAG, P2_TAG):
    # Create players and game
    P1 = Player(P1_MODE, False, P1_TAG)
    P2 = Player(P2_MODE, True, P2_TAG)
    Pikachu = Game("Play", P1_MODE, P2_MODE)

    # Get initial state
    state = Pikachu.reset(True)

    # Keep update the scene
    while True:
        reward, state, done = Pikachu.update(P1.get_act(Pikachu.env, state), P2.get_act(Pikachu.env, state))

if __name__ == '__main__':
    # Initialization
    MODE, P1_MODE, P2_MODE, P1_TAG, P2_TAG = interactive_initialization()


    if MODE == "Play":
        play(P1_MODE, P2_MODE, P1_TAG, P2_TAG)
    
    elif MODE == "Train":
        train(P1_MODE, P2_MODE, P1_TAG, P2_TAG)