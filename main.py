from train import train
from play import play
import os

if __name__ == '__main__':
    # Choose which script to execute
    os.system('cls')
    print('Play or Train? (P/T)')
    op = input()

    if op == 'P':
        play()
    elif op == 'T':
        train()