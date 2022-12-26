from train import train
from play import play

if __name__ == '__main__':
    # Choose which script to execute
    print('Play or Train? (P/T)')
    op = input()

    if op == 'P':
        play()
    elif op == 'T':
        train()