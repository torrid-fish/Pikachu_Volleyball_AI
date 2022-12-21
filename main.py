from game import *

if __name__ == '__main__':
    # Generate a game and run
    P1 = Player("Random")
    P2 = Player("Human")
    Pikachu = Game("Test", P1, P2)
    Pikachu.start()
