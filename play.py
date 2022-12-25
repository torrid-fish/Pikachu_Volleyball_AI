from game import *
from player import Player

def play():
    # Create players and game
    P1 = Player("Attacker", False)
    P2 = Player("D3QN", True)
    Pikachu = Game("Play", "Attacker", "D3QN")

    # Get initial state
    state = Pikachu.reset(True)

    # Keep update the scene.
    while True:
        reward, state, done = Pikachu.update(P1.get_act(Pikachu.env, state), P2.get_act(Pikachu.env, state))