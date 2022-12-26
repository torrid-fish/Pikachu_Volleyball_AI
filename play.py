from game import *
from player import Player

RESOLUTION_RATIO = 2

def play():
    # Create players and game
    P1 = Player("Attacker", False, './model/D3QN_SMIV_A1.pth')
    P2 = Player("D3QN", True, './model/D3QN_SMIV_A0.pth')
    Pikachu = Game("Play", "Attacker", "D3QN", RESOLUTION_RATIO)

    screen = pygame.display.set_mode((RESOLUTION_RATIO * 432, RESOLUTION_RATIO * 304))
    pygame.display.set_caption('Pikachu_Volleyball')

    # Get init state
    state = Pikachu.reset(True)

    # Keep update the scene.
    while True:
        reward, state, done = Pikachu.update(P1.get_act(Pikachu.env, state), P2.get_act(Pikachu.env, state), screen)
        print(screen)
        # Update the window
        pygame.display.flip()
        time.sleep(1.0 / Pikachu.fps)

        # If the window was closed, end the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()