import gym
import time
import sys
import pygame
import numpy as np
import gym_pikachu_volleyball
from player import Player

# The FPS of the display
FPS = 120

def run_train():
    """
    This function will run the game when mode is train.
    """
    pass

def run_play():
    """
    This function will run the game when mode is play.
    """
    pass

def run_test(P1: Player, P2: Player):
    """
    This function will run the game when mode is test.
    """
    # Create the environment
    env = gym.make('PikachuVolleyball-v0', render_mode="human", new_step_api=True)
    env.metadata['render_fps'] = FPS
    # Initialize pygame
    pygame.init()
    # Create the window
    screen = pygame.display.set_mode((432, 304))
    # Set the window title
    pygame.display.set_caption("Pikachu Volleyball")
    # Reset the environment
    observation = env.reset(return_info=True, options={'is_player2_serve': False})
    # Set the font and font size for the text
    font = pygame.font.Font(None, 50)
    while True:
        action = [P1.get_act(), P2.get_act()]
        observation, reward, done, truncated, info = env.step(action)   
        # Draw the text on the screen
        done_text = font.render(f'AI', True, (255, 0, 0))
        truncated_text = font.render(f'Ur Mom', True, (255, 0, 0))
        screen.blit(done_text, (80, 40))
        screen.blit(truncated_text, (260, 40))
        # Update the window
        pygame.display.update()
        # Check if the window was closed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

class Game:
    """
    The game of pikachu volleyball.\n
    It is based on gym_pikachu_volleyball.
    ## parameter
    `mode_list = ["Train", "Play", "Test"]`: The list of valid mode.\n
    `mode(str)`: The mode we use.
    `P1`: The left player.
    `P2`: The right player.  
    ## function
    `start`: Run the game. 
    """
    def __init__(self, mode: str, Player1: Player, Player2: Player):
        self.mode_list = ["Train", "Play", "Test"]
        if mode not in self.mode_list:
            print("Error: Unknown mode is used.")
        self.mode = mode
        self.P1 = Player1
        self.P2 = Player2

    def start(self):
        """
        This function will start the game.
        """
        if self.mode == "Train":
            run_train()
        elif self.mode == "Play":
            run_play()
        elif self.mode == "Test":
            run_test(self.P1, self.P2)
            
