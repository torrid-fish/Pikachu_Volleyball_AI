import keyboard
import numpy as np

class Player:
    """
    The controller of pikachu.
    ## parameter
    `player_list = ["Random", "Human", "apex_DQN", "rainbow_DQN"]`: The list of valid player.\n
    `player(str)`: The player we use.
    `counter(int)`: The counter used in Human (used to generate onepulse). 
    ## function
    `get_act`: Retrieve the next movement. 
    """
    def __init__(self, player: str = "Random"):
        self.counter = 0
        self.player_list = ["Random", "Human", "apex_DQN", "rainbow_DQN"]
        if player not in self.player_list:
            print("Error: Unknown player.")
        self.player = player
    
    def get_act(self):
        """
        This function will return the next movement of the player.
        ## return
        `action(int)`: The index of next action.\n
        (The index here is based on the definition in `gym_pikachu_volleyball\common.py`)
        """
        if self.player == "Random":
            # Randomly return an index
            action = np.random.choice(range(18))

        elif self.player == "Human":
            # One pulse for power_hit
            if keyboard.is_pressed('enter') and self.counter < 3:
                self.counter, power_hit = self.counter + 1, True
            elif keyboard.is_pressed('enter'):
                self.counter, power_hit = self.counter, False
            else:
                self.counter, power_hit = 0, False
            # Decode other inputs
            up = keyboard.is_pressed('up')
            down = keyboard.is_pressed('down')
            left = keyboard.is_pressed('left')
            right = keyboard.is_pressed('right')
            # Generate corresponding action index
            action = (right - left + 1) * 6 + (down - up + 1) + power_hit * 3

        elif self.player == "apex_DQN":
            pass
        elif self.player == "rainbow_DQN": 
            pass

        return action