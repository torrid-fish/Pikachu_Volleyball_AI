import numpy as np
import sys
from Actions.Old_AI_Action import old_ai_act
from Actions.Human_Action import human_act
from Actions.D3QN_Action import *
from gym_pikachu_volleyball.envs import PikachuVolleyballMultiEnv

class Player:
    """
    The controller of pikachu.
    ## parameter
    `player_list = ["Random", "Human", "apex_D3QN", "D3QN", "Old_AI"]`: The list of valid AI_player.\n
    `mode(str)`: The player mode we use.
    ## function
    `get_act`: Retrieve the next movement. 
    """
    def __init__(self, player: str = "Random", isPlayer2: bool = None):
        self.mode_list = ["Random", "Human", "apex_D3QN", "D3QN", "Old_AI"]
        if player not in self.mode_list:
            print(f'Error: :player is an unknown AI_player.')
            sys.exit()
        self.mode = player
        self.isPlayer2 = isPlayer2
        
        if player == "D3QN":
            # Import used model
            PATH = './model/networkdueling.pt' # The path of stored model
            self.model = Dueling_D3QN((304, 432), 18)
            try:
                self.model.load_state_dict(torch.load(PATH))
                print("Model loaded sucessfully.")
            except FileNotFoundError:
                self.model = Dueling_D3QN((304, 432), 18)
                print("Create a new model.")
            # Set to evaluate mode
            self.model.eval()

    ## Public member ##

    def get_act(self, env: PikachuVolleyballMultiEnv = None, state: np.ndarray = None):
        """
        This function will return the next movement of the player.
        ## return
        `action(int)`: The index of next action.\n
        (The index here is based on the definition in `gym_pikachu_volleyball\common.py`)
        """
        if self.mode == "Random":
            action = np.random.choice(range(18))

        elif self.mode == "Human":
            action = human_act(self.isPlayer2)

        elif self.mode == "apex_D3QN":
            action = 0

        elif self.mode == "D3QN": 
            action = d3qn_act(self.isPlayer2, state, self.model)

        elif self.mode == "Old_AI":
            action = old_ai_act(env, self.isPlayer2)

        return action