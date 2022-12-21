import time
import sys
import pygame
import numpy as np
from player import Player
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from gym_pikachu_volleyball.envs.pikachu_volleyball import PikachuVolleyballMultiEnv
from gym_pikachu_volleyball.envs.constants import *




class Game:
    """
    The game of pikachu volleyball.\n
    It is based on gym_pikachu_volleyball.
    ## parameter
    `mode_list = ["Train", "Play"]`: The list of valid mode.\n
    `mode(str)`: The mode we use.
    `P1`: The left player.
    `P2`: The right player.  
    `settings(dict)`: Some settings about the game.
    `state(nparray?)`: The current state.
    `reward(int)`: The current reward. 
    - `reward = 0`: The ball isn't touching the ground.
    - `reward = 1`: The ball is touching the left half ground.
    - `reward = -1`: The ball is touching the right half ground.
    ## function
    `start`: Run the game. 
    ## Mode
    ### Train
    This is for train mode, you can choose whether to show those infomations.
    ### Play
    Some details will be shown, such as title, score, or something else.
    ### Test
    Currently for test mode, nothing special will be shown.
    """
    def __init__(self, mode: str, P1_mode: str, P2_mode: str, is_player2_serve: bool):
        self.mode_list = ["Train", "Play", "Test"]
        if mode not in self.mode_list:
            print(f'Error: {mode} is an unknown mode.')
            sys.exit()

        # Initialize the game
        self.mode = mode
        self.P1_mode = P1_mode
        self.P2_mode = P2_mode

        # local variables
        self.fps = 35
        self.status = "Play"
        self.counter = 0
        self.is_player1_win, self.is_player2_win = 0, 0
        self.beg_time = time.perf_counter()
        self.score, self.lose_pt = [], []
        self.P2win, self.tot = 0, 0
        self.reward = 0
        self.last_time = time.perf_counter()
        self.the_loss = 0
        

        # Create and reset the environment
        self.env = PikachuVolleyballMultiEnv(render_mode=None)
        self.state = self.env.reset(return_info=True, options={'is_player2_serve': is_player2_serve})

        # Initialize pygame
        pygame.init()
        if mode == "Play":
            self.resolution_ratio = 2 # The screen is twice larger
            self.sx, self.sy = 0, 0
            self.screen = pygame.display.set_mode((self.resolution_ratio * 432, self.resolution_ratio * 304))
        else:
            self.resolution_ratio = 1
            self.sx, self.sy = 30, 30
            self.screen = pygame.display.set_mode((1060 * self.resolution_ratio, 304 * self.resolution_ratio + 2 * 30))

        pygame.display.set_caption("Pikachu Volleyball")
    
    ## Private member ##

    def __draw_background(self):
        # Draw result depends on state
        if self.mode == "Train" or self.status == "Play" or (self.status == "Trans" and self.counter >= 10):
            # Draw background
            base_surface = pygame.surfarray.make_surface(self.state.transpose((1, 0, 2)))
            base_surface = pygame.transform.scale(base_surface, (self.resolution_ratio * 432, self.resolution_ratio * 304))
            self.screen.blit(base_surface, (self.sx, self.sy))

        elif self.status == "End" or (self.status == "Trans" and self.counter < 10):
            # Draw background in a reverse color manner
            self.state = 255 - self.state
            base_surface = pygame.surfarray.make_surface(self.state.transpose((1, 0, 2)))
            base_surface = pygame.transform.scale(base_surface, (self.resolution_ratio * 432, self.resolution_ratio * 304))
            self.screen.blit(base_surface, (self.sx, self.sy))

    def __draw_P1_win_text(self):
        # Set the font and font size for the text
        font = pygame.font.Font(None, int(30*self.resolution_ratio))
        # Draw text
        done_text = font.render(f'P1 WIN.', True, (0, 0, 0))
        self.screen.blit(done_text, (75 * self.resolution_ratio + 6 + self.sx, 80 * self.resolution_ratio + 6 + self.sy))
        done_text = font.render(f'P1 WIN.', True, (255, 255, 255))
        self.screen.blit(done_text, (75 * self.resolution_ratio + self.sx, 80 * self.resolution_ratio + self.sy))
    
    def draw_P2_win_text(self):
        # Set the font and font size for the text
        font = pygame.font.Font(None, int(30*self.resolution_ratio))
        done_text = font.render(f'P2 WIN.', True, (0, 0, 0))
        self.screen.blit(done_text, (75 * self.resolution_ratio + 6 + self.sx + 216 * self.resolution_ratio, 80 * self.resolution_ratio + 6 + self.sy))
        done_text = font.render(f'P2 WIN.', True, (255, 255, 255))
        self.screen.blit(done_text, (75 * self.resolution_ratio + self.sx + 216 * self.resolution_ratio, 80 * self.resolution_ratio + self.sy))

    def __draw_Trans(self):
        if self.status == "Trans":
            surface = pygame.Surface((self.resolution_ratio * 432, self.resolution_ratio * 304))
            if self.counter < 10:
                opacity = (self.counter / 10) * 255
            else:
                opacity = 510 - (self.counter / 10) * 255
            surface.set_alpha(opacity)
            surface.fill((0, 0, 0))
            self.screen.blit(surface, (self.sx, self.sy))

    def __draw_figure(self, fig):
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image_width, image_height = canvas.get_width_height()
        surface = pygame.image.frombuffer(buf, (image_width, image_height), 'RGBA')
        surface = pygame.transform.scale(surface, (304 * self.resolution_ratio / image_height * image_width, 304 * self.resolution_ratio / image_height * image_height))
        self.screen.blit(surface, (self.sx + 1 * (self.sx + self.resolution_ratio * 432), self.sy))

    def __draw_lose_pt(self):
        font = pygame.font.Font(None, int(30*self.resolution_ratio))
        for pt in self.lose_pt:
            text = font.render(f'X', True, (255, 0, 0))
            ptrect = text.get_rect(center=(self.sx + pt * self.resolution_ratio, 280 * self.resolution_ratio + self.sy))
            self.screen.blit(text, ptrect)

    def __draw_player(self):
        # Set the font and font size for the text
        font = pygame.font.Font(None, int(30 * self.resolution_ratio))
        text = font.render(f'{self.P1_mode}', True, (0, 0, 0))
        P1_rect = text.get_rect(center=(432 * self.resolution_ratio / 4 + self.sx, 292 * self.resolution_ratio + self.sy))
        self.screen.blit(text, P1_rect)
        text = font.render(f'{self.P2_mode}', True, (0, 0, 0))
        P2_rect = text.get_rect(center=(432 * self.resolution_ratio * 3 / 4 + self.sx, 292 * self.resolution_ratio + self.sy))
        self.screen.blit(text, P2_rect)

    def __draw_info(self):
        surface = pygame.Surface((400, self.resolution_ratio * 304))
        surface.fill((0, 0, 0))
        self.screen.blit(surface, (2 * (30 + self.resolution_ratio * 432), 30))
        font = pygame.font.Font(None, int(20 * self.resolution_ratio))
   

        message = [
        f'Speed(round/s): {self.tot / (time.perf_counter() - self.beg_time):.2f}', 
        f'Speed(frm/s): {1 / (time.perf_counter() - self.last_time):.2f}',
        f'P1 win: {self.tot - self.P2win}', 
        f'P1 win rate: {1 - self.P2win / self.tot if self.tot else 0:.2f}',
        f'P2 win: {self.P2win}',
        f'P2 win rate: {self.P2win / self.tot if self.tot else 0:.2f}',
        f'loss: {self.the_loss:.2f}']
        
        self.last_time = time.perf_counter()
        cnt = 0 
        for sentence in message:
            text = font.render(sentence, True, (255, 255, 255))
            self.screen.blit(text, (2 * (30 + self.resolution_ratio * 432), 30 + cnt * text.get_height()))
            cnt += 2     

    def __update_train(self, P1_act, P2_act):
        # Move to next state
        action = [P1_act, P2_act]
        self.state, self.reward, self.done, _, _ = self.env.step(action)   

        ### Begin: Draw infomations ###
        self.__draw_background()

        self.__draw_player()

        self.__draw_info()

        self.__draw_lose_pt()
        ### End: Draw infomations ###

        # Update the window
        pygame.display.flip()

        # If the window was closed, end the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Go to next status
        self.counter += 1
        if self.status == "Play":
            self.counter = 0
            self.is_player1_win |= self.reward == -1
            self.is_player2_win |= self.reward == 1
            if self.done:
                self.score += [(self.P2win + self.is_player2_win) / (self.tot + 1)]
                self.lose_pt += [self.env.engine.ball.x]
                self.P2win += self.is_player2_win
                self.tot += 1
                self.fps = 10
                # Draw plot
                fig = Figure()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(self.score)
                ax1.set_ylim(0, 1.1)
                ax1.set_title('P2 Win rate')
                ax1.set_xlabel('Round')
                ax1.set_ylabel('Ratio')
                self.__draw_figure(fig)
                # Reset
                self.env.reset(options={'is_player2_serve': self.is_player2_win})
                self.is_player1_win, self.is_player2_win = 0, 0

    def __update_play(self, P1_act, P2_act):
        # Move to next state
        if self.status != "Trans":
            action = [P1_act, P2_act]
        else:
            action = [7, 7] # Stand still
        self.state, self.reward, self.done, _, _ = self.env.step(action)   

        ### Begin: Draw infomations ###
        self.__draw_background()

        if self.is_player1_win == 1 and not(self.status == "Trans" and self.counter > 10):
            self.__draw_P1_win_text()

        if self.is_player2_win == 1 and not(self.status == "Trans" and self.counter > 10):
            self.draw_P2_win_text()

        self.__draw_player()

        self.__draw_Trans() # Draw after everything is drawn
        ### End: Draw infomations ###

        # Update the window
        pygame.display.flip()
        time.sleep(1.0 / self.fps)

        # If the window was closed, end the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Go to next status
        self.counter += 1
        if self.status == "Play":
            self.counter = 0
            self.is_player1_win |= self.reward == -1
            self.is_player2_win |= self.reward == 1
            if self.done:
                self.status = "End"
                self.fps = 10
                self.counter = 0
            
        elif self.status == "End":
            if self.counter == 10:
                self.status = "Trans"
                self.counter = 0
        
        elif self.status == "Trans":
            if self.counter > 10:
                self.env.reset(options={'is_player2_serve': self.is_player2_win})
            if self.counter == 20:
                self.is_player1_win, self.is_player2_win = 0, 0
                self.status = "Play"
                self.fps = 35
                self.counter = 0

    ## Public member ##

    def reset(self):
        # Return reset status
        state = self.env.reset(options={'is_player2_serve': False})
        state = (state[:, :, 0] + state[:, :, 1] + state[:, :, 2]) / 3
        return state

    def update(self, P1_act, P2_act) -> tuple[int, list]:
        if self.mode == "Train":
            self.__update_train(P1_act, P2_act)

        elif self.mode == "Play":
            self.__update_play(P1_act, P2_act)

        state = (self.state[:, :, 0] + self.state[:, :, 1] + self.state[:, :, 2]) / 3
        landing_point = self.env.engine.ball.expected_landing_point_x
        if landing_point > GROUND_HALF_WIDTH:
            if landing_point > GROUND_WIDTH:
                reward = abs(self.env.engine.players[1].x - GROUND_HALF_WIDTH) / GROUND_HALF_WIDTH
            else: 
                reward = 1 - abs(self.env.engine.players[1].x - self.env.engine.ball.expected_landing_point_x) / GROUND_HALF_WIDTH
        else:
            reward = 0
            # reward = abs(self.env.engine.players[0].x - self.env.engine.ball.expected_landing_point_x) / GROUND_HALF_WIDTH
        reward = 0 #round(reward, 2)
        return self.reward + reward / 1000, state, self.done