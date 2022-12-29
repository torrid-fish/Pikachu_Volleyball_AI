import time
import sys
import pygame
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from gym_pikachu_volleyball import *
from gym_pikachu_volleyball.envs.common import *
from gym_pikachu_volleyball.envs import PikachuVolleyballMultiEnv
from gym_pikachu_volleyball.envs.constants import *
from gym_pikachu_volleyball.envs.engine import Ball
from Actions.Old_AI_Action import INFINITE_LOOP_LIMIT
from reward import calculate_reward

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
    def __init__(self, mode: str, P1_mode: str, P2_mode: str, resolution_ratio: float, display: bool, SCREEN_SIZE: tuple = None, POS: tuple = None):
        # Sanity check
        self.mode_list = ["Train", "Play", "Validate"]
        if mode not in self.mode_list:
            print(f'Error: {mode} is an unknown mode.')
            sys.exit()

        # Initialize the game
        self.mode = mode
        self.P1_mode = P1_mode
        self.P2_mode = P2_mode

        # local variables
        self.peak_fps = 120
        self.fps = self.peak_fps    
        self.status = "Play"
        self.counter = 0
        self.is_player1_win, self.is_player2_win = 0, 0
        self.beg_time = time.perf_counter()
        self.winrts, self.lose_pt = [], []
        self.P2win, self.tot = 0, 0
        self.last_time = time.perf_counter()
        self.loss = 0
        self.losses = [0]
        self.reward = 0
        self.adjusted_reward = 0
        self.pre_cal_range = 100
        self.pre_result = []
        self.prewinrt = 0
        self.epsilon = 0
        self.is_random = False

        # Create and reset the environment
        self.env = PikachuVolleyballMultiEnv(render_mode=None)
        # The secene of the back ground
        self.scene = self.env.reset(return_info=True, options={'is_player2_serve': True})

        self.display = display
        if display:
            # Initialize pygame
            pygame.init()
            if mode == "Play":
                self.resolution_ratio = resolution_ratio # The screen is twice larger
                self.sx, self.sy = 0, 0
                self.screen = pygame.display.set_mode((self.resolution_ratio * 432, self.resolution_ratio * 304))
            elif mode == "Train":
                self.resolution_ratio = resolution_ratio
                WIDTH = 1060 * self.resolution_ratio
                HEIGHT = 304 * self.resolution_ratio
                self.sx, self.sy = (POS[0] + 1) * 30 + POS[0] * WIDTH, (POS[1] + 1) * 30 + POS[1] * HEIGHT
                self.screen = pygame.display.set_mode(SCREEN_SIZE)
            elif mode == "Validate":
                self.resolution_ratio = resolution_ratio
                self.sx, self.sy = 0, 0
                self.screen = pygame.display.set_mode((self.resolution_ratio * 432, self.resolution_ratio * 304))

            pygame.display.set_caption("Pikachu Volleyball")
 
    ## Private member ##

    def __draw_background(self):
        # Draw result depends on state
        if self.mode == "Train" or self.status == "Play" or (self.status == "Trans" and self.counter >= 10):
            # Draw background
            base_surface = pygame.surfarray.make_surface(self.scene.transpose((1, 0, 2)))
            base_surface = pygame.transform.scale(base_surface, (self.resolution_ratio * 432, self.resolution_ratio * 304))
            self.screen.blit(base_surface, (self.sx, self.sy))

        elif self.status == "End" or (self.status == "Trans" and self.counter < 10):
            # Draw background in a reverse color manner
            self.scene = 255 - self.scene
            base_surface = pygame.surfarray.make_surface(self.scene.transpose((1, 0, 2)))
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
    
    def __draw_P2_win_text(self):
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
        self.screen.blit(surface, (self.sx + self.resolution_ratio * 432 + 30, self.sy))

    def __draw_lose_pt(self):
        font = pygame.font.Font(None, int(30*self.resolution_ratio))
        for pt in self.lose_pt:
            text = font.render(f'X', True, (255, 0, 0))
            ptrect = text.get_rect(center=(self.sx + pt * self.resolution_ratio, 280 * self.resolution_ratio + self.sy))
            self.screen.blit(text, ptrect)

    def __draw_fall_pt(self):
        font = pygame.font.Font(None, int(30*self.resolution_ratio))
        fallx = self.env.engine.ball.expected_landing_point_x

        text = font.render(f'V', True, (0, 0, 0))
        ptrect = text.get_rect(center=(self.sx + fallx * self.resolution_ratio, 280 * self.resolution_ratio + self.sy))
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
        surface = pygame.Surface((self.resolution_ratio * 190, self.resolution_ratio * 304))
        surface.fill((0, 0, 0))
        self.screen.blit(surface, (self.sx + (30 + 2 * self.resolution_ratio * 432), self.sy))
        font = pygame.font.Font(None, int(20 * self.resolution_ratio))

        curtime = time.gmtime(time.perf_counter() - self.beg_time)

        message = [
        f'Speed(round/s): {self.tot / (time.perf_counter() - self.beg_time):.2f}', 
        f'Speed(frm/s): {self.speed:.2f}',
        f'== P2 info ==',
        f'win rate: {self.prewinrt:.2f}',
        f'== Train info ==',
        f'round: {self.tot}',
        f'time: {curtime.tm_hour:02d}:{curtime.tm_min:02d}:{curtime.tm_sec:02d}',
        f'loss: {self.loss:.6f}',
        f'reward: {self.adjusted_reward:.6f}',
        f'epsilon: {self.epsilon:.6f}'
        ]
        
        cnt = 0 
        for sentence in message:
            text = font.render(sentence, True, (255, 255, 255))
            if cnt == 0: h = text.get_height()
            self.screen.blit(text, (self.sx + (30 + 2 * self.resolution_ratio * 432), self.sy + cnt * h))
            cnt += 2    

    def __draw_control(self, drawP1: bool, P1_act: int, drawP2: bool, P2_act: int):
        # Set the font and font size for the text
        font = pygame.font.Font(None, int(30 * self.resolution_ratio))
        color = [(220, 220, 220), (255, 0, 0)]
        span = 50 * self.resolution_ratio / 1.3
        P1 = (
            (1, 0, 1, 0, 0),
            (1, 0, 0, 0, 0),
            (1, 0, 0, 1, 0),
            (1, 0, 1, 0, 1),
            (1, 0, 0, 0, 1),
            (1, 0, 0, 1, 1),
            (0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0),
            (0, 0, 1, 0, 1),
            (0, 0, 0, 0, 1),
            (0, 0, 0, 1, 1),
            (0, 1, 1, 0, 0),
            (0, 1, 0, 0, 0),
            (0, 1, 0, 1, 0),
            (0, 1, 1, 0, 1),
            (0, 1, 0, 0, 1),
            (0, 1, 0, 1, 1)
        )
        P2 = (
            (0, 1, 1, 0, 0),
            (0, 1, 0, 0, 0),
            (0, 1, 0, 1, 0),
            (0, 1, 1, 0, 1),
            (0, 1, 0, 0, 1),
            (0, 1, 0, 1, 1),
            (0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0),
            (0, 0, 1, 0, 1),
            (0, 0, 0, 0, 1),
            (0, 0, 0, 1, 1),
            (1, 0, 1, 0, 0),
            (1, 0, 0, 0, 0),
            (1, 0, 0, 1, 0),
            (1, 0, 1, 0, 1),
            (1, 0, 0, 0, 1),
            (1, 0, 0, 1, 1)
        )
        if drawP1:
            isright, isleft, isup, isdown, ispower = P1[P1_act]
            power = font.render('P', True, color[ispower])
            up = font.render('^', True, color[isup])
            down = font.render('v', True, color[isdown])
            right = font.render('>', True, color[isright])
            left = font.render('<', True, color[isleft])
            mid = up.get_rect(center=(432 * self.resolution_ratio / 4 + self.sx, 292 * self.resolution_ratio / 8 + self.sy))
            self.screen.blit(up, mid)
            self.screen.blit(power, (mid[0] - span, mid[1]       ))
            self.screen.blit(down,  (mid[0]       , mid[1] + span))
            self.screen.blit(right, (mid[0] + span, mid[1] + span))
            self.screen.blit(left,  (mid[0] - span, mid[1] + span))
        if drawP2:
            isright, isleft, isup, isdown, ispower = P2[P2_act]
            power = font.render('P', True, color[ispower])
            up = font.render('^', True, color[isup])
            down = font.render('v', True, color[isdown])
            right = font.render('>', True, color[isright])
            left = font.render('<', True, color[isleft])
            mid = up.get_rect(center=(432 * self.resolution_ratio * 3 / 4 + self.sx, 292 * self.resolution_ratio / 8 + self.sy))
            self.screen.blit(up, mid)
            self.screen.blit(power, (mid[0] - span, mid[1]       ))
            self.screen.blit(down,  (mid[0]       , mid[1] + span))
            self.screen.blit(right, (mid[0] + span, mid[1] + span))
            self.screen.blit(left,  (mid[0] - span, mid[1] + span))
            if self.is_random:
                rand = font.render('R', True, (0, 0, 0))
                self.screen.blit(rand,  (mid[0] + span, mid[1]))

    def __update_winrt(self):
        # Calculate average win rate
        if self.tot:
            avgwinrt = (self.P2win + self.is_player2_win) / (self.tot + 1)
        else:
            avgwinrt = 0
                
        # Calculate previous win rate
        if self.tot < self.pre_cal_range:
            self.prewinrt = avgwinrt
        else:
            self.prewinrt = np.sum(self.pre_result[-self.pre_cal_range:]) / self.pre_cal_range

        # Update win rate list (for plotting)
        if self.tot < self.pre_cal_range:
            self.winrts += [avgwinrt]
        else:
            self.winrts += [self.prewinrt]

        # Update other elements
        self.P2win += self.is_player2_win
        self.pre_result += [self.is_player2_win]

    def __update_train(self, P1_act, P2_act):
        # Move to next state
        action = [P1_act, P2_act]
        self.scene, self.reward, self.done, _, _ = self.env.step(action)   

        ### Begin: Draw infomations ###
        if self.display:
            self.__draw_background()

            self.__draw_player()

            self.__draw_info()

            self.__draw_lose_pt()

            self.__draw_fall_pt()

            self.__draw_control(True, P1_act, True, P2_act)
        ### End: Draw infomations ###

        # Update the window
        if self.display:
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
                self.__update_winrt()
                self.tot += 1
                self.lose_pt += [self.env.engine.ball.x]
                if self.display:
                    # Draw plot
                    fig = Figure()
                    # Plot P2 win rate
                    ax1 = fig.add_subplot(2, 1, 1)
                    ax1.plot(self.winrts)
                    ax1.set_ylim(0, 1.1)
                    ax1.set_title('P2 Win rate')
                    ax1.set_xlabel('Round')
                    ax2 = fig.add_subplot(2, 1, 2)
                    ax2.plot(self.losses)
                    ax2.set_title('Loss')
                    ax2.set_xlabel('Frame')
                    # Set padding
                    fig.tight_layout(pad=0.5)
                    self.__draw_figure(fig)

    def __update_play(self, P1_act, P2_act):
        # Move to next state
        if self.status != "Trans":
            action = [P1_act, P2_act]
        else:
            action = [7, 7] # Stand still
        self.scene, self.reward, self.done, _, _ = self.env.step(action)   

        ### Begin: Draw infomations ###
        self.__draw_background()

        if self.is_player1_win == 1 and not(self.status == "Trans" and self.counter > 10):
            self.__draw_P1_win_text()

        if self.is_player2_win == 1 and not(self.status == "Trans" and self.counter > 10):
            self.__draw_P2_win_text()

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
                self.reset(False)
            if self.counter == 20:
                self.is_player1_win, self.is_player2_win = 0, 0
                self.status = "Play"
                self.fps = self.peak_fps
                self.counter = 0

    def __update_validate(self, P1_act, P2_act):
        # Move to next state
        action = [P1_act, P2_act]
        self.scene, self.reward, self.done, _, _ = self.env.step(action)   

        ### Begin: Draw infomations ###
        self.__draw_background()

        self.__draw_player()

        self.__draw_lose_pt()

        self.__draw_fall_pt()

        self.__draw_control(True, P1_act, True, P2_act)
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
                self.__update_winrt()
                self.tot += 1
                self.lose_pt += [self.env.engine.ball.x]                

    def __update_expected_landing_point_x(self, ball: Ball):
        """
        Calculate the actual landing point of current ball x
        """
        copyBall = Ball(False)
        copyBall.x, copyBall.y, copyBall.x_velocity, copyBall.y_velocity = ball.x, ball.y, ball.x_velocity, ball.y_velocity

        loopCounter = 0

        while True:
            loopCounter += 1

            futureCopyBallX = copyBall.x_velocity + copyBall.x
            # Reflection happens when ball collide with left, right wall
            if futureCopyBallX < BALL_RADIUS or futureCopyBallX > GROUND_WIDTH:
                copyBall.x_velocity = -copyBall.x_velocity
            # Reflection happens when ball collide with ceiling
            if copyBall.y + copyBall.y_velocity < 0:
                copyBall.y_velocity = 1

            # If copy ball touches net
            if abs(copyBall.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copyBall.y > NET_PILLAR_TOP_TOP_Y_COORD:
            # It maybe should be <= NET_PILLAR_TOP_BOTTOM_Y_COORD as in FUN_00402dc0, is it the original game author's mistake?
                if copyBall.y < NET_PILLAR_TOP_BOTTOM_Y_COORD:
                    if copyBall.y_velocity > 0:
                        copyBall.y_velocity = -copyBall.y_velocity
                
                else :
                    if (copyBall.x < GROUND_HALF_WIDTH) :
                        copyBall.x_velocity = -abs(copyBall.x_velocity)
                    else :
                        copyBall.x_velocity = abs(copyBall.x_velocity)

            copyBall.y = copyBall.y + copyBall.y_velocity
            # if copyBall would touch ground
            if copyBall.y > BALL_TOUCHING_GROUND_Y_COORD or loopCounter >= INFINITE_LOOP_LIMIT:
                break
            
            copyBall.x = copyBall.x + copyBall.x_velocity
            copyBall.y_velocity += 1
        
        ball.expected_landing_point_x = copyBall.x

    def __cal_state(self):
        # Generate info vector
        P1 = self.env.engine.players[0]
        P2 = self.env.engine.players[1]
        ball = self.env.engine.ball
        state = np.array([
            P1.x, P1.y, P1.y_velocity, P1.state, P1.diving_direction, P1.lying_down_duration_left,
            P2.x, P2.y, P2.y_velocity, P2.state, P2.diving_direction, P2.lying_down_duration_left,
            ball.x, ball.y, ball.x_velocity, ball.y_velocity, ball.is_power_hit
        ]).astype(float)
        # Normalize data
        state[0] = state[0] / GROUND_WIDTH # P1.x
        state[1] = state[1] / GROUND_HEIGHT # P1.y
        state[2] = (state[2] + 20) / 40 # P1.y_velocity
        state[3] = state[3] / 10 # P1.state
        state[4] = (state[4] + 1) / 2 # P1.diving_direction
        state[5] = (state[5] + 2) / 4 # P1.lying_down_duration_left
        state[6] = state[6] / GROUND_WIDTH # P2.x
        state[7] = state[7] / GROUND_HEIGHT # P2.y
        state[8] = (state[8] + 20) / 40 # P2.y_velocity
        state[9] = state[9] / 10 # P2.state
        state[10] = (state[10] + 1) / 2 # P2.diving_direction
        state[11] = (state[11] + 2) / 4 # P2.lying_down_duration_left
        state[12] = state[12] / GROUND_WIDTH # ball.x
        state[13] = state[13] / GROUND_HEIGHT # ball.y
        state[14] = (state[14] + 40) / 80 # ball.x_velocity
        state[15] = (state[15] + 40) / 80 # ball.y_velocity
        state[16] = int(state[16]) # ball.is_power_hit

        return state

    ## Public member ##

    def reset(self, reset: bool):
        """
        This function will return the initial state.
        """
        # Reset and cal state
        self.scene = self.env.reset(options={'is_player2_serve': self.is_player2_win}) # self.tot % 2
        state = self.__cal_state()
        # Reset who win
        if reset:
            self.is_player1_win, self.is_player2_win = 0, 0
        return state

    def update(self, P1_act, P2_act):
        """
        This function will return `reward, next_state, done`.
        """
        # Update landing point
        self.__update_expected_landing_point_x(self.env.engine.ball)
        self.speed = 1 / (time.perf_counter() - self.last_time)

        if self.mode == "Train":
            self.__update_train(P1_act, P2_act)

        elif self.mode == "Play":
            self.__update_play(P1_act, P2_act)
        
        elif self.mode == "Validate":
            self.__update_validate(P1_act, P2_act)

        # Add small rewards
        self.adjusted_reward = calculate_reward(self.done, self.is_player2_win, P2_act, self.env)
        self.losses += [self.loss]
        self.last_time = time.perf_counter()

        return self.adjusted_reward, self.__cal_state(), self.done