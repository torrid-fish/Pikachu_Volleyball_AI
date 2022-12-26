from gym_pikachu_volleyball.envs.pikachu_volleyball import PikachuVolleyballMultiEnv
from gym_pikachu_volleyball.envs.engine import Ball
from gym_pikachu_volleyball.envs.constants import *
import random

INFINITE_LOOP_LIMIT = 1000

def update_expected_landing_point_x(ball: Ball):
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

def attacker_act(env: PikachuVolleyballMultiEnv):
    ball = env.engine.ball
    # The attacker, whom is the coach
    coach = env.engine.players[0]
    
    if coach.x == 36: # The initial position of P1
        # Random a new ball position that goes to other side
        while True:
            ball.x_velocity = random.randrange(0, 20)
            ball.y_velocity = random.randrange(0, 20)
            ball.x = random.randrange(40, GROUND_HALF_WIDTH - 40)
            ball.y = random.randrange(60, BALL_TOUCHING_GROUND_Y_COORD - 40)

            ball.is_power_hit = random.randrange(0, 1)
            
            update_expected_landing_point_x(ball)

            if ball.expected_landing_point_x > GROUND_HALF_WIDTH:
                break
        
        coach.x = random.randrange(0, GROUND_HALF_WIDTH)
        coach.y = random.randrange(0, PLAYER_TOUCHING_GROUND_Y_COORD)

    elif ball.expected_landing_point_x < GROUND_HALF_WIDTH:
        # If the opponient can hit to the other side, he wins.
        ball.x, ball.y = ball.expected_landing_point_x, GROUND_HEIGHT - 10


    return 7 # No control
