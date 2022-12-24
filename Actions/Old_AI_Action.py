from gym_pikachu_volleyball.envs.pikachu_volleyball import PikachuVolleyballMultiEnv
from gym_pikachu_volleyball.envs.engine import Ball
from gym_pikachu_volleyball.envs.constants import *
import random
import numpy as np
INFINITE_LOOP_LIMIT = 1000

def expectedLandingPointXWhenPowerHit(x_dir, y_dir, ball: Ball):
    copyBall = Ball(False)
    copyBall.x, copyBall.y, copyBall.x_velocity, copyBall.y_velocity = ball.x, ball.y, ball.x_velocity, ball.y_velocity

    if copyBall.x < GROUND_HALF_WIDTH:
        copyBall.x_velocity = (abs(x_dir) + 1) * 10
    else:
        copyBall.x_velocity = -(abs(x_dir) + 1) * 10
    
    copyBall.y_velocity = abs(copyBall.y_velocity) * y_dir * 2

    loopCounter = 0
    while True:
        loopCounter += 1

        futureballX = copyBall.x + copyBall.x_velocity
        if futureballX < BALL_RADIUS or futureballX > GROUND_WIDTH:
            copyBall.x_velocity = -copyBall.x_velocity

        if copyBall.y + copyBall.y_velocity < 0:
            copyBall.y_velocity = 1

        if abs(copyBall.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copyBall.y > NET_PILLAR_TOP_TOP_Y_COORD:
            if copyBall.y_velocity > 0:
                copyBall.y_velocity = -copyBall.y_velocity

        copyBall.y = copyBall.y + copyBall.y_velocity
        if copyBall.y > BALL_TOUCHING_GROUND_Y_COORD or loopCounter >= INFINITE_LOOP_LIMIT:
            return copyBall.x

        copyBall.x = copyBall.x + copyBall.x_velocity
        copyBall.y_velocity += 1
        
def decideWhetherInputPowerHit(x_dir, y_dir, other_player, ball):
    isPlayer2 = int(not other_player.is_player2)

    if random.randrange(32768) % 2 == 0:
        for _x_dir in range(1, -1, -1):
            for _y_dir in range(-1, 2):
                expectedLandingPointX = expectedLandingPointXWhenPowerHit(_x_dir, _y_dir, ball)
                if (expectedLandingPointX <= \
                isPlayer2 * GROUND_HALF_WIDTH or \
                expectedLandingPointX >= \
                isPlayer2 * GROUND_WIDTH + GROUND_HALF_WIDTH) and \
                abs(expectedLandingPointX - other_player.x) > PLAYER_LENGTH:
                    return _x_dir, _y_dir, 1
    else:
        for _x_dir in range(1, -1, -1):
            for _y_dir in range(1, -2, -1):
                expectedLandingPointX = expectedLandingPointXWhenPowerHit(_x_dir, _y_dir, ball)
                if (expectedLandingPointX <= \
                isPlayer2 * GROUND_HALF_WIDTH or \
                expectedLandingPointX >= \
                isPlayer2 * GROUND_WIDTH + GROUND_HALF_WIDTH) and \
                abs(expectedLandingPointX - other_player.x) > PLAYER_LENGTH:
                    return _x_dir, _y_dir, 1
    return x_dir, y_dir, 0

def old_ai_act(env: PikachuVolleyballMultiEnv, isPlayer2):
    x_dir, y_dir, power_hit = 0, 0, 0
    
    # Get env variables.
    ball = env.engine.ball
    if isPlayer2:
        other_player, AI_player = env.engine.players
    else:
        AI_player, other_player = env.engine.players
            
    # First assume it's target is the x pt of ball
    virtualexpected_landing_point_x = ball.expected_landing_point_x
    
    if abs(ball.x - AI_player.x) > 100 and abs(ball.x_velocity) < AI_player.computer_boldness + 5:
        leftBoundary = int(isPlayer2) * GROUND_HALF_WIDTH
        if (ball.expected_landing_point_x <= leftBoundary or \
        ball.expected_landing_point_x >= int(isPlayer2) * GROUND_WIDTH + GROUND_HALF_WIDTH) and \
        AI_player.computer_where_to_stand_by == 0:
            # If conditions above met, change target location to stay as the middle point of their side
            virtualexpected_landing_point_x = leftBoundary + GROUND_HALF_WIDTH // 2

    if abs(virtualexpected_landing_point_x - AI_player.x) > AI_player.computer_boldness + 8: 
        if AI_player.x < virtualexpected_landing_point_x:
            x_dir = 1
        else:
            x_dir = -1
    elif random.randrange(32768) % 20 == 0:
        AI_player.computer_where_to_stand_by = random.randrange(32768) % 2
    
    if AI_player.state == 0:
        if abs(ball.x_velocity) < AI_player.computer_boldness + 3 and abs(ball.x - AI_player.x) < PLAYER_HALF_LENGTH and \
        ball.y > -36 and ball.y < 10 * AI_player.computer_boldness + 84 and ball.y_velocity > 0:
            y_dir = -1

        leftBoundary = int(isPlayer2) * GROUND_HALF_WIDTH
        rightBoundary = (int(isPlayer2) + 1) * GROUND_HALF_WIDTH

        if ball.expected_landing_point_x > leftBoundary and ball.expected_landing_point_x < rightBoundary and \
        abs(ball.x - AI_player.x) > AI_player.computer_boldness * 5 + PLAYER_LENGTH and\
        ball.x > leftBoundary and ball.x < rightBoundary and ball.y > 174:
            # If conditions above met, the computer decides to dive!
            power_hit = 1

            if AI_player.x < ball.x:
                x_dir = 1
            else:
                x_dir = -1
        
    elif AI_player.state == 1 or AI_player.state == 2:
        if abs(ball.x - AI_player.x) > 8:
            if AI_player.x < ball.x:
                x_dir = 1
            else :
                x_dir = -1

        if abs(ball.x - AI_player.x) < 48 and abs(ball.y - AI_player.y) < 48:
            x_dir, y_dir, power_hit = decideWhetherInputPowerHit(x_dir, y_dir, other_player, ball)
            if power_hit:
                if abs(other_player.x - AI_player.x) < 80 and y_dir != -1:
                    y_dir = -1

    # Decide whether to reverse direction
    if not isPlayer2:
        x_dir = -x_dir

    return (x_dir + 1) * 6 + (y_dir + 1) + power_hit * 3 