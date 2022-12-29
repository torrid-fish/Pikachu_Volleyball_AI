from gym_pikachu_volleyball.envs.constants import *
from gym_pikachu_volleyball.envs.common import convert_to_user_input
from Actions.Old_AI_Action import expectedLandingPointXWhenPowerHit
import random

def get_reward_depends_on_power_hit(P2_act, env):
        player, ball, theOtherPlayer, userInput = env.engine.players[1], env.engine.ball, env.engine.players[0], convert_to_user_input(P2_act, 1)
        y_dirs = [0, -1, 1] if random.randrange(2) else [1, 0, -1]
        reward = 0
        should_power_hit = False
        for xDirection in [1, 0]:
            for yDirection in y_dirs:
                expected_landing_point_x = expectedLandingPointXWhenPowerHit(xDirection, yDirection, ball)
                if (expected_landing_point_x <= int(player.is_player2) * GROUND_HALF_WIDTH or\
                    expected_landing_point_x >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH) and\
                    abs(expected_landing_point_x - theOtherPlayer.x) > PLAYER_LENGTH:
                        reward += int(userInput.x_direction == xDirection) +\
                                  int(userInput.y_direction == yDirection) +\
                                  int(userInput.power_hit == True)
                        should_power_hit = True
                elif (expected_landing_point_x > int(player.is_player2) * GROUND_HALF_WIDTH or\
                    expected_landing_point_x < int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH) and\
                    yDirection == 1:
                        reward -= (
                            int(userInput.x_direction == xDirection) +\
                            int(userInput.y_direction == yDirection) +\
                            int(userInput.power_hit == True)
                        ) * 5
        reward += int(userInput.power_hit == False)
        return reward / 100, should_power_hit

def get_reward_by_user_input(P2_act, env):
        player, ball, theOtherPlayer, userInput = env.engine.players[1], env.engine.ball, env.engine.players[0], convert_to_user_input(P2_act, 1)
        reward = 0 
        virtualexpected_landing_point_x: int = ball.expected_landing_point_x

        if abs(ball.x - player.x) > 100 and abs(ball.x_velocity) < 7:
            leftBoundary: int = int(player.is_player2) * GROUND_HALF_WIDTH
            if (ball.expected_landing_point_x <= leftBoundary or\
            ball.expected_landing_point_x >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH) and\
            player.computer_where_to_stand_by == 0:
                virtualexpected_landing_point_x = leftBoundary + GROUND_HALF_WIDTH // 2

        if abs(virtualexpected_landing_point_x - player.x) > 10:
            reward += int(userInput.x_direction == (1 if player.x < virtualexpected_landing_point_x else -1))
            
        if player.y > 180 and not userInput.power_hit:
            if abs(ball.x_velocity) < 5 and\
            abs(ball.x - player.x) < PLAYER_HALF_LENGTH and\
            ball.y > -36 and ball.y < 104 and ball.y_velocity > 0:
                reward += int(userInput.y_direction == -1)
            
            leftBoundary: int = int(player.is_player2) * GROUND_HALF_WIDTH
            rightBoundary: int = (int(player.is_player2) + 1) * GROUND_HALF_WIDTH
            
            if ball.expected_landing_point_x > leftBoundary and ball.expected_landing_point_x < rightBoundary and\
            abs(ball.x - player.x) > 10 + PLAYER_LENGTH and\
            ball.x > leftBoundary and ball.x < rightBoundary and ball.y > 174:
                reward += int(userInput.power_hit == 1)
                reward += int(userInput.x_direction == (1 if player.x < ball.x else -1))

        elif player.state == 1 or player.state == 2:
            if abs(ball.x - player.x) > 8:
                reward += int(userInput.x_direction == (1 if player.x < ball.x else -1))

            if abs(ball.x - player.x) < 48 and abs(ball.y - player.y) < 48:
                temp_reward, shouldInputPowerHit = get_reward_depends_on_power_hit(P2_act, env)
                reward += temp_reward
                if shouldInputPowerHit:
                    reward += int(userInput.power_hit == 1)
                    if abs(theOtherPlayer.x - player.x) < 80 and userInput.y_direction != -1:
                        reward += int(userInput.y_direction == -1)
        return reward / 100


def get_reward_by_ball_and_junp(env,prev_ball_land_x,is_fire,fire_collide_ball):
    player, ball, theOtherPlayer = env.engine.players[1], env.engine.ball, env.engine.players[0]
    is_def = 0
    reward = 0
    if(ball.expected_landing_point_x > GROUND_HALF_WIDTH):
        is_def = 1
    else:
        is_def = 0

        
    is_colliding = abs(ball.x - player.x) <= PLAYER_HALF_LENGTH and abs(ball.y - player.y) <= PLAYER_HALF_LENGTH
    
    if(is_colliding and is_fire == 1):
        fire_collide_ball = 1

    if(player.y < PLAYER_TOUCHING_GROUND_Y_COORD ):
        if(is_fire != 1):
            is_fire = 1
    else:
        if(is_fire != 0):
            if(fire_collide_ball == 0):
                reward -= 0.4
            else:
                reward += 0.4
            is_fire = 0
            fire_collide_ball = 0
    
    if(is_def):
        warning_height = NET_PILLAR_TOP_TOP_Y_COORD * 0.6
        if(ball.y < warning_height):
            if(abs(player.x-ball.expected_landing_point_x) > 50):
                reward -= ((abs(player.x-ball.expected_landing_point_x)-40)%10)*0.003
        else:
            if(abs(player.x-ball.expected_landing_point_x) > 30):
                reward -= ((abs(player.x-ball.expected_landing_point_x)-20)%10)*0.006
            if(player.y < ball.y):
                reward -= 0.03
    else:
        if(abs(theOtherPlayer.x-ball.expected_landing_point_x) > 50):
                reward += ((abs(player.x-ball.expected_landing_point_x)-40)%10)*0.015
    
    return reward

prev_ball_land_x = 0 
is_fire = 0
fire_collide_ball = 0
def calculate_reward(done: bool, is_P2_win: bool, P2_act, env):
    """
    This is the place you can adjust the reward funtion.
    """
    if done and is_P2_win:
    # P2 WIN
        reward = 1 + get_reward_by_ball_and_junp( env,prev_ball_land_x,is_fire,fire_collide_ball)
    elif done and not is_P2_win:
    # P2 LOSE
        reward = -2
    else:
    # GAME IN PROGESS
        reward = get_reward_by_ball_and_junp(env,prev_ball_land_x,is_fire,fire_collide_ball)

    return reward