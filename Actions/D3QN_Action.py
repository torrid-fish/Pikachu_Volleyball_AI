from network import Dueling_D3QN
import torch

# Choose CPU or GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def d3qn_act(isPlayer2: bool, state, model: Dueling_D3QN) -> int:
    # Flip state for other side, assume we always let computer at right side (P2)
    if not isPlayer2:
        temp = state
        """
        state = np.array([
            P1.x, P1.y, P1.y_velocity, P1.state, P1.diving_direction, P1.lying_down_duration_left,
            P2.x, P2.y, P2.y_velocity, P2.state, P2.diving_direction, P2.lying_down_duration_left,
            ball.x, ball.y, ball.x_velocity, ball.y_velocity, ball.is_power_hit
        ])
        """
        # P1.x, P1.y = 1 - P2.x, P2.y
        state[0], state[1] = 1 - temp[6], temp[7]
        # P2.x, P2.y = 1 - P1.x, P1.y
        state[6], state[7] = 1 - temp[0], temp[1]
        # swap(state[2:6], state[8:12])
        state[2], state[3], state[4], state[5] = temp[8], temp[9], temp[10], temp[11]
        state[8], state[9], state[10], state[11] = temp[2], temp[3], temp[4], temp[5]
        # ball.x = 1 - ball.x
        state[12] = 1 - temp[12]
        # ball.y = ball.y
        state[13] = temp[13]
        # ball.x_velocity = -ball.x_velocity
        state[14] = -temp[14]
        # ball.y_velocity = ball.y_velocity
        state[15] = temp[15]
        # ball.is_power_hit
        state[16] = temp[16]

    state = torch.as_tensor(state, dtype=torch.float32).to(device)
    model = model.to(device)
    # Get the model
    return model.select_action(state)
