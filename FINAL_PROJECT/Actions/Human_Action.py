import keyboard

counter = 0

def human_act(isPlayer2: bool):
    global counter
    if isPlayer2:
        # One pulse for power_hit
        if keyboard.is_pressed('enter') and counter < 3:
            counter, power_hit = counter + 1, True
        elif keyboard.is_pressed('enter'):
            counter, power_hit = counter, False
        else:
            counter, power_hit = 0, False
        # Decode other inputs
        up = keyboard.is_pressed('up')
        down = keyboard.is_pressed('down')
        left = keyboard.is_pressed('left')
        right = keyboard.is_pressed('right')
    else:   
        # One pulse for power_hit
        if keyboard.is_pressed('Z') and counter < 3:
            counter, power_hit = counter + 1, True
        elif keyboard.is_pressed('Z'):
            counter, power_hit = counter, False
        else:
            counter, power_hit = 0, False
        # Decode other inputs
        up = keyboard.is_pressed('R')
        down = keyboard.is_pressed('F')
        right = keyboard.is_pressed('D') # Switched
        left = keyboard.is_pressed('G') # Switched

    # Generate corresponding action index
    return (right - left + 1) * 6 + (down - up + 1) + power_hit * 3
