atari_actions = [
    "PLAYER_A_NOOP"
    ,"PLAYER_A_FIRE"
    ,"PLAYER_A_UP"
    ,"PLAYER_A_RIGHT"
    ,"PLAYER_A_LEFT"
    ,"PLAYER_A_DOWN"
    ,"PLAYER_A_UPRIGHT"
    ,"PLAYER_A_UPLEFT"
    ,"PLAYER_A_DOWNRIGHT"
    ,"PLAYER_A_DOWNLEFT"
]

a_actions = [
    "",
    "UP",
    "DOWN"
]

b_actions = [
    "",
    "LEFT",
    "RIGHT"
]

FIRE = 1

def action_number(action_a, action_b):
    a = a_actions[action_a]
    b = b_actions[action_b]
    for i, action in enumerate(atari_actions):
        if a in action and b in action:
            return i

