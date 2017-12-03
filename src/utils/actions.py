"""
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


def action_number(action_a, action_b):
    a = a_actions[action_a]
    b = b_actions[action_b]
    for i, action in enumerate(atari_actions):
        if a in action and b in action:
            return i
"""
trans_a = [
    0,  #NOOP
    3,  # RIGHT - aka up
    4,  # LEFT - aka down
]

trans_a_fire = [
    1,  #NOOP
    11,  # RIGHT - aka up
    12,  # LEFT - aka down
]

trans_b = [
    20,  #NOOP
    23,  # RIGHT - aka up
    24,  # LEFT - aka down
]

# need_ball_trans = [1, # FIRE
#                   10, # UPFIRE
#                    13, # DOWNFIRE
#]


FIRE = 1
B_OFFSET = 20

def trans(act_a, act_b):
    # B's actions are offset by 20 (ALE Config)
    return trans_a_fire[act_a], trans_a_fire[act_b] + 20

def trans_single(raw_action):
    if raw_action == 0:
        return 0
    return trans_a[raw_action] 

