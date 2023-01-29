import numpy as np

STEP_COST = -0.5
STAG_VALUE = 50
RABBIT_VALUE = 5
MAX_GAME_LENGTH = 30

'''
Terminal Printing Highlighting Guide
Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
Style: DIM, NORMAL, BRIGHT, RESET_ALL
'''
CLEAR_SCREEN = '\033[2J'
RED = '\033[31m'   # mode 31 = red forground
RESET = '\033[0m'  # mode 0  = reset

''' Character Configurations '''
character_setup_full = {
    "r1": (1, 1),
    "r2": (1, 1),
    "s1": (1, 1),
    "s2": (1, 1),
    "h1": (1, 1),
    "h2": (1, 1),
    "h3": (1, 1)
}

character_setup_2h1r1s = {
    "r1": {"position": (1, 1)},
    "s1": {"position": (1, 1)},
    "h1": {"position": (2, 2)},
    "h2": {"position": (2, 2)}
}

character_setup_1h1s1r = {
    "r1": {"position": (1, 1)},
    "s1": {"position": (2, 2)},
    "h1": {"position": (2, 2)}
}

character_setup_simple = {
    "r1": {"position": (1, 1)},
    "h1": {"position": (2, 2)}
}

character_setup = {
    "character_setup_simple" : character_setup_simple,
    "character_setup_1h1s1r" : character_setup_1h1s1r,
    "character_setup_2h1r1s" : character_setup_2h1r1s,
    "character_setup_full" : character_setup_full
}

''' Maps '''
simple_map = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0]])

custom_map_square = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 1, 1, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 1, 1, 1, 0, 1, 1, 0],
                              [0, 0, 1, 0, 1, 0, 1, 0, 0],
                              [0, 0, 1, 1, 1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0]])

shum_map_A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 1, 0, 1, 1, 0],
                       [0, 0, 1, 0, 1, 0, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])

shum_map_D = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 0, 1, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])

shum_map_E = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 0, 1, 0, 0],
                       [0, 0, 1, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 0, 1, 0, 1, 0, 0],
                       [0, 0, 1, 0, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])

shum_map_F = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0, 1, 0, 0],
                       [0, 1, 1, 1, 1, 0, 1, 1, 0],
                       [0, 0, 1, 0, 1, 0, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])

shum_map_G = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])

shum_map_I = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])

maps = {
    "simple_map" : simple_map,
    "shum_map_A" : shum_map_A,
    "shum_map_D" : shum_map_D,
    "shum_map_E" : shum_map_E,
    "shum_map_F" : shum_map_F,
    "shum_map_G" : shum_map_G,
    "shum_map_I" : shum_map_I
}
