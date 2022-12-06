import numpy as np
from encoder import StaghuntEncoder

MAX_GAME_LENGTH = 30

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

''' Create Setups and Maps '''
def map_init(enc=StaghuntEncoder(), map=shum_map_A, positions={}):
    map = map.copy()
    for key in positions.keys():
        x, y = positions[key]
        map[y][x] = enc.encode_id([key])
    return map

def character_setup_init(setup_name="character_setup_2h1r1s", agent=None):
    setup = character_setup[setup_name]
    if agent is not None:
        setup["h2"]["agent"] = agent
    return setup

def setup_init(setup_name="character_setup_2h1r1s", agent=None, map_name="shum_map_A"):
    setup = character_setup_init(setup_name, agent)
    map = maps[map_name]
    return setup, map
