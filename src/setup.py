# Setup and create game maps and character arrangments
from encoder import StaghuntEncoder
from game_configs import shum_map_A, character_setup, maps

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
