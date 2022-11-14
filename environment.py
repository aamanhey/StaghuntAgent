import cv2
import sys
import math
import random
import pickle
import colorama
import numpy as np
import beepy as beep
import matplotlib.pyplot as plt

from state import State
from os.path import exists
from colorama import Fore, Back, Style
from IPython.display import clear_output

# Staghunt Libraries
from registry import Registry
from encoder import StaghuntEncoder
from agents import BasicHunterAgent, StaghuntAgent
from interaction_manager import InteractionManager, RABBIT_VALUE, STAG_VALUE

'''
Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
Style: DIM, NORMAL, BRIGHT, RESET_ALL
'''

colorama.init()
CLEAR_SCREEN = '\033[2J'
RED = '\033[31m'   # mode 31 = red forground
RESET = '\033[0m'  # mode 0  = reset

class StaghuntEnv():
    def __init__(self, map_dim=7, game_length=30, characters={}, map=None):
        # Setup/Meta-data
        self.NUM_ACTIONS = 4
        self.MAP_DIMENSION = map_dim
        self.MAX_STEP = game_length
        self.current_episode = 0
        self.encoder = StaghuntEncoder()
        self.play_status = False
        random.seed(42)

        # Create map for the game
        if map is not None:
            self.map = map.copy()
            self.x_bounds = [1, len(map[0])-2] # inclusive, -2 for walls and 0-index
            self.y_bounds = [1, len(map)-2]
            self.base_map = self.map.copy()
        else:
            # Create a nxn matrix to represent the world
            n = self.MAP_DIMENSION

            self.x_bounds = [1, map_dim-2] # inclusive, -2 for walls and 0-index
            self.y_bounds = self.x_bounds

            self.map = self.create_map(n, c=len(characters))
            self.base_map = self.map.copy()

        self.subject = ""

        # Create a registry for all the characters in the game associated with a position and an agent
        if len(characters) > 0:
            for character in characters:
                if "agent" not in characters[character].keys():
                    type = character[0]
                    if type == "r":
                        characters[character]["agent"] = StaghuntAgent(character, type, kind="static-prey")
                    elif type == "s":
                        # @TODO: Change kind to mobile
                        characters[character]["agent"] = StaghuntAgent(character, type, kind="static-prey")
                    else:
                        characters[character]["agent"] = BasicHunterAgent(character)
            self.c_reg = Registry(characters)
            self.subject = self.set_random_subject() # the subject is going to act as a main player
        elif len(characters > 8):
            print("E: Environment cannot handle more than 8 characters.")
        else:
            print("E: No characters give, initializing Basic Hunter.")
            basic_hunter = {"agent": BasicHunterAgent("h1"), "position": (0, 0)}
            self.c_reg = Registry({"h1": basic_hunter})
            self.subject = "h1"

        # Declare an interaction manager to handle logic of when characters interact
        self.i_manager = InteractionManager(self.c_reg.get_characters())

        # Display
        self.window_shape = (500, 500, 3)
        self.canvas = np.ones(self.window_shape) * 1

    ''' Environment Methods '''
    def reset(self):
        self.play_status = True
        self.current_step = 0

        # Reset map
        self.map = self.base_map.copy()

        # Move characters to random positions
        self.set_random_characters()
        self.c_reg.reset_rewards()

        # Put elements on the canvas
        self.create_canvas()

        self.state = self.encoder.encode(self.map)

        return self.state

    def step(self):
        # Call agents to get moves for each character
        if not self.play_status:
            return self.state

        map = self.map
        state_id = self.encoder.encode(map)

        new_positions = {}
        actions = {}

        reg = self.c_reg
        characters = reg.get_characters()
        positions = reg.get_positions()

        state = State(state_id, map, positions, self.current_step, positions[self.subject])

        for id in reg.get_ids():
            # Get move from agent and save the move and position
            agent = reg.get_agent(id)
            action = agent.get_move(state)
            actions[id] = action

            if action in new_positions.keys():
                new_positions[action].append(id)
            else:
                new_positions[action] = [id]

        # Update map and character registry
        self.update_map(new_positions)
        new_reg = self.c_reg
        self.i_manager.set_reg(new_reg.get_characters())

        next_state_id = self.encoder.encode(self.map)
        self.state = next_state_id

        # End game when game is over
        self.play_status = self.check_game_rules() # Return False if game is over

        # Give rewards to and update each agent
        points = self.get_rewards()

        next_state = State(next_state_id, self.map, new_reg.get_positions(), self.current_step + 1, new_reg.get_position(self.subject))

        for id in new_reg.get_ids():
            agent = new_reg.get_agent(id)
            action = actions[id]
            reward = points[id] if id in points.keys() else 0
            # penalize agent for game ending with no points
            if agent.type == "h" and reward == 0 and not self.play_status:
                point = -5

            agent.step(state, action, next_state, reward)

        self.current_step += 1

        return next_state_id

    def render(self, printToScreen=True):
        curr_pos = (0, 0)
        export = ''
        for row in self.map:
            output = ''
            curr_pos = (0, curr_pos[1])
            for space in row:
                s = ''
                if space == 0:
                    s = Back.BLACK + '0'+ Style.RESET_ALL
                elif space == 1:
                    s = ' '
                elif self.c_reg.in_reg(self.encoder.decode_id(space)):
                    characters = self.encoder.decode_id(space)
                    types = self.encoder.decode_type(space)

                    colors = {
                        "r": Back.YELLOW,
                        "h": Back.CYAN,
                        "s": Back.GREEN,
                        "o": Back.WHITE, # open space
                        "i": Back.MAGENTA # interaction
                    }

                    color = colors["o"]
                    id = " "

                    if len(characters) <= 1: # single character
                        character = characters[0]
                        color = colors[types[0]]
                        num = character[1]
                        id = "P" if (num == self.subject) else character[0]
                    else:
                        color = colors["i"]

                    s = color + id + Style.RESET_ALL
                output += s
                curr_pos = (curr_pos[0] + 1, curr_pos[1])
            curr_pos = (curr_pos[0], curr_pos[1] + 1)
            export += output + '\n'
        if printToScreen:
            print(export)
        return export

    ''' Util Functions '''
    def get_status(self):
        return self.play_status

    def get_character_positions(self):
        character_pos = []
        for c_key in self.characters_registry.keys():
            character_pos.append()

    # Map Methods

    def get_min_num_bits(self, m):
        n = 1
        while len(str(2 ** n - 1)) < m + 1:
            n += 1
        return n

    def round_base_2(self, m):
        if m <= 8:
            return 8
        n = 1
        while (2**n -1) < m:
            n += 1
        return (2**n)

    def get_d_type(self, num_chars):
        reference_dict = {
            8: np.int8,
            16: np.int16,
            32: np.int32,
            64: np.int64
        }
        n = self.get_min_num_bits(num_chars)
        m = self.round_base_2(n)
        if m in reference_dict.keys():
            return reference_dict[m]
        else:
            print("E: Too many characters in the game")
            return reference_dict[64]

    def create_map(self, m, n=0, c=1):
        d = self.get_d_type(c)
        l = m # map length
        while (l-1)**2 < c:
            l += 1
        if l != m:
            print("E: Too many characters for the desired map, map length increased to {}.".format(l))
        k = n if n >= l else l
        world = np.zeros((l, k), dtype=d)
        for i in range(l):
            for j in range(l):
                if self.validate_character_position([i, j]):
                    world[i][j] = 1
        return world

    def set_map(self, map):
        self.map = map

    def update_map(self, positions):
        self.map = self.base_map.copy()
        for pos in positions.keys():
            x , y = pos
            id = self.encoder.encode_id(positions[pos])
            self.map[y][x] = id
            for c_id in positions[pos]:
                self.c_reg.update_character(c_id, pos)

    # Agent Modification Methods

    def set_random_subject(self):
        ids = self.c_reg.get_ids()
        self.subject = random.choice(list(ids))

    def set_subject(self, id):
        if id in self.c_reg.get_ids():
            self.subject = id
        else:
            print("E: Was not given valid id for main agent.")

    def get_subject(self):
        return self.c_reg.get_character(self.subject)

    def add_agent(self, agent):
        if agent.id in self.c_reg.get_ids():
            self.c_reg.update_character(agent.id, agent, "agent")
        else:
            print("E: Character ({}) was not found in character registry.".format(a_key))

    def set_agents(self, agents):
        for a_key in agents.keys():
            agent = agents[a_kay]
            self.add_agent(agent)

    # Game Logic Methods
    def get_rewards(self):
        groups = self.i_manager.get_interactions()
        types = ['r', 's', 'h']
        points = []
        for group_id in groups:
            group = groups[group_id]
            group_points = self.i_manager.calculate_reward(group)
            points.append(group_points)

        final_points = {}
        for group_points in points:
            overlapping_keys = list(set(group_points.keys()) & set(final_points.keys()))
            if len(overlapping_keys) > 1:
                print("E: Character(s) in multiple groups:", overlapping_keys)
            final_points.update(group_points)
        return final_points

    def check_game_rules(self):
        if self.current_step >= self.MAX_STEP:
            return False

        groups = self.i_manager.get_interactions()
        counts = self.i_manager.get_multi_type_counts(groups)
        for count in counts:
            r, s, h = count
            if (r > 0 and h > 0) or (s > 0 and h > 1):
                return False
        return True

    def validate_character_position(self, position):
        x, y = position
        within_bounds = (self.x_bounds[0] <= x <= self.x_bounds[1]) and (self.y_bounds[0] <= y <= self.y_bounds[1])
        open_space = (self.map[y][x] != 0)
        return within_bounds and open_space

    def equivalent_positions(self, c1, c2):
        return (c1[0] == c2[0] and c1[1] == c2[1])

    # Generating Random Character Positions Methods

    def get_rand_bdd_number(self, bounds):
        # Returns a random number between two values, inclusive
        return random.randint(bounds[0], bounds[1])

    def rand_bdd_pair(self, bounds):
        # Returns a random pair of numbers
        pos = [-1, -1]
        for i in range(2):
            pos[i] = self.get_rand_bdd_number(bounds[i])
        return pos

    def rand_bdd_position(self):
        # Return a pair of random integers bounded by the x and y bounds
        bounds = [self.x_bounds, self.y_bounds]
        x, y = self.rand_bdd_pair(bounds)
        while self.map[y][x] != 1:
            x, y = self.rand_bdd_pair(bounds)
        return (x, y)

    def set_random_characters(self):
        # @TODO: Get grouping in case characters are at same position at start
        for id in self.c_reg.get_ids():
            # Encode character into a numerical ID
            encoded_id = self.encoder.encode_id([id]) # assuming space is empty
            x, y = self.rand_bdd_position()
            self.map[y][x] = encoded_id
            # Update position of character in registry
            self.c_reg.update_character(id, (x, y))
        self.i_manager.set_reg(self.c_reg.get_characters())

    # Display Methods

    def create_canvas(self):
        # Reset the canvas
        self.canvas = np.ones(self.window_shape) * 1
        text = 'Staghunt Environment'
        font = font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.canvas = cv2.putText(self.canvas, text, (10,20), font,
                    0.8, (0,0,0), 1, cv2.LINE_AA)

def main():
    # Setup Staghunt Environment
    env = StaghuntEnv()

    # Create Random Environment
    env.reset()
    env.render()

if __name__ == '__main__':
    main()
