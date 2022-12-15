import random
import numpy as np
from encoder import StaghuntEncoder
from interaction_manager import STEP_COST

TABLE_AGENTS = ["QLearningAgent"]

class StaticAgent:
    def __init__(self, id="default"):
        self.id = id
        self.map = None

    def get_name(self):
        return self.__class__.__name__

    def direction_to_vector(self, direction):
        reference_dict = {
            'n':[0, -1],
            'e':[1, 0],
            's':[0, 1],
            'w':[-1, 0],
        }
        return reference_dict[direction]

    def check_within_bounds(self, a, b, x):
        return (a <= x) and (x <= b)

    def validate_agent_position(self, x, y):
        within_y_bounds = self.check_within_bounds(1, len(self.map)-2, y)
        within_x_bounds = self.check_within_bounds(1, len(self.map[0])-2, x)
        open_space = (self.map[y][x] != 0)
        return open_space and (within_y_bounds and within_x_bounds)

    def calc_move(self, pos, dir_indx):
        # dir represents direction as a number from 0 to 3, N to W
        dir = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        a = pos[0] + dir[i][0]
        b = pos[1] + dir[i][1]
        return tuple([a, b])

    def generate_valid_moves(self, pos):
        moves = []
        dir = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        for i in range(4):
            a = pos[0] + dir[i][0]
            b = pos[1] + dir[i][1]
            if self.validate_agent_position(a, b):
                moves.append(tuple([a, b]))
        return moves

    # Overriden Methods
    def get_move(self, state):
        self.map = state.map.copy()
        return state.positions[self.id]

class RandomAgent(StaticAgent):
    def __init__(self, id):
       StaticAgent.__init__(self, id)

    def get_rand_move(self, pos):
         moves = self.generate_valid_moves(pos)
         return random.choice(moves)

    def get_move(self, state):
        pos = state.positions[self.id]
        self.map = state.map.copy()
        return self.get_rand_move(pos)

class StaghuntAgent(RandomAgent):
    def __init__(self, id='default-staghunt', type="", kind="general"):
        RandomAgent.__init__(self, id)
        self.kinds = {
            "general":"general",
            "prey":"prey",
            "static-prey": "static-prey",
            "general-hunter":"general-hunter",
            "specific-hunter":"specific-hunter",
        }

        self.type = type

        self.kind = self.kinds[kind]

        self.reward = 0

        self.encoder = StaghuntEncoder()

    def reset(self):
        self.reward = 0

    def get_move(self, state):
        pos = state.positions[self.id]
        self.map = state.map.copy()
        if self.kind != self.kinds["static-prey"]:
            return self.get_rand_move(pos)
        return pos

    def step(self, state, action, next_state, reward):
        # Update agent logic
        final_reward = reward
        if reward == 0 and self.type == "h":
            final_reward = STEP_COST
        self.reward += final_reward

class ManualAgent(StaghuntAgent):
    def __init__(self, id, type):
        # print("Creating Manual Control Agent.")
        StaghuntAgent.__init__(self, id, type)

    def convert_input(self, user_input, pos):
        dir = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        index_dict = {
            "N": 0,
            "E": 1,
            "S": 2,
            "W": 3
        }
        if user_input.capitalize() not in index_dict.keys():
            return tuple([0, 0])
        index = index_dict[user_input.capitalize()]
        moves = np.add(pos, dir[index])
        actions = [tuple(x) for x in moves]
        return actions

    def contains_pos(self, sub_arr, arr):
        for element in arr:
            if tuple(sub_arr) == tuple(element):
                return True
        return False

    def get_move(self, state):
        pos = state.positions[self.id]
        self.map = state.map.copy()
        moves = self.generate_valid_moves(pos)
        move = []
        prompt = 'Input direction (NESW) for {}:'.format(self.id)
        while not self.contains_pos(move, moves):
            user_input = input(prompt)
            move = self.convert_input(user_input, pos)
            prompt = 'Input a valid direction (NESW) for {}:'.format(self.id)
        return move

class PreyAgent(StaghuntAgent):
    # Create a class for the stags to run away
    def __init__(self, id, type, predator_type="h"):
        StaghuntAgent.__init__(self, id, type, kind="prey")
        self.predator_type = predator_type
        self.reach = 2

    def get_move(self, state):
        # Prey Policy: Run away from hunters
        curr_pos = state.positions[self.id]
        self.map = state.map.copy()
        moves = self.generate_valid_moves(curr_pos)
        optimal_moves = []
        any_hunter_present = False
        for move in moves:
            x, y = move
            hunter_present = False
            for r in range(self.reach):
                reach = r * np.subtract(curr_pos, move)
                hunter_present = hunter_present or self.predator_type in self.encoder.decode_type(self.map[y + reach[1]][x + reach[0]])
            any_hunter_present = any_hunter_present or hunter_present
            if not hunter_present:
                optimal_moves.append(move)

        if len(optimal_moves) > 0: # ∃ position s.t. no hunters are there
            if any_hunter_present or (not any_hunter_present and random.uniform(0, 1) < 0.6):
                return random.choice(optimal_moves)

        return curr_pos

class BasicHunterAgent(StaghuntAgent):
    # Agent inspired by Project Malmo's Blue Agent
    def __init__(self, id):
        StaghuntAgent.__init__(self, id, "h")
        self.kind = self.kinds["general-hunter"]
        self.prey_types = ["r", "s"]
        self.prob_stag_target = 0.75
        self.target = None
        self.reset_target()

    def set_target(self, target_id):
        self.target = target_id

    def reset_target(self):
        # @TODO: Make the target setting more dynamic
        if random.uniform(0, 1) < self.prob_stag_target:
            self.target = "s1"
        else:
            self.target = "r1"

    def reset(self):
        self.reward = 0
        self.reset_target()

    def get_distances(self, state):
        # A BFS to calculate distance of hunter to each prey character
        character_distances = {}
        src = state.positions[self.id]
        visited = [tuple(src)]
        queue = [(src, [src])]

        while queue and len(character_distances.keys()) < len(state.positions.keys()):
            next_node, path = queue.pop(0)
            dist = len(path) - 1
            x, y = next_node
            ids = self.encoder.decode_id(state.map[y][x])
            for id in ids:
                if id not in character_distances.keys() or dist < character_distances[id]:
                    character_distances[id] = { "dist" : dist, "path" : path }

            neighbors = self.generate_valid_moves(next_node)
            for nbr in neighbors:
                node = tuple(nbr)
                if node not in visited:
                    new_path = path + [node]
                    visited.append(node)
                    queue.append((node, new_path))

        return character_distances

    def calc_proximally_optimal_move(self, state, target_char=None):
        character_distances = self.get_distances(state)
        characters = []
        for c_key in character_distances.keys():
            if c_key != self.id:
                character = character_distances[c_key]
                characters.append(tuple((character["dist"], c_key)))
        characters.sort()

        for target in [target_char, self.target]:
            if target is not None and target in character_distances.keys():
                path = character_distances[target]["path"]
                return path[1] if len(path) > 1 else path[0]

        while characters:
            dist, c_id = characters.pop(0)
            type = c_id[0]
            path = character_distances[c_id]["path"]
            move = path[1] if len(path) > 1 else path[0]
            if type == "r" or (type == "s" and random.uniform(0, 1) >= self.prob_target_stag):
                return move

        return self.get_rand_move(state.positions[self.id])

    def get_move(self, state):
        pos = state.positions[self.id]
        self.map = state.map.copy()

        move = self.calc_proximally_optimal_move(state)

        return move

class BruteForceAgent(BasicHunterAgent):
    def __init__(self, id):
        BasicHunterAgent.__init__(self, id)

    def get_rand_move(self, pos):
         moves = self.generate_valid_moves(pos)
         return random.choice(moves)

    def get_move(self, state):
        pos = state.positions[self.id]
        self.map = state.map.copy()
        return self.get_rand_move(pos)
