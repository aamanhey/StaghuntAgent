import random
import collections
import math
import numpy as np

from interaction_manager import InteractionManager
from encoder import StaghuntEncoder

class StaticAgent:
    def __init__(self, id="default"):
        self.id = id
        self.map = None

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
        return  within_y_bounds and within_x_bounds

    def calc_move(self, pos, dir_indx):
        # dir represents direction as a number from 0 to 3, N to W
        dir = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        a, b = np.add(pos, dir[dir_indx])
        return [a, b]

    def generate_valid_moves(self, pos):
        moves = []
        dir = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        for i in range(4):
            a, b = np.add(pos, dir[i])
            if self.validate_agent_position(a, b):
                moves.append([a, b])
        return moves

    # Overriden Methods
    def get_move(self, map, pos):
        self.map = map.copy()
        return pos

class RandomAgent(StaticAgent):
    def __init__(self, id):
       StaticAgent.__init__(self, id)

    def get_rand_move(self, pos):
         moves = self.generate_valid_moves(pos)
         return random.choice(moves)

    def get_move(self, map, pos):
        self.map = map.copy()
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

    def reset(self):
        self.reward = 0

    def get_move(self, map, pos):
        self.map = map.copy()
        if self.kind != self.kinds["static-prey"]:
            return self.get_rand_move(pos)
        return pos

    def step(self, state, move, next_state, reward):
        # Update agent logic
        final_reward = reward
        if reward == 0 and self.type == "h":
            final_reward = -1
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
            return [0, 0]
        index = index_dict[user_input.capitalize()]
        move = np.add(pos, dir[index])
        return move

    def contains_pos(self, sub_arr, arr):
        for element in arr:
            if tuple(sub_arr) == tuple(element):
                return True
        return False

    def get_move(self, map, pos):
        self.map = map.copy()
        moves = self.generate_valid_moves(pos)
        move = []
        prompt = 'Input direction (NESW) for {}:'.format(self.id)
        while not self.contains_pos(move, moves):
            user_input = input(prompt)
            move = self.convert_input(user_input, pos)
            prompt = 'Input a valid direction (NESW) for {}:'.format(self.id)
        return move

class BasicHunterAgent(StaghuntAgent):
    def __init__(self, id):
        StaghuntAgent.__init__(self, id, "h")
        self.kind = self.kinds["general-hunter"]
        self.prey_types = ["r", "s"]

class ProximityAgent(BasicHunterAgent):
    # @TODO: Make this a functional agent
    def __init__(self, id, targets=[], reach=1):
        BasicHunterAgent.__init__(self, id)
        self.targets = targets # ids of each target
        self.reach = reach # how far away the agent can see other characters
        self.encoder = StaghuntEncoder()
        self.current_target = None
        self.current_target_dist = -1

    def set_targets(self, targets):
        self.targets = targets

    def set_targets_from_map(self, targets, map):
        # Finds targets on a given map
        target_positions = {}
        for i in range(len(map)):
            for j in range(len(map[0])):
                value = map[i][j]
                characters = self.encoder.decode_id(value)
                if self.kind == self.kinds["specific-hunter"]:
                    characters = self.encoder.decode_id_to_character(value)
                for character in characters:
                    if character in targets:
                        target_positions[character] = (i, j)
        return target_positions

    def get_dist(self, c1, c2):
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def get_dir_to_target(self, pos, target_pos):
        max = self.max_dist
        optimal_dirs = [] # optimal directions

        # @TODO: Fix map indexing

        # Get a list of moves with shortest distance to target
        for i in range(4):
            move = self.calc_move(pos, i)
            if self.validate_agent_position(move[0], move[1]):
                dist = self.get_dist(move, target_pos)
                if dist < max:
                    optimal_dirs = [i]
                    max = dist
                elif dist == max:
                    optimal_dirs.append(i)

        # Return optimal move, if any
        if len(optimal_dirs) > 0:
            i = random.randint(0, len(optimal_dirs) - 1)
            move = self.calc_move(pos, optimal_dirs[i])

            return (self.get_dist(move, target_pos), optimal_dirs[i])

        # Target not found
        return (-1, -1)

    def get_move(self, map, pos):
        self.map = map.copy()
        self.max_dist = len(map ** 2) + 1

        x, y = pos
        r = self.reach
        observable_space = map[y-r:y+r+1, x-r:x+r+1]
        # Gives the position in the sub matrix, not the original map
        target_positions = self.set_targets_from_map(self.targets, observable_space)

        # Choose shortest path to closest target
        optimal_dist = self.max_dist
        optimal_targets = []
        optimal_moves = []
        for target in self.targets:
            # Pass if target not found
            if target in target_positions.keys():
                target_pos = target_positions[target]
                # ProximityAgent Policy: Attempt to capture the closest prey
                dist, dir = self.get_dir_to_target(pos, target_pos)
                move = self.calc_move(pos, dir)
                if 0 < dist < optimal_dist:
                    optimal_dist = dist
                    optimal_moves = [move]
                    optimal_targets = [target]
                elif dist == optimal_dist:
                    optimal_moves.append(move)
                    optimal_targets.append(target)
        if len(optimal_moves) > 0:
            i = random.randint(0, len(optimal_moves) - 1)
            self.current_target = optimal_targets[i]
            self.current_target_dist = optimal_dist

            return optimal_moves[i]

        # Target not found, get random move
        return self.get_rand_move(pos)

class PreyAgent(StaghuntAgent):
    # Create a class for the stags to run away
    def __init__(self, id, type, predator_type="h"):
        StaghuntAgent.__init__(self, id, type, kind="prey")

    def get_move(self, map, pos):
        self.map = map.copy()
        moves = self.generate_valid_rand_moves()
        optimal_moves = []
        for move in moves:
            x, y = move
            # Prey Policy: Stay away from hunters
            unoccupied = (self.map[y][x] == 1)
            unoccupied_by_predator = (predator_ids and self.encoder.decode_type(self.map[y][x]) != predator_type)
            if unoccupied or unoccupied_by_predator:
                optimal_moves.append(move)
        if len(optimal_moves) > 0:
            return random.choice(optimal_moves)
        else:
            return pos

class BruteForceAgent(BasicHunterAgent):
    def __init__(self, id):
        BasicHunterAgent.__init__(self, id)

    def get_rand_move(self, pos):
         moves = self.generate_valid_moves(pos)
         return random.choice(moves)

    def get_move(self, map, pos):
        self.map = map.copy()
        return self.get_rand_move(pos)

class QLearningAgent(BasicHunterAgent):
    def __init__(self, id, alpha=0.1, epsilon=0.5, gamma=0.8, delta=0.001):
        BasicHunterAgent.__init__(self, id)
        '''
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        delta    - convergence factor
        '''
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)

        self.delta = float(delta)
        self.hasConverged = False
        self.deltas = []

        self.inTraining = False

        self.encoder = StaghuntEncoder()

        self.q_value = {}

    # Convergence Methods

    def get_delta_avg(self):
        return np.average(self.deltas)

    def calc_delta(self, q_i, q_j):
        delta = (q_i - q_j)
        divisor = q_j
        if q_i == 0 and q_j == 0:
            divisor = 1
        elif q_i != 0 and q_j == 0:
            divisor = q_i
        return abs(delta/divisor)

    # Training Methods

    def reset(self):
        self.reward = 0
        self.hasConverged = False
        self.deltas = []

    def training_complete(self):
        return self.hasConverged

    def toggleTraining(self):
        self.inTraining = not self.inTraining

    # Q-Value Methods
    def init_q(self, map_id, move_id):
        c = False
        if map_id not in self.q_value.keys():
            self.q_value[map_id] = {}
            c = True
        if move_id not in self.q_value[map_id].keys():
            self.q_value[map_id][move_id] = 0
            c = True
        return c

    def get_q_values(self, state_id):
        return self.q_value[state_id]

    def get_q_value(self, map, move):
        map_id = self.encoder.encode(map)
        move_id = tuple(move)
        self.init_q(map_id, move_id)
        return self.q_value[map_id][move_id]

    def print_q_table(self):
        print("Q-Table for {}:".format(self.id))
        for state_id in self.q_value.keys():
            values = self.q_value[state_id]
            print("{}:".format(state_id))
            for val in values:
                print("- {}: {}".format(val, round(self.q_value[state_id][val], 4)))

    # Q-Policy Methods

    def calc_max_value(self, state):
        # Gives max Q-Value of possible moves at a state, equating the value of that state
        map, pos = state
        max_val = -999999
        moves = self.generate_valid_moves(pos)
        for move in moves:
            q_val = self.get_q_value(map, move)
            if q_val > max_val:
                max_val = q_val
        return max_val

    def calc_optimal_move(self, state):
        # Gives optimal move at a state
        map, pos = state
        max_val = -999999
        moves = self.generate_valid_moves(pos)
        optimal_moves = []
        for move in moves:
            q_val = self.get_q_value(map, move)
            if q_val > max_val:
                max_val = q_val
                optimal_moves = [move]
            elif q_val == max_val:
                optimal_moves.append(move)
        return random.choice(optimal_moves)

    def get_move(self, map, pos):
        # Gives move based on Q-Value
        self.map = map.copy()
        moves = self.generate_valid_moves(pos)

        if random.uniform(0, 1) < self.epsilon and self.inTraining:
            move = random.choice(moves)
        else:
            move = self.calc_optimal_move((map, pos))

        return move

    def step(self, state, move, next_state, reward):
        # Update agent logic
        map, pos = state
        final_reward = reward
        if reward == 0 and self.type == "h":
            final_reward = -1

        if self.inTraining:
            feedback = final_reward + self.gamma * self.calc_max_value(next_state)
            map_id = self.encoder.encode(map)
            move_id = tuple(move)
            self.init_q(map_id, move_id)

            # Update convergence metrics
            old_q_val = self.q_value[map_id][move_id]
            new_q_val = (1.0 - self.alpha) * self.get_q_value(map, move) + self.alpha * feedback
            delta = self.calc_delta(new_q_val, old_q_val)
            self.deltas.append(round(delta, len(str(delta)) - 1))
            if np.average(self.deltas) < self.delta:
                self.hasConverged = True

            self.q_value[map_id][move_id] = new_q_val

        self.reward += final_reward

'''
class ApprxQLearningAgent(StaghuntAgent):
    # @TODO: Create ApprxQLearningAgent Class
'''
