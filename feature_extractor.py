import collections
import numpy as np

from setup import MAX_GAME_LENGTH
from encoder import StaghuntEncoder
from interaction_manager import InteractionManager, RABBIT_VALUE, STAG_VALUE

class FeatureExtractor():
    def __init__(self, id):
        self.id = id # id of agent
        self.enc = StaghuntEncoder()
        self.im = InteractionManager()
        self.max_cache_size = 20
        self.cache = {}
        self.next_item = None
        self.num_cache_accesses = 0
        self.num_cache_additions = 0

    def valid_pos(self, map, pos):
        x, y = pos
        if (0 <= y < len(map)) and (0 <= x < len(map[0])) and map[y][x] != 0:
            return True
        return False

    def generate_valid_moves(self, map, pos):
        moves = []
        dir = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        for i in range(4):
            a = pos[0] + dir[i][0]
            b = pos[1] + dir[i][1]
            if map[b][a] != 0:
                moves.append(tuple([a, b]))
        return moves

    def check_cache(self, sa_id):
        if sa_id in self.cache.keys():
            return True, self.cache[sa_id]["features"]
        return False, None

    def update_cache_ctr(self, sa_id):
        self.cache[sa_id]["ctr"] += 1

    def update_next_item(self):
        self.next_item = None
        min_count = 0
        for key in self.cache.keys():
            count, features = self.cache[key].values()
            if count < min_count:
                self.next_item = key

    def add_to_cache(self, sa_id, value):
        # Delete most infrequent item if cache full
        if len(self.cache) >= self.max_cache_size:
            del self.cache[self.next_item]
            self.next_item = sa_id
        else:
            self.next_item = sa_id if self.next_item is None or 1 < self.cache[self.next_item]["ctr"] else self.next_item

        self.cache[sa_id] = {"ctr" : 1, "features": value }

    def update_cache(self, sa_id, value):
        exists, preexisting_value = self.check_cache(sa_id)
        if exists:
            self.update_cache_ctr(sa_id)
            self.num_cache_accesses += 1
        else:
            self.add_to_cache(sa_id, value)
            self.num_cache_additions += 1

    def print_cache(self):
        print("------Cache for {}------".format(self.__class__.__name__))
        print("Cache has {} accesses and {} additions.".format(self.num_cache_accesses, self.num_cache_additions))
        for key in self.cache.keys():
            item =  self.cache[key]
            print(" {} had {} references.".format(key, item["ctr"]))

class SimpleExtractor(FeatureExtractor):
    def __init__(self, id):
        FeatureExtractor.__init__(self, id)
        self.pre_processed = False
        self.values = {
            'r' : RABBIT_VALUE,
            's' : STAG_VALUE,
            'h' : 1
        }

    def pre_process_map(self, map):
        max_dist = 0
        num_inner_walls = 0
        num_turns = 0
        total_rabbits = 0
        total_stags = 0

        for i in range(1, len(map) - 1):
            for j in range(1, len(map[0]) - 1):
                top = self.valid_pos(map, (j, i - 1))
                right = self.valid_pos(map, (j + 1, i))
                center = map[i][j]
                bottom = self.valid_pos(map, (j, i + 1))
                left = self.valid_pos(map, (j - 1, i))
                if center == 0:
                    num_inner_walls += 1
                else:
                    r, s, h = self.im.get_type_counts(self.enc.decode_id(center))
                    total_rabbits += r
                    total_stags += s
                    max_dist += 1
                if (top and right) or (top and left) or (bottom and right) or (bottom and left):
                    num_turns += 1

        self.max_dist = max_dist
        self.max_num_walls = num_inner_walls
        self.max_num_turns = num_turns
        self.max_num_rabbits = total_rabbits
        self.max_num_stags = total_stags
        self.pre_processed = True

    def get_character_distances(self, state, id):
        # A BFS to calculate distance of hunter to each prey character
        character_distances = {}
        src = state.positions[id]
        visited = [tuple(src)]
        queue = [(src, [src])]

        while queue and len(character_distances.keys()) < len(state.positions.keys()):
            next_node, path = queue.pop(0)
            dist = len(path) - 1
            x, y = next_node
            ids = self.enc.decode_id(state.map[y][x])
            for id in ids:
                if id not in character_distances.keys() or dist < character_distances[id]:
                    character_distances[id] = { "dist" : dist, "path" : path }

            neighbors = self.generate_valid_moves(state.map, next_node)
            for nbr in neighbors:
                node = tuple(nbr)
                if node not in visited:
                    new_path = path + [node]
                    visited.append(node)
                    queue.append((node, new_path))

        return character_distances

    def calculate_features(self, state):
        '''
        Preprocessed data:
        - max distance in the map

        Features:
        - bias
        - distance-to-{}: distance hunter to each prey / max distance ∈ [0, 1) -> Expand for all hunters
        '''
        features = collections.Counter()
        features["bias"] = 1.0

        # Get the ids of all the hunters
        hunters = [id for id in state.positions.keys() if id[0] == "h"]
        for id in hunters:
            # Get distance and path of a hunter (id) to each character using BFS
            character_distances = self.get_character_distances(state, id)
            for c_key in character_distances.keys():
                if c_key != id:
                    dist, path = character_distances[c_key].values()
                    # distance: self.max_dist - dist
                    # Note: If feature weight < 0 then worse distances will be be selected
                    features["{}-dist-to-{}".format(id, c_key)] = (self.max_dist - dist) / self.max_dist #1 / (dist + 1)

        return features

    def get_features(self, state):
        exists, value = self.check_cache(state.id)
        if exists:
            self.update_cache(state.id, None)
            return value

        features = self.calculate_features(state)

        self.update_cache(state.id, features)

        return features

class StaghuntExtractor(SimpleExtractor):
    def __init__(self, id):
        SimpleExtractor.__init__(self, id)

    def valid_prey_and_pos(self, map, pos):
        x, y = pos
        num_rabbits = 0
        num_stags = 0
        if self.valid_pos(map, pos):
            space = map[y][x]
            r, s, h = self.im.get_type_counts(space)
            num_rabbits += r
            num_stags += s
        return (num_rabbits, num_stags)

    def count_turns(self, path):
        if len(path) <= 1:
            return 0
        num_turns = 0
        diff = [path[1][0] - path[0][0], path[1][1] - path[0][1]]
        for i in range(len(path) - 1):
            curr = path[i]
            next = path[i+1]
            if tuple((curr[0] + diff[0], curr[1] + diff[1])) != next:
                diff = [next[0] - curr[0], next[1] - curr[1]]
                num_turns += 1
        return num_turns

    def count_num_prey_adjacent(self, state):
        num_rabbits = 0
        num_stags = 0
        x, y = state.positions[self.id]
        map = state.map
        adjacent = [self.valid_prey_and_pos(map, (x, y - 1)), self.valid_prey_and_pos(map, (x + 1, y)), self.valid_prey_and_pos(map, (x, y + 1)), self.valid_prey_and_pos(map, (x - 1, y))]
        for count in adjacent:
            r, s = count
            num_rabbits += r
            num_stags += s
        return (num_rabbits, num_stags)

    def calculate_features(self, state):
        '''
        Features:
        - bias
        - num-steps: current number of steps / total number of steps
        - distance-to-{}: distance hunter to each prey / max distance ∈ [0, 1) -> Expand for all hunters
        - labor vs. return-{} (LVR): value of each prey * distance to each prey /  max distance ∈ [0, 1)
        - count-turns-to-{}: the number of turns that a hunter needs to take to get to target ∈ [0, 1)
        - num-{}-adjacent: # of each kind prey one step away / total # of each prey ∈ [0, 1]
        '''

        features = collections.Counter()
        features["bias"] = 1.0

        # @TODO: Add more features
        num_rabbits, num_stags = self.count_num_prey_adjacent(state)
        features["num_rabbits"] = num_rabbits
        features["num_stags"] = num_stags

        return features
