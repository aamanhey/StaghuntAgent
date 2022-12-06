class State:
    def __init__(self, id=None, map=None, positions=None, steps=None, pos=None):
        '''
        - id: A numerical staghunt map encoding
        - map: A numerical matrix with encoded characters
        - positions: A dictionary of character ids as keys with position tuples (x, y)
        - steps: A number representing how many steps have been taken in the game
        - pos: A tuple representing the position of the subject
        '''
        self.id = id
        self.map = map
        self.positions = positions
        self.steps = steps
        self.pos = pos

    def set_id(self, id):
        self.id = id

    def set_map(self, map):
       self.map = map

    def set_positions(self, positions):
        self.positions = positions

    def set_steps(self, steps):
        self.steps = steps

    def set_pos(self, pos):
        self.pos = pos

    def move_character(self, id, position, encoder, set_as_state=False):
        # @TODO: The positions are not changing
        map = self.map.copy()
        x_old, y_old = self.pos
        x_new , y_new = position
        # delete character from old position
        ids = encoder.decode_id(map[y_old][x_old])
        ids.remove(id)
        map[y_old][x_old] = encoder.encode_id(ids)
        # add character to new position
        ids = encoder.decode_id(map[y_new][x_new])
        ids.append(id)
        map[y_new][x_new] = encoder.encode_id(ids)

        new_id = encoder.encode(map)

        if not set_as_state:
            new_positions = self.positions.copy()
            new_positions[id] = position
            new_state = State(new_id, map, new_positions, self.steps, position)
            return new_state

        self.id = new_id
        self.map = map
        self.positions[id] = position
        self.pos = position

        return None

    def print(self):
        print("State {}:".format(self.id))
        print("- map:\n{}".format(self.map))
        print("- positions:\n{}".format(self.positions))
        print("- steps: {}".format(self.steps))
        print("- pos: {}".format(self.pos))
