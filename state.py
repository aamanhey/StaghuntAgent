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
