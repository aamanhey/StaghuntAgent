import numpy as np

class Encoder():
    def __init__(self, setupData=[["c"], {"c1": 2}, ["c1"]]):
        self.types = setupData[0]
        self.numeric_ids = setupData[1] # O and 1 reserved for closed and open spaces
        self.alpha_numeric_ids = setupData[2]

    '''
    The encoding process transforms an array of character ids into a numeric value.
    '''

    def get_reg(self):
        return self.numeric_ids.copy()

    def encode_id(self, ids):
        if ids == []:
            return 1
        encoded = ''
        for id in ids:
            encoded += str(self.numeric_ids[id])
        return int(encoded)

    def decode_id(self, id):
        ids = []
        if not (id == 0 or id == 1):
            str_id = str(id)
            for i in str_id:
                s = self.alpha_numeric_ids[int(i) - 2]
                ids.append(s)
        return ids

    def decode_type(self, id):
         ids = self.decode_id(id)
         str_id = str(id)
         types = set()
         for str_id in ids:
             types.add(str_id[0])
         return list(types)

    def decode_id_to_character(self, id):
        return self.decode_id(id)[0]

    def encode_simple(self, map):
        return str(map)

    def decode_simple(self, map):
        return np.matrix(map)

    def encode(self, state_map):
        return (f"{len(state_map)}{len(state_map[0])}{''.join(state_map.flatten().astype('str'))}")

    def decode(self, encoded_state):
        m = str(encoded_state[0])
        n = str(encoded_state[1])
        map = self.create_map(m, n)

        for i in range(m):
            for j in range(n):
                map[i][j] = int(encoded_state[i*n + j])
        return index_arr

class StaghuntEncoder(Encoder):
    def __init__(self):
        self.types = ["r", "s", "h"]

        self.numeric_ids = {
            "r1": 2,
            "r2": 3,
            "s1": 4,
            "s2": 5,
            "h1": 6,
            "h2": 7,
            "h3": 8
        }

        self.type_ids = {
            "r" : [2, 3],
            "s" : [4, 5],
            "h" : [6, 7, 8]
        }

        self.alpha_numeric_ids = ["r1", "r2", "s1", "s2", "h1", "h2", "h3"]

        setupData = [self.types, self.numeric_ids, self.alpha_numeric_ids]

        Encoder.__init__(self, setupData)
