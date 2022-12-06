# Test Class for the Encoder
import math
import unittest
import numpy as np

from state import State
from encoder import StaghuntEncoder
from feature_extractor import SimpleExtractor, StaghuntExtractor
from setup import simple_map, character_setup_1h1s1r, custom_map, map_init

e = StaghuntEncoder()

id = "h1"

# low value map
positions_1 = {
    "r1" : (4, 1),
    "h1" : (6, 1)
}

map_1 = map_init(enc=e, positions=positions_1)

state_1 = State(1, map_1, positions_1, 5, positions_1[id])

# medium value map
positions_2 = {
    "r1" : (4, 1),
    "h1" : (2, 5)
}

map_2 = map_init(enc=e, positions=positions_2)

state_2 = State(2, map_2, positions_2, 5, positions_2[id])

# high value map
positions_3 = {
    "r1" : (4, 1),
    "h1" : (2, 1)
}

map_3 = map_init(enc=e, positions=positions_3)

state_3 = State(3, map_3, positions_3, 5, positions_3[id])

'''
Use 'python -m unittest feature_extractor_test.TestFeatureExtractor.<Test Name>'
to run a specific test from this file.
'''

class TestFeatures(unittest.TestCase):

    ''' distance '''
    def test_distance(self):
        extr = SimpleExtractor(id)
        extr.pre_process_map(state_1.map)

        actual = []
        for state in [state_1, state_2, state_3]:
            features = extr.calculate_features(state, None)
            value = 0
            for feature in features.keys():
                value += features[feature]
            actual.append(value)
        last_val = -999999
        result = True
        for feature_val in actual:
            if feature_val > last_val:
                last_val = feature_val
            else:
                result = False
        self.assertTrue(result, "Test Failed: Resulting values are {}.".format(actual))

    ''' State movement '''
    def test_movement(self):
        new_state = state_3.move_character("h1", tuple((3, 1)), e, set_as_state=False)
        self.assertNotEqual(state_3.id, new_state.id)

if __name__ == '__main__':
    unittest.main()
