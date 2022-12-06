# Test Class for the Encoder
import math
import unittest
import numpy as np

from state import State
from encoder import StaghuntEncoder
from feature_extractor import StaghuntExtractor
from setup import simple_map, character_setup_1h1s1r, custom_map, map_init

id = "h1"
extr = StaghuntExtractor(id)
e = StaghuntEncoder()

positions = {
    "r1" : (1, 1),
    "s1" : (2, 4),
    "h1" : (4, 1),
    "h2" : (3, 3)
}

map = map_init(positions=positions)

state = State(0, map, positions, 5, positions[id])

'''
Use 'python -m unittest feature_extractor_test.TestFeatureExtractor.<Test Name>'
to run a specific test from this file.
'''

class TestFeatureExtractor(unittest.TestCase):

    ''' get_dir() '''
    def test_get_dir(self):
        action = (4, 2)
        expected = 3 / 7
        actual = extr.get_dir(state, action)
        self.assertEqual(expected, actual)

    ''' get_dist_to_character() '''
    def test_get_dist_to_character1(self):
        action = (4, 2)
        expected = math.sqrt(10)
        actual = extr.get_dist_to_character(state, action, "r1")
        self.assertEqual(expected, actual)

    def test_get_dist_to_character2(self):
        action = (3, 1)
        expected = 2
        actual = extr.get_dist_to_character(state, action, "r1")
        self.assertEqual(expected, actual)

    ''' get_num_obstacles() '''
    def test_get_num_obstacles1(self):
        action = (4, 2)
        expected = 0
        actual = extr.get_num_obstacles(state, action, "r1")
        self.assertEqual(expected, actual)

    def test_get_num_obstacles2(self):
        action = (4, 2)
        expected = 2
        map = np.ones((7,7)) #state.map
        map[1, 2] = 0
        map[2, 2] = 0
        custom_state = State(0, map, positions, 5, positions[id])
        actual = extr.get_num_obstacles(custom_state, action, "r1")
        self.assertEqual(expected, actual)

    ''' get_character_distances() '''
    def test_get_character_distances(self):
        complex_map = custom_map.copy()
        positions_2 = {
            "r1" : (2, 1),
            "s1" : (4, 5),
            "h1" : (3, 3),
            "h2" : (2, 5)
        }
        for key in positions_2.keys():
            x, y = positions_2[key]
            complex_map[y][x] = e.encode_id([key])
        custom_state = State(0, complex_map, positions_2, 5, positions_2[id])
        expected = {
            'h1': {'dist': 0, 'path': [(3, 3)]},
            'h2': {'dist': 3, 'path': [(3, 3), (2, 3), (2, 4), (2, 5)]},
            'r1': {'dist': 3, 'path': [(3, 3), (2, 3), (2, 2), (2, 1)]},
            's1': {'dist': 3, 'path': [(3, 3), (4, 3), (4, 4), (4, 5)]}
        }
        actual = extr.get_character_distances(custom_state)
        self.assertEqual(expected, actual)

    ''' check_inline() '''
    def test_count_turns1(self):
        path = [(3, 3), (2, 3), (2, 4), (2, 5), (3, 5)]
        actual = extr.count_turns(path)
        expected = 2
        self.assertEqual(expected, actual)

    def test_count_turns2(self):
        path = [(2, 5), (2, 4), (2, 3), (2, 2), (2, 1)]
        actual = extr.count_turns(path)
        expected = 0
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
