# Test Class for the Encoder
from interaction_manager import InteractionManager, RABBIT_VALUE, STAG_VALUE

import unittest

c_reg = {}
im = InteractionManager(c_reg)


class TestInteractionManager(unittest.TestCase):

    ''' get_type_from_group() '''
    def test_gtfg_single(self):
        input = ["r1"]
        expected = ["r1"]
        actual = im.get_type_from_group("r", input)
        self.assertEqual(expected, actual)

    def test_gtfg_multiple(self):
        input = ["r1", "h1", "s1", "r2"]
        expected = ["r1", "r2"]
        actual = im.get_type_from_group("r", input)
        self.assertEqual(expected, actual)

    ''' calculate_reward() '''
    def test_cr_single(self):
        input = ["r1", "h1"]
        expected = {"r1": 0, "h1": RABBIT_VALUE}
        actual = im.calculate_reward(input)
        self.assertEqual(expected, actual)

    def test_cr_multiple_1(self):
        input = ["r1", "h1", "h2"]
        expected = {"r1": 0, "h1": RABBIT_VALUE, "h2": 0}
        actual = im.calculate_reward(input)
        self.assertEqual(expected, actual)

    def test_cr_multiple_2(self):
        input = ["r1", "h1", "r2", "h2"]
        expected = {"r1": 0, "r2": 0, "h1": RABBIT_VALUE, "h2": RABBIT_VALUE}
        actual = im.calculate_reward(input)
        self.assertEqual(expected, actual)

    def test_cr_multiple_3(self):
        input = ["r1", "h1", "s1", "h2"]
        expected = {"r1": RABBIT_VALUE, "s1": 0, "h1": STAG_VALUE / 2, "h2": STAG_VALUE / 2}
        actual = im.calculate_reward(input)
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
