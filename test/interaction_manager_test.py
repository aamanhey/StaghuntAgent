# Test Class for the Encoder
import unittest

from context import InteractionManager, RABBIT_VALUE, STAG_VALUE

c_reg = {}
im = InteractionManager(c_reg)

'''
Use 'python -m unittest interaction_manager_test.TestInteractionManager.<Test Name>'
to run a specific test from this file.
'''

class TestInteractionManager(unittest.TestCase):

    ''' get_type_from_group() '''
    def test_gtfg_r1(self):
        input = ["r1"]
        expected = ["r1"]
        actual = im.get_type_from_group("r", input)
        self.assertEqual(expected, actual)

    def test_gtfg_r1r2s1h1(self):
        input = ["r1", "r2", "s1", "h1",]
        expected = ["r1", "r2"]
        actual = im.get_type_from_group("r", input)
        self.assertEqual(expected, actual)

    ''' calculate_reward() '''
    def test_cr_r1h1(self):
        input = ["r1", "h1"]
        expected = {"r1": 0, "h1": RABBIT_VALUE}
        actual = im.calculate_reward(input)
        self.assertEqual(expected, actual)

    def test_cr_h1(self):
        input = ["h1"]
        expected = {"h1": 0}
        actual = im.calculate_reward(input)
        self.assertEqual(expected, actual)

    def test_cr_s1h1(self):
        input = ["s1", "h1"]
        expected = {"s1": STAG_VALUE, "h1": 0}
        actual = im.calculate_reward(input)
        self.assertEqual(expected, actual)

    def test_cr_r1h1h2(self):
        input = ["r1", "h1", "h2"]
        expected = {"r1": 0, "h1": RABBIT_VALUE, "h2": 0}
        actual = im.calculate_reward(input)
        self.assertEqual(expected, actual)

    def test_cr_r1r1h1h2(self):
        input = ["r1", "r2", "h1", "h2"]
        expected = {"r1": 0, "r2": 0, "h1": RABBIT_VALUE, "h2": RABBIT_VALUE}
        actual = im.calculate_reward(input)
        self.assertEqual(expected, actual)

    def test_cr_r1s1h1h2(self):
        input = ["r1", "s1", "h1", "h2"]
        expected = {"r1": RABBIT_VALUE, "s1": 0, "h1": STAG_VALUE / 2, "h2": STAG_VALUE / 2}
        actual = im.calculate_reward(input)
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
