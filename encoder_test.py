# Test Class for the Encoder
from encoder import Encoder

import unittest

e = Encoder()

class TestEncoder(unittest.TestCase):

    ''' Encoding '''
    def test_encoding_single(self):
        input = ["r1"]
        expected = 2
        actual = e.encode_id(input)
        self.assertEqual(expected, actual)

    def test_encoding_multiple(self):
        input = ["r1", "s1", "h1"]
        expected = 246
        actual = e.encode_id(input)
        self.assertEqual(expected, actual)

    ''' Decoding '''
    def test_decoding_single(self):
        input = 2
        expected = ["r1"]
        actual = e.decode_id(input)
        self.assertEqual(expected, actual)

    def test_decoding_multiple(self):
        input = 246
        expected = ["r1", "s1", "h1"]
        actual = e.decode_id(input)
        self.assertEqual(expected, actual)

    def test_decoding_type_single(self):
        input = 2
        expected = ["r"]
        actual = e.decode_type(input)
        self.assertEqual(expected, actual)

    def test_decoding_type_multiple_diff(self):
        input = 246
        expected = ["r", "s", "h"].sort()
        actual = e.decode_type(input).sort()
        self.assertEqual(expected, actual)

    def test_decoding_type_multiple_same(self):
        input = 245
        expected = ["r", "s"].sort()
        actual = e.decode_type(input).sort()
        self.assertEqual(expected, actual)

    def test_decode_id_to_character(self):
        input = 4
        expected = "s1"
        actual = e.decode_id_to_character(input)
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
