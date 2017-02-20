from unittest import TestCase

from utils import preprocess_to_chars, preprocess_to_hashes, Encoder


class TestUtils(TestCase):
    def test_preprocess_to_chars(self):
        tab = preprocess_to_chars(["ab b", "a"])
        self.assertEqual(
            tab,
            [97, 98, 32, 98, 59, 97]
        )

    def test_preprocess_to_hashes(self):
        tab = preprocess_to_hashes(["ab b", "a"])
        self.assertEqual(
            tab,
            [hash("ab") % 8000, hash("b") % 8000, hash(";") % 8000, hash("a") % 8000]
        )

    def test_encoder(self):
        encoder = Encoder()
        self.assertEqual(encoder.transform("a"), 0)
        self.assertEqual(encoder.transform("b"), 1)
        self.assertEqual(encoder.transform("a"), 0)
 
