import unittest

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.biopython import get_preprocessed_index

class TestGetPreprocessedIndex(unittest.TestCase):
    def test_basic_sequence(self):
        """Test basic sequence with no special characters"""
        sequence = "ABCDE"
        self.assertEqual(get_preprocessed_index(sequence, 0), 0)
        self.assertEqual(get_preprocessed_index(sequence, 4), 4)
    
    def test_with_dashes(self):
        """Test sequence with dashes that should be removed"""
        sequence = "AB-CD-E"
        self.assertEqual(get_preprocessed_index(sequence, 0), 0)  # A -> 0
        self.assertEqual(get_preprocessed_index(sequence, 1), 1)  # B -> 1
        self.assertEqual(get_preprocessed_index(sequence, 2), None)  # - -> None
        self.assertEqual(get_preprocessed_index(sequence, 3), 2)  # C -> 2
        self.assertEqual(get_preprocessed_index(sequence, 6), 4)  # E -> 4

    def test_truncation(self):
        """Test sequence truncation at stop codon"""
        sequence = "ABC*DE"
        self.assertEqual(get_preprocessed_index(sequence, 0), 0)  # A -> 0
        self.assertEqual(get_preprocessed_index(sequence, 2), 2)  # C -> 2
        self.assertEqual(get_preprocessed_index(sequence, 3), None)  # * -> None
        self.assertEqual(get_preprocessed_index(sequence, 4), None)  # D -> None (after *)
        
        # Test with truncation disabled
        self.assertEqual(get_preprocessed_index(sequence, 4, truncate=False), 3)

    def test_multiple_replacements(self):
        """Test sequence with multiple characters to be replaced"""
        sequence = "A.B-C.D"
        self.assertEqual(get_preprocessed_index(sequence, 0), 0)  # A -> 0
        self.assertEqual(get_preprocessed_index(sequence, 1), None)  # . -> None
        self.assertEqual(get_preprocessed_index(sequence, 2), 1)  # B -> 1
        self.assertEqual(get_preprocessed_index(sequence, 6), 3)  # D -> 3

    def test_edge_cases(self):
        """Test edge cases including empty sequences and out of bounds indices"""
        # Empty sequence
        self.assertEqual(get_preprocessed_index("", 0), None)
        
        # Out of bounds indices
        sequence = "ABCDE"
        self.assertEqual(get_preprocessed_index(sequence, -1), None)
        self.assertEqual(get_preprocessed_index(sequence, 5), None)
        
        # Single character sequence
        self.assertEqual(get_preprocessed_index("A", 0), 0)

    def test_complex_sequence(self):
        """Test a complex sequence with multiple features"""
        sequence = "AB-C.D*EFG"
        self.assertEqual(get_preprocessed_index(sequence, 0), 0)  # A -> 0
        self.assertEqual(get_preprocessed_index(sequence, 1), 1)  # B -> 1
        self.assertEqual(get_preprocessed_index(sequence, 2), None)  # - -> None
        self.assertEqual(get_preprocessed_index(sequence, 3), 2)  # C -> 2
        self.assertEqual(get_preprocessed_index(sequence, 4), None)  # . -> None
        self.assertEqual(get_preprocessed_index(sequence, 5), 3)  # D -> 3
        self.assertEqual(get_preprocessed_index(sequence, 6), None)  # * -> None
        self.assertEqual(get_preprocessed_index(sequence, 7), None)  # E -> None (after *)

    def test_custom_stop_str(self):
        """Test using a custom stop string"""
        sequence = "ABC#DE"
        self.assertEqual(get_preprocessed_index(sequence, 4, stop_str='#'), None)
        self.assertEqual(get_preprocessed_index(sequence, 4, stop_str='$'), 4)

    def test_custom_replace_chars(self):
        """Test using custom replacement characters"""
        sequence = "A#B@C"
        self.assertEqual(get_preprocessed_index(sequence, 1, replace=['#', '@']), None)
        self.assertEqual(get_preprocessed_index(sequence, 1, replace=[]), 1)

    def test_return_map(self):
        """Test returning the full position mapping"""
        sequence = "AB-C*DE"
        expected_map = {0: 0, 1: 1, 3: 2}  # Only positions before * are mapped
        self.assertEqual(get_preprocessed_index(sequence, 0, return_map=True), expected_map)

if __name__ == '__main__':
    unittest.main() 
    