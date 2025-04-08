import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vep_metrics import get_mutation_index, get_mutation_indices
from src.biopython import as_msa

class TestMutationIndex(unittest.TestCase):
    def test_get_mutation_index_basic(self):
        """Test basic mutation index calculation with a simple sequence"""
        sequence = "MALWMRLLPLLALLALWGPDPAAA"
        pos = 195
        idx = get_mutation_index(sequence, pos)
        self.assertEqual(idx, 194)  # 195 - 1 (offset_idx)

    def test_get_mutation_index_msa(self):
        """Test mutation index calculation with MSA sequence"""
        # Create a simple MSA with gaps
        msa = as_msa([
            "MALWMRLLPLLALLALWGPDPAAA",  # Reference sequence
            "MALWMRLLP-LLALLALWGPDPAAA"  # Query sequence with gap
        ])
        pos = 195
        idx = get_mutation_index(msa, pos)
        self.assertEqual(idx, 194)  # Should handle gaps correctly

    def test_get_mutation_index_different_offset(self):
        """Test mutation index calculation with different offset"""
        sequence = "MALWMRLLPLLALLALWGPDPAAA"
        pos = 195
        idx = get_mutation_index(sequence, pos, offset_idx=0)
        self.assertEqual(idx, 195)  # 195 - 0

    def test_get_mutation_index_different_ref_query(self):
        """Test mutation index calculation with different ref and query indices"""
        msa = as_msa([
            "MALWMRLLPLLALLALWGPDPAAA",  # Reference sequence
            "MALWMRLLP-LLALLALWGPDPAAA",  # Query sequence with gap
            "MALWMRLLPLLALLALWGPDPAAA"   # Another sequence
        ])
        pos = 195
        idx = get_mutation_index(msa, pos, ref=1, query=2)
        self.assertEqual(idx, 194)

    def test_get_mutation_indices_basic(self):
        """Test getting mutation indices for a dataframe"""
        # Create test dataframe
        df = pd.DataFrame({
            'mutant': ['G195S', 'A196T']
        })
        sequence = "MALWMRLLPLLALLALWGPDPAAA"
        
        # Get mutation indices
        indices = get_mutation_indices(df, sequence)
        self.assertEqual(indices, [194, 195])  # 195-1, 196-1

    def test_get_mutation_indices_msa(self):
        """Test getting mutation indices for a dataframe with MSA sequence"""
        # Create test dataframe
        df = pd.DataFrame({
            'mutant': ['G195S', 'A196T']
        })
        # Create MSA with gaps
        msa = as_msa([
            "MALWMRLLPLLALLALWGPDPAAA",  # Reference sequence
            "MALWMRLLP-LLALLALWGPDPAAA"  # Query sequence with gap
        ])
        
        # Get mutation indices
        indices = get_mutation_indices(df, msa)
        self.assertEqual(indices, [194, 195])  # Should handle gaps correctly

    def test_get_mutation_indices_different_column(self):
        """Test getting mutation indices with different column name"""
        # Create test dataframe with different column name
        df = pd.DataFrame({
            'mutation': ['G195S', 'A196T']
        })
        sequence = "MALWMRLLPLLALLALWGPDPAAA"
        
        # Get mutation indices
        indices = get_mutation_indices(df, sequence, mutation_col='mutation')
        self.assertEqual(indices, [194, 195])

    def test_get_mutation_indices_different_offset(self):
        """Test getting mutation indices with different offset"""
        # Create test dataframe
        df = pd.DataFrame({
            'mutant': ['G195S', 'A196T']
        })
        sequence = "MALWMRLLPLLALLALWGPDPAAA"
        
        # Get mutation indices with different offset
        indices = get_mutation_indices(df, sequence, offset_idx=0)
        self.assertEqual(indices, [195, 196])

    def test_get_mutation_indices_empty_df(self):
        """Test getting mutation indices with empty dataframe"""
        # Create empty dataframe
        df = pd.DataFrame({
            'mutant': []
        })
        sequence = "MALWMRLLPLLALLALWGPDPAAA"
        
        # Get mutation indices
        indices = get_mutation_indices(df, sequence)
        self.assertEqual(indices, [])

if __name__ == '__main__':
    unittest.main() 