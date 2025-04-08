import pytest
from src.utils import query_msa

def test_query_msa_valid_input():
    """
    Test query_msa with a valid sequence input.
    """
    # Sample input
    input_sequence = "ACDEFGHIKLMNPQRSTVWY"
    
    # Expected output (this should be based on the actual expected result)
    expected_output = {
        "alignment": [
            "ACDEFGHIKLMNPQRSTVWY",
            "AC-DEFGHIKLMNPQRSTVWY",
            "ACDEFGHIKL-MNPQRSTVWY"
        ],
        "metadata": {
            "source": "example_source",
            "sequence_id": "seq123"
        }
    }
    
    # Execute the function
    result = query_msa(input_sequence)
    
    # Assert the results
    assert result == expected_output, "The MSA query did not return the expected alignment."

def test_query_msa_empty_input():
    """
    Test query_msa with an empty string as input.
    """
    # Sample input
    input_sequence = ""
    
    # Expected to raise a ValueError
    with pytest.raises(ValueError):
        query_msa(input_sequence)

def test_query_msa_invalid_characters():
    """
    Test query_msa with invalid characters in the input sequence.
    """
    # Sample input with invalid characters
    input_sequence = "ACDEFGHIKLMNPQRSTVWYZ"  # 'Z' is typically not a standard amino acid
    
    # Expected to raise a ValueError or return None based on implementation
    with pytest.raises(ValueError):
        query_msa(input_sequence)

def test_query_msa_large_input():
    """
    Test query_msa with a large input sequence to check performance or handling.
    """
    # Sample large input
    input_sequence = "ACDEFGHIKLMNPQRSTVWY" * 1000  # Repeat the sequence to make it large
    
    # Execute the function
    result = query_msa(input_sequence)
    
    # Simple assertion to check if result is not None
    assert result is not None, "The MSA query failed to handle large input sequences."

def test_query_msa_specific_alignment():
    """
    Test query_msa with a specific sequence to verify alignment correctness.
    """
    # Sample input
    input_sequence = "ACD"
    
    # Expected output based on known alignment
    expected_output = {
        "alignment": [
            "ACD",
            "A-D",
            "AC-"
        ],
        "metadata": {
            "source": "known_test_case",
            "sequence_id": "seq_test"
        }
    }
    
    # Execute the function
    result = query_msa(input_sequence)
    
    # Assert the results
    assert result == expected_output, "The MSA alignment does not match the expected output for the test case."