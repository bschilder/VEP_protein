import src.biopython as bp


def seq_to_data(sequence,
                name="protein1",
                **kwargs):
    
    """Convert a sequence to a batch of data.
    
    Args:
        sequence: A single sequence string to convert.
        name: The name of the protein.
        kwargs: Additional keyword arguments.
    """
    data = [
        (name, bp.preprocess_sequence(sequence, **kwargs)),
    ]
    return data

def tokenize_batch_esm2(sequence,
                        alphabet,
                        **kwargs):
    """Convert a sequence to a batch of data.
    
    Args:
        sequence: The sequence to convert.
        alphabet: The alphabet to use.
    Returns:
        batch_labels: The labels of the batch.
        batch_strs: The strings of the batch.
        batch_tokens: The tokens of the batch.
    """
    data = seq_to_data(sequence, **kwargs)
    # Get the batch converter
    batch_converter = alphabet.get_batch_converter()
        
    # Convert the data to a batch
    (batch_labels, 
        batch_strs, 
        batch_tokens) = batch_converter(data)

    return (batch_labels, batch_strs, batch_tokens) 

         
def get_autotokenizer(model_name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


