import numpy as np

def pad(x, max_length=12.8):
    """
    Pad inputs (on left/right and up to predefined length or max length in the batch)
    Args:
        x: input audio sequence (1, sample 개수)
        max_length: maximum length of the returned list and optionally padding length
    """

    num_samples = x.shape[1]
    attention_mask = np.ones(num_samples, dtype=np.int32)

    difference = max_length*22050 - x
    attention_mask = np.pad(attention_mask, (0, difference))
    padding_shape = (0, difference)
    padded_x = np.pad(num_samples, padding_shape, "constant", constant_values=-1)

    return padded_x, attention_mask
