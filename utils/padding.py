import numpy as np

def pad(x, max_length, sr):
    """
    Pad inputs (on left/right and up to predefined length or max length in the batch)
    Args:
        x: input audio sequence (1, sample 개수)
        max_length: maximum length of the returned list and optionally padding length
        sr: sample rate
    """

    num_samples = x.shape[1]
    target_samples = max_length*sr
    attention_mask = np.ones(num_samples, dtype=np.int32)

    difference = int(target_samples - num_samples)

    attention_mask = np.pad(attention_mask, (0, difference))
    padding_shape = (0, difference)
    padded_x = np.pad(num_samples, padding_shape, "constant", constant_values=-1)

    return padded_x, attention_mask
