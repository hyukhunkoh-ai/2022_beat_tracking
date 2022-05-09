import torch
from torch import nn

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
    attention_mask = torch.ones(num_samples, dtype=torch.float32)

    difference = int(target_samples - num_samples)
    padding_shape = (0, difference)

    attention_mask = nn.functional.pad(attention_mask, padding_shape)
    padded_x = nn.functional.pad(x.squeeze(), padding_shape, "constant", value=0).unsqueeze(0)

    return padded_x, attention_mask
