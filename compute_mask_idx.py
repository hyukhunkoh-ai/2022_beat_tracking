import torch
from typing import Tuple, Optional

def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    device: torch.device,
    attention_mask: Optional[torch.tensor] = None,
    min_masks: int = 0,
) -> torch.tensor:
    """
    Computes random mask spans for a given shape. Used to implement `SpecAugment: A Simple Data Augmentation Method for
    ASR <https://arxiv.org/abs/1904.08779>`__.

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans

    """
    batch_size, sequence_length = shape
    attention_mask_length = attention_mask.sum(1)

    # compute number of masked spans in batch (span 개수: 8개)
    num_masked_spans = int(mask_prob * sequence_length / (mask_length + attention_mask_length) + torch.rand((1,)).item())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // (mask_length + attention_mask_length)

    # SpecAugment mask to fill
    spec_aug_mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones((batch_size, sequence_length - (mask_length + attention_mask_length - 1)), device=device) # 가중치 동일하게 만들기

    # get random indices to mask
    spec_aug_mask_idxs = torch.multinomial(uniform_dist, num_masked_spans) # 같은 확률로 span개수만큼 뽑기(비복원)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = (
        spec_aug_mask_idxs.unsqueeze(dim=-1) # b,spans,1
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length) # ex) spec_aug_mask_idxs[0] = [1111111111777777777..]
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    spec_aug_mask = spec_aug_mask.scatter(1, spec_aug_mask_idxs, True) # 기존 텐서의 인덱스에 true값 주고 아니면 false

    return spec_aug_mask # true/false로 이루어진 b,l