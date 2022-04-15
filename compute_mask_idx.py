import torch
import numpy as np
from typing import Tuple, Optional

def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.
    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )
   # print((attention_mask*-1 + 1).sum())

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=np.bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask

# def _compute_mask_indices(
#     shape: Tuple[int, int],
#     mask_prob: float,
#     mask_length: int,
#     device: torch.device,
#     attention_mask: Optional[torch.tensor] = None,
#     min_masks: int = 0,
# ) -> torch.tensor:
#     """
#     Computes random mask spans for a given shape. Used to implement `SpecAugment: A Simple Data Augmentation Method for
#     ASR <https://arxiv.org/abs/1904.08779>`__.

#     Args:
#         shape: the the shape for which to compute masks.
#             should be of size 2 where first element is batch size and 2nd is timesteps
#         mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
#             number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
#             however due to overlaps, the actual number will be smaller (unless no_overlap is True)
#         mask_length: size of the mask
#         min_masks: minimum number of masked spans

#     """
#     batch_size, sequence_length = shape
#     attention_mask_length = attention_mask.sum(1)

#     # compute number of masked spans in batch (span 개수: 8개)
#     epsilon = np.random.rand(1).item()
#     num_masked_spans = int(mask_prob * sequence_length / mask_length + epsilon)
#     num_masked_spans = max(num_masked_spans, min_masks)

#     # make sure num masked indices <= sequence_length
#     if num_masked_spans * mask_length > sequence_length:
#         num_masked_spans = sequence_length // mask_length

#     # make sure num_masked span is also <= input_length - (mask_length - 1)
#     if sequence_length - (mask_length - 1) < num_masked_spans:
#         num_masked_spans = max(sequence_length - (mask_length - 1), 0)

#     # SpecAugment mask to fill
#     spec_aug_mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

#     # uniform distribution to sample from, make sure that offset samples are < sequence_length
#     uniform_dist = torch.ones((batch_size, sequence_length - (mask_length + attention_mask_length - 1)), device=device) # 가중치 동일하게 만들기

#     # get random indices to mask
#     spec_aug_mask_idxs = torch.multinomial(uniform_dist, num_masked_spans) # 같은 확률로 span개수만큼 뽑기(비복원)

#     # expand masked indices to masked spans
#     spec_aug_mask_idxs = (
#         spec_aug_mask_idxs.unsqueeze(dim=-1) # b,spans,1
#         .expand((batch_size, num_masked_spans, mask_length))
#         .reshape(batch_size, num_masked_spans * mask_length) # ex) spec_aug_mask_idxs[0] = [1111111111777777777..]
#     )
#     offsets = (
#         torch.arange(mask_length, device=device)[None, None, :]
#         .expand((batch_size, num_masked_spans, mask_length))
#         .reshape(batch_size, num_masked_spans * mask_length)
#     )
#     spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

#     # scatter indices to mask
#     spec_aug_mask = spec_aug_mask.scatter(1, spec_aug_mask_idxs, True) # 기존 텐서의 인덱스에 true값 주고 아니면 false

#     return spec_aug_mask # true/false로 이루어진 b,l