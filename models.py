import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from typing import Optional, Tuple
from argparse import ArgumentParser
import numpy as np
from vector_quantizer import Wav2Vec2GumbelVectorQuantizer
from compute_mask_idx import _compute_mask_indices

'''
>>> torch.Size([32])
    # 1d: [batch_size] 
    # use for target labels or predictions.
>>> torch.Size([12, 256])
    # 2d: [batch_size, num_features (aka: C * H * W)]
    # use for nn.Linear() input.
>>> torch.Size([10, 1, 2048])
    # 3d: [batch_size, channels, num_features (aka: H * W)]
    # when used as nn.Conv1d() input.
    # (but [seq_len, batch_size, num_features]
    # if feeding an RNN).
>>> torch.Size([16, 3, 28, 28])
    # 4d: [batch_size, channels, height, width]
    # use for nn.Conv2d() input.
>>>  torch.Size([32, 1, 5, 15, 15])
    # 5D: [batch_size, channels, depth, height, width]
    # use for nn.Conv3d() input.


'''

class Music2VecModel(Module):
    def __init__(self, out_features=768, num_layers=12):
        super().__init__()
        # input: (22050*12.8) -> 1280
        # wavebeat의 channel 늘리는 것과 wav2vec2.0에서 channel 유지시키는 것을 mix했고 2019 tcn 모델의 dilated 부분을 참고했다
        # dilated는 128까지
        # cnn block 개수는 wav2vec와 동일함
        # NOTE!!!! 마지막 block에서 padding을 하나 줄임 (1283 -> 1280)
        # out_features=768는 transformer의 embedding size
        # todo: change stride, from 5?
        shapes = [
            (32, 10, 11),
            (64, 3, 5),
            (128, 3, 2),
            (256, 3, 2),
            (256, 3, 1),
            (512, 2, 1),
            (512, 2, 1)
        ]
        
        # Expected values are False for Base (12 layers) and True for Large arch (24 layers).
        self.layer_norm_first = num_layers > 12
        
        # Expected values are 12 for Base and 16 for Large arch.
        self.num_heads = 16 if num_layers > 12 else 12

        # Expected values are 0.1 for Base and 0.0 for Large arch.
        self.attention_dropout = 0.0 if num_layers > 12 else 0.1

        # Expected values are 3072 for Base and 4096 for Large arch.
        self.ff_interm_features = 4096 if num_layers > 12 else 3072
        
        # Expected values are 0.1 for both Base and Large arch.
        self.ff_interm_dropout = 0.1

        # Expected values are 0.1 for Base and 0.0 for Large arch.
        self.ff_dropout = 0.0 if num_layers > 12 else 0.1

        # Expected values are 0.1 for both Base and Large arch.
        self.transformer_layer_drop = 0.1

        blocks = nn.ModuleList()
        in_channels = 1

        for i, (out_channels, kernel_size, stride) in enumerate(shapes):
            # 2^7 = 128
            dilation = 2 ** i
            pad_value = ((kernel_size - 1) * dilation) // 2
            if i == len(shapes) - 1:
                pad_value -= 1

            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )

            blocks.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=pad_value,
                    bias=False,
                )
            )
            in_channels = out_channels

        # initialize
        #self.feature_extractor = FeatureExtractor(nn.ModuleList(blocks))
        self.blocks = blocks
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

        #########################

        in_features = out_channels

        self.projection_layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features,)
        self.projection_dropout = nn.Dropout(0.1)

        # b, 1283, 768

        embedding_kernel_size = 128
        self.conv = nn.Conv1d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=embedding_kernel_size,
            padding=embedding_kernel_size // 2,
            groups=16,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2) # weight norm
        self.num_remove2: int = 1 if embedding_kernel_size % 2 == 0 else 0

        #########################

        self.encoder_layers = nn.ModuleList()
        self.attention_layer_norm = nn.LayerNorm(out_features)
        self.transformer_dropout = nn.Dropout(self.ff_dropout)

        for _ in range(num_layers):
            attention = SelfAttention(
                embed_dim=out_features,
                num_heads=self.num_heads,
                dropout=self.attention_dropout,
            ) #
            feed_forward = FeedForward(
                io_features=out_features,
                intermediate_features=self.ff_interm_features,
                intermediate_dropout=self.ff_interm_dropout,
                output_dropout=self.ff_dropout,
            )
            self.encoder_layers.append(
                EncoderLayer(
                    attention=attention,
                    dropout=self.ff_dropout,
                    layer_norm_first=self.layer_norm_first,
                    feed_forward=feed_forward,
                )
            )

        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())

        ### pretrain###
        self.quantizer = Wav2Vec2GumbelVectorQuantizer()
        self.project_q = nn.Linear(256, 256) # from codebook to compare
        self.project_hid = nn.Linear(768, 256) # from c to compare

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        batch_size, sequence_length, hidden_size = hidden_states.size()
        mask_time_length = 10
        mask_time_prob = 0.065
        mask_time_indices = _compute_mask_indices(
            (batch_size, sequence_length),
            mask_prob=mask_time_prob,
            mask_length=mask_time_length,
            device=hidden_states.device,
            attention_mask=attention_mask,
            min_masks=2,
        )
        hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        return hidden_states,mask_time_indices

    def set_gumbel_temperature(self, temperature: int):
        return self.quantizer.set_temperature(temperature)

    def _init_weights(self, module):
        if isinstance(module, Wav2Vec2GumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight.data)

        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def _sample_negatives(
        features: torch.FloatTensor, num_negatives: int, attention_mask: Optional[torch.LongTensor] = None
    ):
        batch_size, sequence_length, hidden_size = features.shape
        if sequence_length <= 1:
            raise ValueError(
                f"`features should have `sequence_length` > 1, but are of shape (batch_size, sequence_length, hidden_size) = ({batch_size, sequence_length, hidden_size})."
            )

        features = features.view(-1, hidden_size)  # B,l,C => (B*l),C

        with torch.no_grad():

            sampled_negative_indices = []
            for batch_idx in range(batch_size):
                high = attention_mask[batch_idx].sum() - 1 if attention_mask is not None else sequence_length - 1
                sampled_indices_slice = torch.randint(
                    0, high, size=(num_negatives * sequence_length,), device=features.device
                )
                sampled_negative_indices.append(sampled_indices_slice)

            sampled_negative_indices = torch.stack(sampled_negative_indices)



            feature_indices = (
                torch.arange(sequence_length, device=features.device)[:, None]
                .expand(sequence_length, num_negatives)
                .flatten()
            )


            sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1

        for batch_idx in range(1, batch_size):
            sampled_negative_indices[batch_idx] += batch_idx * sequence_length

        sampled_negatives = features[sampled_negative_indices.view(-1)]
        sampled_negatives = sampled_negatives.view(batch_size, sequence_length, num_negatives, hidden_size)
        sampled_negatives = sampled_negatives.permute(2, 0, 1, 3)

        return sampled_negatives     # K,b,l,256

    @staticmethod
    def compute_contrastive_logits(
            target_features: torch.FloatTensor, # 1,b,l,256
            negative_features: torch.FloatTensor,
            predicted_features: torch.FloatTensor,  # b,l,256
            temperature=1.0,
    ):
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        logits = logits / temperature # 첫번째는 유사도가 높아야함
        return logits

    def calculate_loss(self, waveforms: Tensor) -> Tensor:
        extract_x = self.sample_to_tcn(waveforms)
        transformer_x = self.tcn_to_transformer(extract_x)

        hidden_states, mask_time_indices = self._mask_hidden_states(transformer_x) # B, l (T/F)
        encoder_outputs = self.transformer(hidden_states, None)

        transformer_features = self.project_hid(encoder_outputs)

        quantized_features, codevector_perplexity = self.quantizer(extract_x, mask_time_indices)
        quantized_features = self.project_q(quantized_features) # z->q(b,l,256)

        num_negatives = 100
        negative_quantized_features = self._sample_negatives(
            quantized_features, num_negatives, attention_mask=None
        )

        logits = self.compute_contrastive_logits(
            quantized_features[None, :],
            negative_quantized_features,
            transformer_features,
            0.1)

        neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf") # k,b,l

        preds = logits.transpose(0, 2).reshape(-1, logits.size(0))

        target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
        contrastive_loss = nn.functional.cross_entropy(preds.float(), target, reduction="sum")
        
        num_codevectors_per_group = 320
        num_codevector_groups = 2
        diversity_loss_weight = 0.1
        num_codevectors = num_codevectors_per_group * num_codevector_groups
        diversity_loss = (num_codevectors - codevector_perplexity) / num_codevectors
        print(contrastive_loss, diversity_loss_weight * diversity_loss)
        loss = contrastive_loss + diversity_loss_weight * diversity_loss

        return loss

    def sample_to_tcn(self, x):
        #########################
        # Pre-TCN
        # torch.Size([2, 1, 282240]) (batch, feature, audio sample)
        print("Pre-TCN", x.shape)

        # 마지막 block에서 padding을 하나 줄임 (1283 -> 1280)
        for block in self.blocks:
            x = block(x)

        # batch size, channel, sequence length
        # 마지막 block에서 padding을 하나 줄임 (1283 -> 1280)
        if self.num_remove > 0:
            x = x[..., :-self.num_remove]

        x = x.transpose(-2, -1)

        return x

    def tcn_to_transformer(self, x):
        #########################
        # Projection (TCN -> Transformer)
        # torch.Size([2, 1280, 512]) (batch size, sequence length, channel)
        print("Projection (TCN -> Transformer)", x.shape)

        x = self.projection_layer_norm(x)
        x = self.projection(x)
        x = self.projection_dropout(x)

        return x

    def transformer_embedding(self, x):
        #########################
        # Transformer embedding
        # torch.Size([2, 1280, 768])
        print("Transformer embedding", x.shape)

        embedded_x = x.transpose(-2, -1)
        embedded_x = self.conv(embedded_x)
        if self.num_remove2 > 0:
            embedded_x = embedded_x[..., :-self.num_remove2]
        embedded_x = torch.nn.functional.gelu(embedded_x)
        embedded_x = embedded_x.transpose(-2, -1)

        x = x + embedded_x

        return x

    def generate_mask(self, x, lengths):
        #########################
        # Mask generation
        print("Mask generation", x.shape)

        attention_mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # create mask for padded elements and zero-out them
            attention_mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[attention_mask] = 0.0
            # extend the mask to attention shape and set weight
            attention_mask = -10000.0 * attention_mask[:, None, None, :].to(dtype=features.dtype)
            attention_mask = attention_mask.expand(batch_size, 1, max_len, max_len)

        return attention_mask

    def transformer(self, x, attention_mask):
        #########################
        # Transformer
        # torch.Size([2, 1280, 768])
        print("Transformer", x.shape)

        if not self.layer_norm_first:
            x = self.attention_layer_norm(x)

        x = self.transformer_dropout(x)
        for layer in self.encoder_layers:
            if not (self.training and torch.rand(1).item() <= self.transformer_layer_drop):
                x = layer(x, attention_mask)

        return x

    def forward(self, x, lengths=None):
        x = self.sample_to_tcn(x)
        x = self.tcn_to_transformer(x)
        x = self.transformer_embedding(x)
        attention_mask = self.generate_mask(x, lengths)
        x = self.transformer(x, attention_mask)

        #########################
        # Result
        # torch.Size([2, 1280, 768])
        print("Result", x.shape)

        return x

class MusicDetectionModel(Module):
    def __init__(self, detect_out=256, out_features=768, num_layers=12):
        super().__init__()
        # wavebeat의 channel 늘리는 것과 wav2vec2.0에서 channel 유지시키는 것을 mix했고 2019 tcn 모델의 dilated 부분을 참고했다
        # dilated는 128까지
        # cnn block 개수는 wav2vec와 동일함
        # NOTE!!!! 마지막 block에서 padding을 하나 줄임 (1283 -> 1280)
        # out_features=768는 transformer의 embedding size
        shapes = [
            (32, 10, 11),
            (64, 3, 5),
            (128, 3, 2),
            (256, 3, 2),
            (256, 3, 1),
            (512, 2, 1),
            (512, 2, 1)
        ]
        
        # Expected values are False for Base (12 layers) and True for Large arch (24 layers).
        self.layer_norm_first = num_layers > 12
        
        # Expected values are 12 for Base and 16 for Large arch.
        self.num_heads = 16 if num_layers > 12 else 12

        # Expected values are 0.1 for Base and 0.0 for Large arch.
        self.attention_dropout = 0.0 if num_layers > 12 else 0.1

        # Expected values are 3072 for Base and 4096 for Large arch.
        self.ff_interm_features = 4096 if num_layers > 12 else 3072
        
        # Expected values are 0.1 for both Base and Large arch.
        self.ff_interm_dropout = 0.1

        # Expected values are 0.1 for Base and 0.0 for Large arch.
        self.ff_dropout = 0.0 if num_layers > 12 else 0.1

        # Expected values are 0.1 for both Base and Large arch.
        self.transformer_layer_drop = 0.1

        blocks = nn.ModuleList()
        in_channels = 1

        for i, (out_channels, kernel_size, stride) in enumerate(shapes):
            # 2^7 = 128
            dilation = 2 ** i
            pad_value = ((kernel_size - 1) * dilation) // 2
            if i == len(shapes) - 1:
                pad_value -= 1

            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )

            blocks.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=pad_value,
                    bias=False,
                )
            )
            in_channels = out_channels

        # initialize
        #self.feature_extractor = FeatureExtractor(nn.ModuleList(blocks))
        self.blocks = blocks
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

        #########################

        in_features = out_channels

        self.projection_layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features,)
        self.projection_dropout = nn.Dropout(0.1)

        # b, 1283, 768

        embedding_kernel_size = 128
        self.conv = nn.Conv1d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=embedding_kernel_size,
            padding=embedding_kernel_size // 2,
            groups=16,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2) # weight norm
        self.num_remove2: int = 1 if embedding_kernel_size % 2 == 0 else 0

        #########################

        self.encoder_layers = nn.ModuleList()
        self.attention_layer_norm = nn.LayerNorm(out_features)
        self.transformer_dropout = nn.Dropout(self.ff_dropout)

        for _ in range(num_layers):
            attention = SelfAttention(
                embed_dim=out_features,
                num_heads=self.num_heads,
                dropout=self.attention_dropout,
            ) #
            feed_forward = FeedForward(
                io_features=out_features,
                intermediate_features=self.ff_interm_features,
                intermediate_dropout=self.ff_interm_dropout,
                output_dropout=self.ff_dropout,
            )
            self.encoder_layers.append(
                EncoderLayer(
                    attention=attention,
                    dropout=self.ff_dropout,
                    layer_norm_first=self.layer_norm_first,
                    feed_forward=feed_forward,
                )
            )

        #########################

        detect_in = out_features
        self.detect_project = nn.Linear(
            in_features=detect_in,
            out_features=detect_out,
        )

    def forward(self, x, lengths=None):
        #########################
        # Pre-TCN
        # torch.Size([2, 1, 282240]) (batch, feature, audio sample)
        print("Pre-TCN", x.shape)

        # 마지막 block에서 padding을 하나 줄임 (1283 -> 1280)
        for block in self.blocks:
            x = block(x)

        # batch size, channel, sequence length
        # 마지막 block에서 padding을 하나 줄임 (1283 -> 1280)
        if self.num_remove > 0:
            x = x[..., :-self.num_remove]

        x = x.transpose(-2, -1)

        #########################
        # Projection (TCN -> Transformer)
        # torch.Size([2, 1280, 512]) (batch size, sequence length, channel)
        print("Projection (TCN -> Transformer)", x.shape)

        x = self.projection_layer_norm(x)
        x = self.projection(x)
        x = self.projection_dropout(x)

        #########################
        # Transformer embedding
        # torch.Size([2, 1280, 768])
        print("Transformer embedding", x.shape)

        embedded_x = x.transpose(-2, -1)
        embedded_x = self.conv(embedded_x)
        if self.num_remove2 > 0:
            embedded_x = embedded_x[..., :-self.num_remove2]
        embedded_x = torch.nn.functional.gelu(embedded_x)
        embedded_x = embedded_x.transpose(-2, -1)

        x = x + embedded_x

        #########################
        # Mask generation
        print("Mask generation", x.shape)

        attention_mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # create mask for padded elements and zero-out them
            attention_mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[attention_mask] = 0.0
            # extend the mask to attention shape and set weight
            attention_mask = -10000.0 * attention_mask[:, None, None, :].to(dtype=features.dtype)
            attention_mask = attention_mask.expand(batch_size, 1, max_len, max_len)

        #########################
        # Transformer
        # torch.Size([2, 1280, 768])
        print("Transformer", x.shape)

        if not self.layer_norm_first:
            x = self.attention_layer_norm(x)

        x = self.transformer_dropout(x)
        for layer in self.encoder_layers:
            if not (self.training and torch.rand(1).item() <= self.transformer_layer_drop):
                x = layer(x, attention_mask)

        #########################
        # Projection layer (Music2Vec -> Object Detection)
        # torch.Size([2, 1280, 768])
        print("Projection layer (Music2Vec -> Object Detection)", x.shape)

        x = self.detect_project(x)

        #########################
        # Result
        # torch.Size([2, 1280, 256])
        print("Result", x.shape)

        return x

class LayerNorm(nn.LayerNorm):
    """Layer norm with transpose"""
    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(-2, -1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)
        return x


class ConvLayerBlock(Module):
    """Convolution unit of FeatureExtractor"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            bias: bool,
            layer_norm: Optional[Module],
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(
            self,
            x: Tensor,
            length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
            length (Tensor, optional): Shape ``[batch, ]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = nn.functional.gelu(x)

        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode='floor') + 1
            # When input length is 0, the resulting length can be negative. So fix it here.
            length = torch.max(torch.zeros_like(length), length)
        return x, length


class FeatureExtractor(Module):
    """Extract features from audio

    Args:
        conv_layers (nn.ModuleList):
            convolution layers
    """
    def __init__(
            self,
            conv_layers: nn.ModuleList,
    ):
        super().__init__()
        self.conv_layers = conv_layers

    def forward(
            self,
            x: Tensor,
            length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor):
                Input Tensor representing a batch of audio,
                shape: ``[batch, time]``.
            length (Tensor, optional):
                Valid length of each input sample. shape: ``[batch, ]``.

        Returns:
            Tensor:
                The resulting feature, shape: ``[batch, frame, feature]``
            Optional[Tensor]:
                Valid length of each output sample. shape: ``[batch, ]``.
        """
        if x.ndim != 2:
            raise ValueError(
                "Expected the input Tensor to be 2D (batch, time), "
                "but received {list(x.shape)}")

        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.conv_layers:
            x, length = layer(x, length)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        return x, length

class ConvolutionalPositionalEmbedding(Module):
    """Positional embedding which is placed at the beginning of Transformer.

    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
    """
    def __init__(
            self,
            embed_dim: int,
            kernel_size: int,
            groups: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2) # weight norm
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def __prepare_scriptable__(self):
        for hook in self.conv._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                    hook.__module__ == 'torch.nn.utils.weight_norm' and
                    hook.__class__.__name__ == 'WeightNorm'
            ):
                _LG.warning('Removing weight_norm from %s', self.__class__.__name__)
                torch.nn.utils.remove_weight_norm(self.conv)
        return self

    def forward(self, x):
        """
        Args:
            x (Tensor): shape ``[batch, frame, feature]``.

        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., :-self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x


class SelfAttention(Module):
    """Multihead Self Attention module

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probabiliry on attn_output_weights. Default: ``0.0``
    """
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads # 동일한 문장도 8명(8 heads)이 각각의 관점에서 보고 추후에 합치는 과정
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim

        self.scaling = self.head_dim ** -0.5 # QK/root(d)의 root(d)

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``

        Returns:
            Tensor: The resulting tensor. shape: ``[batch, sequence_length, embed_dim]``
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). "
                f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(
                    f"The expected attention mask shape is {shape_}. "
                    f"Found {attention_mask.size()}."
                )

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        ## weights 계산
        weights = self.scaling * (q @ k)  # B, nH, L, L
        if attention_mask is not None:
            weights += attention_mask

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = torch.nn.functional.dropout(weights, p=self.dropout, training=self.training)
        ## V마다 가중치 부여
        output = weights @ v  # B, nH, L, Hd
        # multi-head로 쪼갠 것 다시 합치기기
        output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)

        output = self.out_proj(output)
        return output


class FeedForward(Module):
    """Layer that follows attention layer in encoder layer.
    """
    def __init__(
            self,
            io_features: int,
            intermediate_features: int,
            intermediate_dropout: float,
            output_dropout: float,
    ):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        '''
        Multi Head Attention에서 각 head가 자신의 관점으로만 문장을 Self-Attention 하게 된다면 각 head에 따라 Attention이 치우쳐질 것
        PoswiseFeedForwardNet은 각 head가 만들어낸 Self-Attention을 치우치지 않게 균등하게 섞는 역할
        '''
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: ``(batch, sequence_length, io_features)``
        Returns:
            x (Tensor): shape: ``(batch, sequence_length, io_features)``
        """
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x


class EncoderLayer(Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward.
    """
    def __init__(
            self,
            attention: Module,
            dropout: float,
            layer_norm_first: bool,
            feed_forward: Module,
    ):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(attention.embed_dim)

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): shape: ``(batch, sequence_length, embed_dim)``
            attention_mask (Tensor, optional):
                shape: ``(batch, 1, sequence_length, sequence_length)``
        """
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x


class Transformer(Module):
    def __init__(
            self,
            pos_conv_embed: Module,
            dropout: float,
            layers: Module,
            layer_norm_first: bool,
            layer_drop: float,
    ):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
    ):
        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x = layer(x, attention_mask)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        return x


class Encoder(Module):
    def __init__(
            self,
            # feature_projection: Module,
            transformer: Module,
            readout: Module,
    ):
        super().__init__()
        # self.feature_projection = feature_projection
        self.transformer = transformer
        self.readout = readout

    def forward(
            self,
            features: Tensor,
            lengths: Optional[Tensor] = None,
    ) -> Tensor:
        # x = self.feature_projection(features)
        x = features

        mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # create mask for padded elements and zero-out them
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[mask] = 0.0
            # extend the mask to attention shape and set weight
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)

        x = self.transformer(x, attention_mask=mask)
        x = self.readout(x)
        return x

def get_activation(act_type,ch=None):
    """ Helper function to construct activation functions by a string.
    Args:
        act_type (str): One of 'ReLU', 'PReLU', 'SELU', 'ELU'.
        ch (int, optional): Number of channels to use for PReLU.

    Returns:
        torch.nn.Module activation function.
    """

    if act_type == "PReLU":
        return torch.nn.PReLU(ch)
    elif act_type == "ReLU":
        return torch.nn.ReLU()
    elif act_type == "SELU":
        return torch.nn.SELU()
    elif act_type == "ELU":
        return torch.nn.ELU()

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=1, num_classes=2, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv1d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        out1 = out.permute(0, 2, 1)

        batch_size, length, channels = out1.shape

        out2 = out1.view(batch_size, length, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=1, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv1d(feature_size, num_anchors * 1, kernel_size=3, padding=1)

    def forward(self, x):
        print("regression forward", x.shape)
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        out = out.permute(0, 2, 1)

        return out.contiguous().view(out.shape[0], -1, 1)
