import torch
import random
import numpy as np
import numpy as np
import torch.nn.functional as F

from typing import Optional, Tuple, List
from torch import Tensor, nn
from torch.nn import Module

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


class LayerNorm(nn.LayerNorm):
    """Layer norm with transpose"""
    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(-2, -1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)
        return x



class ConvolutionalPositionalEmbedding(Module):
    """Positional embedding which is placed at the beginning of Transformer.
    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
    """
    def __init__(
            self,
            kernel_size: int,
            groups: int,
            embed_dim: int=256,
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
            layer: Module,
            layer_norm_first: bool,
            layer_drop: float,
            num_layers: int=12,
    ):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layer = layer
        self.num_layers = num_layers

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
    ):
        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        for _ in range(self.num_layers):
            x = self.layer(x, attention_mask)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        return x


class Encoder(Module):
    def __init__(
            self,
            # feature_projection: Module,
            transformer: Module,
    ):
        super().__init__()
        # self.feature_projection = feature_projection
        self.transformer = transformer

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
        return x




def get_encoder(
        embed_dim: int,
        pos_conv_kernel: int,
        pos_conv_groups: int,
        num_heads: int,
        attention_dropout: float,
        ff_interm_features: int,
        ff_interm_dropout: float,
        dropout: float,
        layer_norm_first: bool,
        layer_drop: float,
        num_out: int,
) -> Encoder:
    """
    Args:
        embed_dim (int):
            The dimension of embedding.
            This option corresponds to "encoder_embed_dim" from fairseq.
            Expected values are 768 for Base arch, and 1024 for Large arch.
        dropout_input (float):
            The dropout probability applied after the input feature is projected
            to ``embed_dim``.
            This option corresponds to "dropout_input" from fairseq.
            Expected values are 0.1 for both Base and Large arch.
        pos_conv_kernel (int):
            The kernel size of convolutional positional embeddings.
            This option corresponds to "conv_pos" from fairseq.
            Expected values are 128 for both Base and Large arch.
        pos_conv_groups (int):
            The number of groups of convolutional positional embeddings.
            This option corresponds to "conv_pos_groups" from fairseq.
            Expected values are 16 for both Base and Large arch.
        num_heads (int):
            The number of heads in self attention layers.
            This option corresponds to "encoder_attention_heads" from fairseq.
            Expected values are 12 for Base and 16 for Large arch.
        attention_dropout (float):
            The dropout probability applied after softmax in self-attention layer.
            This option corresponds to "attention_dropout" from fairseq.
            Expected values are 0.1 for Base and 0.0 for Large arch.
        ff_interm_features (int):
            The dimension of hidden features in feed forward layer.
            This option corresponds to "encoder_ffn_embed_dim" from fairseq.
            Expected values are 3072 for Base and 4096 for Large arch.
        ff_interm_dropout (float):
            The dropout probability applied in feedforward layer.
            This option correspinds to "activation_dropout" from fairseq.
            Expected values are 0.1 for both Base and Large arch.
        dropout (float):
            The dropout probability applied at the end of feed forward layer.
            This option corresponds to "dropout" from fairseq.
            Expected values are 0.1 for Base and 0.0 for Large arch.
        layer_norm_first (bool):
            Control the order of layer norm in transformer layer and each encoder layer.
            If True, in transformer layer, layer norm is applied before features are fed
            to encoder layers. In encoder layer, two layer norms are applied before and after
            self attention.
            If False, in transformer layer, layer norm is applied after features are fed
            to encoder layers. In encoder layer, two layer norms are applied after self
            attention, before and after feed forward.
            This option corresponds to "layer_norm_first" from fairseq.
            Expected values are False for Base and True for Large arch.
        layer_drop (float):
            Probability to drop each encoder layer during training.
            This option corresponds to "layerdrop" from fairseq.
            Expected values are 0.1 for both Base and Large arch.
        num_out (int):
            The dimension of the output. The number of labels.
    """
    # feature_projection = FeatureProjection(in_features, embed_dim, dropout_input) # convolution blocks to transformer
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups) #CPE : 768

    
    attention = SelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=attention_dropout,
    ) #
    feed_forward = FeedForward(
        io_features=embed_dim,
        intermediate_features=ff_interm_features,
        intermediate_dropout=ff_interm_dropout,
        output_dropout=dropout,
    )
    encoderlayer = EncoderLayer(
            attention=attention,
            dropout=dropout,
            layer_norm_first=layer_norm_first,
            feed_forward=feed_forward,
        )

    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layer=encoderlayer,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop, #transformer layer 통째로 드롭
    )

    return Encoder(transformer)





