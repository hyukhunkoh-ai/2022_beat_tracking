import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

class ConvLayerBlock(nn.Module):
    """Convolution unit of FeatureExtractor"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            padding: int,
            bias: bool,
            layer_norm: Optional[nn.Module],
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
            dilation=dilation,
            padding=padding,
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

class SelfAttention(nn.Module):
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

        weights = nn.functional.softmax(weights, dim=-1)
        weights = nn.functional.dropout(weights, p=self.dropout, training=self.training)
        ## V마다 가중치 부여
        output = weights @ v  # B, nH, L, Hd
        # multi-head로 쪼갠 것 다시 합치기기
        output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)

        output = self.out_proj(output)
        return output


class FeedForward(nn.Module):
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
        x = nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x


class EncoderLayer(nn.Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward.
    """
    def __init__(
            self,
            attention: nn.Module,
            dropout: float,
            layer_norm_first: bool,
            feed_forward: nn.Module,
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


class FeatureExtractor(nn.Module):
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
