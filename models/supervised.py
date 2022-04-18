import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from models.components import SelfAttention, FeedForward, EncoderLayer, ConvLayerBlock

class MusicDetectionModel(nn.Module):
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
                ConvLayerBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=pad_value,
                    bias=False,
                    layer_norm=normalization
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
            x = block(x, lengths)

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
        embedded_x = nn.functional.gelu(embedded_x)
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
