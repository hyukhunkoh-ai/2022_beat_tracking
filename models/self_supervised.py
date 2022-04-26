import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional
from vector_quantizer import Wav2Vec2GumbelVectorQuantizer
from compute_mask_idx import _compute_mask_indices
from models.components import ConvLayerBlock, SelfAttention, FeedForward, EncoderLayer

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _get_feature_vector_attention_mask(
    feature_vector_length: int,
    attention_mask: torch.LongTensor,
    input_lengths
):
    # Effectively attention_mask.sum(-1), but not inplace to be able to run
    # on inference mode.
    non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

    output_lengths = input_lengths

    batch_size = attention_mask.shape[0]

    attention_mask = torch.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
    )
    # these two operations makes sure that all values before the output lengths idxs are attended to
    attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return attention_mask

class Music2VecModel(nn.Module):
    def __init__(self, out_features=768, num_layers=12, sr=22050):
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

        self.sr = sr
        
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
            attention_mask=(1 - attention_mask),
            min_masks=2,
        )
        #print(hidden_states.shape, mask_time_indices.shape)
        mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
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
        attention_mask = 1 - attention_mask
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

    def calculate_loss(self, waveforms: Tensor, attention_mask=None) -> Tensor:
        # to-do: add length
        # sample_to_tcn을 feature extractor length 생성 추가
        # calculate_loss에서 length 받아야함
        # extract_features = extract_x
        # sample_to_tcn = feature_extractor
        # tcn_to_transformer = encoder (readout 뺐음)
        # 헥심: length를 처리하고 받아야함
        # sample_to_tcn, sample_to_transformer, mask_hidden_states는 반드시 length 정보가 들어가야함
        # 다 하면 학습돌릴 수 있음

        lengths = None
        if attention_mask is not None:
            lengths = attention_mask.sum(-1)

        extract_x, lengths = self.sample_to_tcn(waveforms, lengths)
        transformer_x = self.tcn_to_transformer(extract_x)

        if attention_mask is not None:
            lengths = lengths.squeeze()

            batch_size, sequence_length, _ = extract_x.size()
            attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.int32, device=device)

            mask_criterion = torch.lt(lengths, sequence_length)
            mask_criterion_sum = mask_criterion.sum()
            if mask_criterion_sum != 0:
                attention_mask[mask_criterion, lengths] = 1
                attention_mask = torch.cumsum(attention_mask, dim=-1)

        # to-do: test 10초
        hidden_states, mask_time_indices = self._mask_hidden_states(transformer_x, attention_mask=attention_mask)
        encoder_outputs = self.transformer(hidden_states, attention_mask)

        transformer_features = self.project_hid(encoder_outputs)

        quantized_features, codevector_perplexity = self.quantizer(extract_x, mask_time_indices)
        quantized_features = self.project_q(quantized_features) # z->q(b,l,256)

        num_negatives = 100
        negative_quantized_features = self._sample_negatives(
            quantized_features, num_negatives, attention_mask=attention_mask
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
        #print(contrastive_loss, diversity_loss_weight * diversity_loss)
        loss = contrastive_loss + diversity_loss_weight * diversity_loss

        return loss

    def sample_to_tcn(self, x, length):
        #length 추가

        #########################
        # Pre-TCN
        # torch.Size([2, 1, 282240]) (batch, feature, audio sample)
        #print("Pre-TCN", x.shape)

        # 마지막 block에서 padding을 하나 줄임 (1283 -> 1280)
        for block in self.blocks:
            x, length = block(x, length)

        # batch size, channel, sequence length
        # 마지막 block에서 padding을 하나 줄임 (1283 -> 1280)
        if self.num_remove > 0:
            x = x[..., :-self.num_remove]
            length = length - self.num_remove

        x = x.transpose(-2, -1)

        #length 계산 (feature extractor 참고)

        #length output 추가
        return x, length

    def tcn_to_transformer(self, x):
        #########################
        # Projection (TCN -> Transformer)
        # torch.Size([2, 1280, 512]) (batch size, sequence length, channel)
        #print("Projection (TCN -> Transformer)", x.shape)

        x = self.projection_layer_norm(x)
        x = self.projection(x)
        x = self.projection_dropout(x)

        return x

    def transformer_embedding(self, x):
        #########################
        # Transformer embedding
        # torch.Size([2, 1280, 768])
        #print("Transformer embedding", x.shape)

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
        #print("Mask generation", x.shape)

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
        #print("Transformer", x.shape)

        if attention_mask is not None:
            # 00000111
            # make sure padded tokens output 0
            x[attention_mask.bool()] = 0.0

            # extend attention_mask
            attention_mask = (attention_mask[:, None, None, :].to(dtype=x.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )
        # length is not none 추가
        if not self.layer_norm_first:
            x = self.attention_layer_norm(x)

        x = self.transformer_dropout(x)
        for layer in self.encoder_layers:
            if not (self.training and torch.rand(1).item() <= self.transformer_layer_drop):
                x = layer(x, attention_mask)

        return x

    def forward(self, x, lengths=None):
        x, length = self.sample_to_tcn(x, lengths)
        x = self.tcn_to_transformer(x)
        x = self.transformer_embedding(x)
        attention_mask = self.generate_mask(x, lengths)
        x = self.transformer(x, attention_mask)

        #########################
        # Result
        # torch.Size([2, 1280, 768])
        #print("Result", x.shape)

        return x
