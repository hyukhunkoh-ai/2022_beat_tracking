import torch
from torch import nn
G = 2
V = 320
cdim = 256
tcn_conv_dim = 512
class Wav2Vec2GumbelVectorQuantizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_groups = G # G
        self.num_vars = V # V

        # storage for codebook variables (codewords)
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, cdim // self.num_groups) # 1*640*128
        )
        self.weight_proj = nn.Linear(tcn_conv_dim, self.num_groups * self.num_vars) # 512 to 320*2

        # can be decayed for training
        # 실험할 값: 0.5, 1, 2, 5
        self.temperature = 1

    def set_temperature(self, temperature: int):
        self.temperature = temperature

    @staticmethod
    def _compute_perplexity(probs, mask=None): # 얼마나 애매한가
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs)) # if true -> probs, false -> zero
            mean_probs_by_each_component = probs.sum(dim=0) / mask.sum() # mean concerning mask
        else:
            mean_probs_by_each_component = probs.mean(dim=0)

        perplexity = torch.exp(-torch.sum(mean_probs_by_each_component * torch.log(mean_probs_by_each_component + 1e-7), dim=-1)).sum()
        # exp(엔트로피(주변확률의 합))
        return perplexity # 1이면 제일 낮음.

    def forward(self, hidden_states, mask_time_indices=None):
        # input은 tcn의 output임.
        # 4*4=16, 1280, 512
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states) # codebook linear mapping -> B,L,640
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1) # B*L*2,320
        # 해당 코드북 id에 가까울 확률

        if self.training:
            # sample code vector probs via gumbel in differentiateable way

            # gumbel softmax 계산
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states) # 각각 gumbel softmax하고

            # compute perplexity
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1) #B*L,640 # argmax들이 연달아 있음
        # 결론적으로 cnn채널에서 확장된 맵핑에서 가장 큰값을 쓰겠다는 것인데..?

        # use probs to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors # (B*L,640,1) * (1,640,128) ->  B*L,640,128
        # 확률 * 코드백터 = bs별 length별 확률을 곱한 코드백터
        codevectors = (
            codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1) # B*L,2,320,128
            .sum(-2) # B*L,2,128
            .view(batch_size, sequence_length, -1) # B,L,256
        )

        return codevectors, perplexity