import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


def DotProduct(anchor, positive):
    anchor = F.normalize(anchor,dim=-1,p=2)
    positive = F.normalize(positive,dim=-1,p=2)
    return torch.mm(anchor,positive.t())

def get_efficient_net_encoder(input_shape, pooling):
    # input shape = 64,1
    # pooling = 'max'
    return EfficientNet.from_name("efficientnet-b0", include_top=False,in_channels=1)




class Cola(nn.Module):
    ''' implement cola '''
    def __init__(self,embedding_dim,temperature=0.2,p=0.2):
        super(Cola, self).__init__()
        self.p = p
        self.do = nn.Dropout(p)
        self.temperature = temperature


        # embedding model
        # embedding dim = 512
        self.encoder = get_efficient_net_encoder()
        self.embed = nn.Linear(1280,embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=512)
        self.linear = torch.nn.Linear(512, 512, bias=False)

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        # anchors, positives = x
        x1,x2 = x

        x1 = self.encoder(x1)
        x1 = self.do(self.embed(x1))
        x1 = torch.tanh(self.layer_norm(x1))


        x2 = self.encoder(x2)
        x2 = self.do(self.embed(x2))
        x2 = torch.tanh(self.layer_norm(x2))


        x1 = self.linear(x1)

        return x1,x2


    def start_step(self, x):
        x1,x2 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)
        pred_y = self.similarity_layer(x1,x2)


        # loss
        loss = F.cross_entropy(pred_y,y)

        # 정확도
        _, preds = torch.max(pred_y,1)
        acc = (preds==y).mean()

        return loss,acc


