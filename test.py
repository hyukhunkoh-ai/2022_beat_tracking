from BeatDataset import ContrastiveDataset
from tcn import TcnModel
import torch
datapth = "./datapath/simac"


trdata = ContrastiveDataset(datapth)



test = torch.Tensor(trdata[0][0]).unsqueeze(0).unsqueeze(-1).permute(0,2,1)
model = TcnModel()
print(test.shape)
out = model(test)
print(out.shape)