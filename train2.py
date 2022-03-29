import torch
from models import Music2VecModel

model = Music2VecModel()
print(torch.arange(22050*12.8*2).view(2, 1, -1).shape)
result = model(torch.arange(22050*12.8*2).view(2, 1, -1))
print(result.shape)
