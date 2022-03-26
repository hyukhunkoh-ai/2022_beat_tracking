import torch
from models import Music2VecModel

model = Music2VecModel()
result = model(torch.arange(22050*12.8).view(1, -1))
print(result.shape)
