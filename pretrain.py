import torch
import os
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from wav2vec2 import Wav2Vec2Model
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# epochs = 100000
epochs = 1
bs = 5 # 5 * 4 = 20

# to-do:
# dataloader
# padding된 audio sequence, 10초 length
# Dat에서 audio sequence랑 length를 읽어온다
# audio sequence에 padding을 입히는것임
# padding된 Audio sequence랑 length가 뱉어준다

# model에 들어갈 때
# audio sequence 모델에서 계산하는 데 쓰이고
# length는 ATTention mask를 만들어서 계산이 안될 부분을 표시해주는 역할


#train

model = Wav2Vec2Model()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.0005,steps_per_epoch=per_bs, epochs=epochs, pct_start=0.08)


for epoch in range(epochs):
    total_loss = []

    # to-do: 2개(audio seq, length) 뽑아야함
    for x in train_dl:
        dx = x[0] # to-do: add time


        # device 반드시 넣어야함
        dx = dx.to(device)
        model.to(device)
        optim.zero_grad()
        loss = model.calculate_loss(dx, ) #to do: length, attention mask 넣어야함 
        total_loss.append(loss.item())
        loss.backward()
        optim.step()
        scheduler.step()
    print(np.mean(total_loss))


