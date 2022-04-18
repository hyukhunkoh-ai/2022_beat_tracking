import torch
import os
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from dataset import SelfSupervisedDataset
from argparse import ArgumentParser
from models.self_supervised import Music2VecModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# epochs = 100000
epochs = 1
bs = 4 # 5 * 4 = 20

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

parser = ArgumentParser()
parser.add_argument('--unlabel_dir', type=str, default='./datapath/unlabel')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()

dataset_types = ["60_excerpts_30"]#["60_excerpts_30", "extended_ballroom_30", "acm_mirum_tempo_30_60", "fma_30", "openmic_10"]

train_datasets = []
num_files = 0

for dataset_type in dataset_types:
    audio_dir = os.path.join(args.unlabel_dir, dataset_type)
    dataset = SelfSupervisedDataset(audio_dir)

    train_datasets.append(dataset)
    num_files += len(dataset)

steps_per_epoch = num_files // bs + 1

train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
train_dataloader = torch.utils.data.DataLoader(train_dataset_list,
                                            shuffle=True,
                                            batch_size=bs,
                                            num_workers=0,
                                            pin_memory=True)

model = Music2VecModel()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optim,
    max_lr=0.0005,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    pct_start=0.08
)

for epoch in range(epochs):
    total_loss = []

    # to-do: 2개(audio seq, length) 뽑아야함
    for data in train_dataloader:
        inputs, attention_masks = data

        # device 반드시 넣어야함
        inputs = inputs.to(device)
        attention_masks = attention_masks.to(device)
        print(inputs.shape, attention_masks.shape)

        model.to(device)
        optim.zero_grad()
        loss = model.calculate_loss(inputs, attention_masks)
        total_loss.append(loss.item())
        loss.backward()
        optim.step()
        scheduler.step()
    print(np.mean(total_loss))


