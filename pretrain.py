import torch
import os
import soundfile as sf
import numpy as np
from dataset import SelfSupervisedDataset
from argparse import ArgumentParser
from models.self_supervised import Music2VecModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 1
bs = 1 # 5 * 4 = 20

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
parser.add_argument('--unlabel_dir', type=str, default='/beat_tracking/unlabel')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()

# memoization을 하게끔 만들었음에도 fma_30가 엄청 크기 때문에 데이터로딩이 오래걸리므로. 테스트 시 fma_30 dataset을 생략하는 것이 유리함
#dataset_types = ["60_excerpts_30", "extended_ballroom_30", "acm_mirum_tempo_30_60", "fma_30", "openmic_10"]
#dataset_types = ["60_excerpts_30", "extended_ballroom_30", "acm_mirum_tempo_30_60", "openmic_10"]
dataset_types = ["openmic_10"]

train_datasets = []
num_files = 0

for dataset_type in dataset_types:
    print("Now loading:", dataset_type)
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

print(len(train_dataloader))

for epoch in range(epochs):
    total_loss = []

    for data in train_dataloader:
        inputs, attention_masks, _ = data

        inputs = inputs.to(device)
        attention_masks = attention_masks.to(device)

        model.to(device)
        optim.zero_grad()
        loss = model.calculate_loss(inputs, attention_masks)
        total_loss.append(loss.item())
        loss.backward()
        optim.step()
        scheduler.step()
        break
    print(np.mean(total_loss))

torch.save(model, 'model.pt')
