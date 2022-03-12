import torch
from argparse import ArgumentParser

from dataset import BeatDataset
from models import TcnModel

parser = ArgumentParser()
parser.add_argument('--ballroom_dir', type=str, default='./datapath/ballroom')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()

dataset_types = ["ballroom"]
train_datasets = []

for dataset_type in dataset_types:
    if dataset_type == "ballroom" and args.ballroom_dir is not None:
        audio_dir = args.ballroom_dir

    train_datasets.append(BeatDataset(audio_dir))


train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
train_dataloader = torch.utils.data.DataLoader(train_dataset_list,
                                            shuffle=True,
                                            batch_size=16,
                                            num_workers=24,
                                            pin_memory=True)

dict_args = vars(args)
model = TcnModel(**dict_args)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(args.epochs):
    running_loss = 0.0

    for index, data in enumerate(train_dataloader, 0):
        inputs, targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if index % 2000 == 1999:
            print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print("Finished")
