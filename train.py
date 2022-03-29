import torch
from argparse import ArgumentParser

from dataset import BeatDataset
from models import TcnModel, RegressionModel, ClassificationModel
from anchors import Anchors
from loss import FocalLoss

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

    dataset = BeatDataset(audio_dir)
    train_datasets.append(dataset)

def make_batch(samples):
    audio_list = torch.stack([sample[0] for sample in samples])
    batch_size = len(audio_list)

    desired_length = max([len(sample[1]) for sample in samples])
    padded_data = torch.zeros(batch_size, desired_length, 3)
    for index in range(batch_size):
        padded_data[index, :len(samples[index][1]), :] = torch.Tensor(samples[index][1])

    return audio_list, padded_data

train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
train_dataloader = torch.utils.data.DataLoader(train_dataset_list,
                                            shuffle=True,
                                            batch_size=16,
                                            num_workers=0,
                                            pin_memory=True,
                                            collate_fn=make_batch)

dict_args = vars(args)
model = TcnModel(**dict_args)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

regressionModel = RegressionModel(256)
classificationModel = ClassificationModel(256)
anchorsModel = Anchors(audio_length=12.8, sr=22050, num_anchors=1280)
focalLoss = FocalLoss()

for epoch in range(args.epochs):
    running_loss = 0.0
    for index, data in enumerate(train_dataloader, 0):
        inputs, annotations = data
        optimizer.zero_grad()
        outputs = model(inputs)

        regression = regressionModel(outputs)
        classification = classificationModel(outputs)
        anchors = anchorsModel(inputs)
        loss = focalLoss(classification, regression, anchors, annotations)

        optimizer.step()
        #running_loss += loss.item()
        print("aaa")
        if index % 2000 == 1999:
            print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print("Finished")
