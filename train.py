from sklearn.linear_model import MultiTaskElasticNet
import torch
from argparse import ArgumentParser

from dataset import BeatDataset
from models import TcnModel, Music2VecModel, RegressionModel, ClassificationModel
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
    padded_data = torch.ones(batch_size, desired_length, 3) * -1
    for index in range(batch_size):
        padded_data[index, :len(samples[index][1]), :] = torch.Tensor(samples[index][1])

    return audio_list, padded_data

train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
train_dataloader = torch.utils.data.DataLoader(train_dataset_list,
                                            shuffle=True,
                                            batch_size=4,
                                            num_workers=0,
                                            pin_memory=True,
                                            collate_fn=make_batch)

dict_args = vars(args)
model = Music2VecModel().cuda()#TcnModel(**dict_args)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

regressionModel = RegressionModel(256).cuda()
classificationModel = ClassificationModel(256).cuda()
#anchorsModel = Anchors(audio_length=12.8, sr=22050, num_anchors=1280)
focalLoss = FocalLoss()

for epoch in range(args.epochs):
    running_loss = 0.0
    for index, data in enumerate(train_dataloader, 0):
        inputs, annotations = data
        inputs = inputs.cuda()
        annotations = annotations.cuda()

        optimizer.zero_grad()
        #outputs = torch.randn(16, 1280, 256).cuda()
        outputs = model(inputs)
        outputs = outputs.permute(0, 2, 1)

        regression = regressionModel(outputs)
        classification = classificationModel(outputs)
        #anchors = anchorsModel(inputs)
        loss = focalLoss(classification, regression, annotations)

        optimizer.step()
        #running_loss += loss.item()
        if index % 2000 == 1999:
            print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print("Finished")
