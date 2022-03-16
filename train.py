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

def make_batch(samples):
    audio_list = torch.stack([sample[0] for sample in samples])
    target_list = torch.stack([sample[1] for sample in samples])

    beat_indices_list = [torch.Tensor(sample[2]['beat_indices']) for sample in samples]
    normalized_beat_times_list = [torch.Tensor(sample[2]['normalized_beat_times']) for sample in samples]
    beats_by_type_list = [torch.Tensor(sample[2]['beats_by_type']) for sample in samples]

    desired_length = len(list(sorted(beat_indices_list, key=len))[-1])

    beat_indices_list = [torch.nn.ConstantPad1d((0, desired_length - len(beat_indices)), 0)(beat_indices) for beat_indices in beat_indices_list]
    normalized_beat_times_list = [torch.nn.ConstantPad1d((0, desired_length - len(normalized_beat_times)), 0)(normalized_beat_times) for normalized_beat_times in normalized_beat_times_list]
    beats_by_type_list = [torch.nn.ConstantPad1d((0, desired_length - len(beats_by_type)), 0)(beats_by_type) for beats_by_type in beats_by_type_list]

    new_annotations = {
        'beat_indices': beat_indices_list,
        'normalized_beat_times': normalized_beat_times_list,
        'beats_by_type': beats_by_type_list,
    }

    return audio_list, target_list, new_annotations

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
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(args.epochs):
    running_loss = 0.0
    for index, data in enumerate(train_dataloader, 0):
        inputs, targets, annotations = data
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
