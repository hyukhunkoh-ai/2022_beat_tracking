import torchaudio
import librosa
import torch.nn.functional as F
from torch.utils.data import Dataset
# data = torchaudio.load("./datapath/simac/data/001_A love_supreme_part_1__acknowledgement.wav")
data,_ = librosa.load("../datapath/simac/data/001_A love_supreme_part_1__acknowledgement.wav")
print(data.shape)
# tensor([0.0120, 0.0344, 0.0262,  ..., 0.0119, 0.0616, 0.1040])

# tempo = librosa.beat.tempo(x, sr=sr)

class ColaDataset(Dataset):

    def __init__(self):
        super(ColaDataset, self).__init__()



    def __len__(self):
        return


    def __getitem__(self, item):
        return


    def pad_data(self):



        return

    def add_noise(self):
        return


    def random_crop_two_pairs(self):

        return




