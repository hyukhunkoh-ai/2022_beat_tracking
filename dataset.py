import torch
import os
import librosa
import julius
import torchaudio
import random
from glob import glob
from torch.utils.data import Dataset
from torchaudio_augmentations import *
# 룸바는 26/27이 표준이다. 차차차는 28/30, 자이브 42/44, 삼바는 50/52, 파소도블레는 60/62 BPM이다. 숫자가 높을수록 빠르다.
#
# 왈츠는 30 BPM, 퀵스텝 50, 폭스트로트 30, 탱고 33/34이 표준으로 되어 있다. 소셜 리듬댄스는 26/50으로 폭이 넓다.
# 그만큼 소셜 리듬댄스는 웬만한 음악이면 다 가능하다는 뜻도 된다. 그래서 편안한 파티용으로 많이 쓴다.

def pad(x, max_length=12.8):
    """
    Pad inputs (on left/right and up to predefined length or max length in the batch)
    Args:
        x: input audio sequence (1, sample 개수)
        max_length: maximum length of the returned list and optionally padding length
    """

    num_samples = x.shape[1]
    attention_mask = np.ones(num_samples, dtype=np.int32)

    difference = max_length*22050 - x
    attention_mask = np.pad(attention_mask, (0, difference))
    padding_shape = (0, difference)
    padded_x = np.pad(num_samples, padding_shape, "constant", constant_values=-1)

    return padded_x, attention_mask

def get_audio(audio, audio_length, target_sr):
    num_audio_samples = int(audio_length*sr)
    audio, sr = torchaudio.load(audio)

    audio = audio.float()
    audio /= audio.abs().max() # normalize
    anot = ''

    # sampling control
    if sr != target_sr:
        audio = julius.resample_frac(audio, sr, target_sr)

    if audio.size(dim=1) < num_audio_samples:
        #audio = torch.nn.ConstantPad1d((0, num_audio_samples - audio.size(dim=1)), 0)(audio)
        audio, attention_mask = pad(audio)
    elif audio.size(dim=1) > num_audio_samples:
        # todo: replace
        audio = audio.narrow(1, 0, num_audio_samples)
        attention_mask = None

    return audio, attention_mask

class BeatDataset():
    def __init__(self, path, audio_length=12.8, sr=22050):
        '''
        --datapath
            -- dataname
                -- data
                    -- *.wav
                -- label
                    -- *.txt
                    -- *.beats
        '''
        self.data = []
        self.label = list(glob(os.path.join(path, 'label', '*.beats')))
        self.audio_length = audio_length
        self.sr = sr

        with open(os.path.join(path, 'new_data.txt'), 'r') as fp:
            for line in fp.readlines():
                self.data.append(line.strip('\n'))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        audio, attention_mask = get_audio(self.data[idx], self.audio_length, self.sr)

        annotations = []

        filename = self.label[idx]
        start_time = 0

        with open(filename, 'r') as fp:
            for index, line in enumerate(fp.readlines()):
                time_start, time_end, is_downbeat = line.strip('\n').split('\t')
                time_start = round(float(time_start), 4)
                time_end = round(float(time_end), 4)
                is_downbeat = int(is_downbeat)

                if time_start < self.audio_length or time_end <= self.audio_length:
                    annotations.append([time_start, time_end, is_downbeat])

        return audio, annotations
        
    def random_crop(self,item):
        crop_size = int(self.sequence_len * self.sr)
        start = int(random.random() * (item.shape[0] - crop_size))
        return item[start:(start+crop_size)]

class SelfSupervisedDataset(Dataset):
    def __init__(self, path, audio_length=12.8, sr=22050):
        self.data =
            list(glob(os.path.join(path, 'data', '*.wav'))) +
            list(glob(os.path.join(path, 'data', '*.mp3')))

        self.audio_length = audio_length
        self.sr = sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # todo: augment
        audio, attention_mask = get_audio(self.data[idx], self.audio_length, self.sr)
        return audio

transforms_polarity = 0.8
transforms_noise = 0.01
transforms_gain = 0.3
transforms_filters = 0.8
transforms_delay = 0.3
transforms_pitch = 0.6
transforms_reverb = 0.6



class ContrastiveDataset(Dataset):

    def __init__(self,datapath,sequence_len=12.8,sr=22050):
        super(ContrastiveDataset, self).__init__()
        self.data = []
        self.sequence_len = sequence_len
        self.sr = sr
        with os.scandir(datapath) as fs:
            for f in fs:
                if f.is_dir():
                    wav = os.path.join(f.path,'*.wav')
                    mp3 = os.path.join(f.path,'*.mp3')
                    self.data += list(glob(wav))
                    self.data += list(glob(mp3))
        self.transform = [
            RandomApply([PolarityInversion()], p=transforms_polarity),
            RandomApply([Noise()], p=transforms_noise),
            RandomApply([Gain()], p=transforms_gain),
            RandomApply(
                [HighLowPass(sample_rate=self.sr)], p=transforms_filters
            ),
            RandomApply([Delay(sample_rate=self.sr)], p=transforms_delay),
            RandomApply(
                [
                    PitchShift(
                        n_samples=self.sequence_len,
                        sample_rate=self.sr,
                    )
                ],
                p=transforms_pitch,
            ),
            RandomApply(
                [Reverb(sample_rate=self.sr)], p=transforms_reverb
            ),
        ]

    def __len__(self):
        return len(self.data)




    def __getitem__(self, idx,transform=False):
        y, sr = librosa.load(self.data[idx])
        # tempo = self.get_tempo(y,sr)
        x1 = self.random_crop(y)
        x2 = self.random_crop(y)
        if transform:
            x1 = self.augment(x1)
            x2 = self.augment(x2)

        return x1,x2


    def get_tempo(self,item,sr):
        onset_env = librosa.onset.onset_strength(item, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return tempo


    def random_crop(self,item):
        crop_size = int(self.sequence_len * self.sr)
        start = int(random.random() * (item.shape[0] - crop_size))
        return item[start:(start+crop_size)]

    def augment(self,item):
        return self.transform(item)
