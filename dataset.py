import torch
import os
import librosa
import random
import numpy as np
import json
from glob import glob
from audioread.exceptions import NoBackendError
from torch.utils.data import Dataset
from torchaudio_augmentations import *
from utils.data_processing import process_pretrain_data, process_training_data
from utils.slicing import get_slice_count, get_slices
# 룸바는 26/27이 표준이다. 차차차는 28/30, 자이브 42/44, 삼바는 50/52, 파소도블레는 60/62 BPM이다. 숫자가 높을수록 빠르다.
#
# 왈츠는 30 BPM, 퀵스텝 50, 폭스트로트 30, 탱고 33/34이 표준으로 되어 있다. 소셜 리듬댄스는 26/50으로 폭이 넓다.
# 그만큼 소셜 리듬댄스는 웬만한 음악이면 다 가능하다는 뜻도 된다. 그래서 편안한 파티용으로 많이 쓴다.

class BeatDataset():
    def __init__(self, path, audio_length=12.8, sr=22050, augment=False):
        self.audio_slices = []
        self.annotations = []

        with open(os.path.join(path, 'new_data.txt'), 'r') as fp:
            audio_file_paths = [line.strip('\n') for line in fp.readlines()]
            self.audio_slices, self.annotations = process_training_data(audio_file_paths, audio_length, sr, augment)

    def __len__(self):
        return len(self.audio_slices)

    def __getitem__(self, idx):
        return self.audio_slices[idx], self.annotations[idx]

class SelfSupervisedDataset(Dataset):
    def __init__(self, path, audio_length=12.8, sr=22050, augment=True):
        self.audio_length = audio_length
        self.sr = sr
        self.augment = augment

        audio_file_paths = list(glob(os.path.join(path, '*.wav'))) + list(glob(os.path.join(path, '*.mp3')))
        audio_lengths_json_file_name = 'audio_lengths.json'
        try:
            json_file = open(audio_lengths_json_file_name)
            audio_lengths = json.load(json_file)
        except FileNotFoundError:
            audio_lengths = {}

        self.audio_slices = []
        for index, audio_file_path in enumerate(audio_file_paths):
            #print(index, len(audio_file_paths), audio_file_path)

            audio_duration = None
            if audio_file_path in audio_lengths:
                audio_duration = audio_lengths[audio_file_path]
            else:
                try:
                    audio_duration = librosa.get_duration(filename=audio_file_path)
                except (RuntimeError, NoBackendError):
                    pass

            if audio_duration is not None:
                slice_count, slice_overlap = get_slice_count(audio_duration, audio_length)

                for slice_index in range(slice_count):
                    slice_start = int((audio_length - slice_overlap)*slice_index)
                    self.audio_slices.append({
                        "path": audio_file_path,
                        "start": slice_start,
                    })

                if not audio_file_path in audio_lengths:
                    audio_lengths[audio_file_path] = audio_duration

                    with open(audio_lengths_json_file_name, 'w') as json_file:
                        json.dump(audio_lengths, json_file)

    def __len__(self):
        return len(self.audio_slices)

    def __getitem__(self, index):
        audio_slice = self.audio_slices[index]
        if "data" in audio_slice and "mask" in audio_slice:
            # data와 mask가 이미 준비되어 있음
            #print(1,audio_slice["data"], audio_slice["mask"])
            return audio_slice["data"], audio_slice["mask"]

        first_audio_slice_index = index
        while first_audio_slice_index > 0 and self.audio_slices[first_audio_slice_index - 1]["path"] == audio_slice["path"]:
            first_audio_slice_index -= 1

        first_audio_slice = self.audio_slices[first_audio_slice_index]
        index_offset_from_first_slice = index - first_audio_slice_index

        audio_file_path = audio_slice["path"]

        try:
            new_audio_slices, _, attention_masks = get_slices(
                audio_file_path,
                None,
                self.audio_length,
                self.sr,
                self.augment
            )

            for slice_index_offset, new_audio_slice in enumerate(new_audio_slices):
                attention_mask = attention_masks[slice_index_offset]
                slice_index = first_audio_slice_index + slice_index_offset
                self.audio_slices[slice_index]["data"] = new_audio_slice#.cpu().detach().numpy()
                self.audio_slices[slice_index]["mask"] = attention_mask#.cpu().detach().numpy()

            #print(2, self.audio_slices[index]["data"].shape, self.audio_slices[index]["mask"].shape)
            return self.audio_slices[index]["data"], self.audio_slices[index]["mask"]
        except RuntimeError:
            pass

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
