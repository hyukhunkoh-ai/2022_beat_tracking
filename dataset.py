import torch
import os
import librosa
import julius
import torchaudio
import random
import math
import re
import numpy as np
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

def slice_label(label_file_path, slice_start_times, audio_length, target_sr, slice_overlap):
    annotations = []

    slice_index = 0
    slice_annotations = []

    with open(label_file_path, 'r') as fp:
        line_index = 0
        next_line_index = 0
        lines = fp.readlines()

        while line_index < len(lines):
            line = lines[line_index]

            current_slice_start_time = slice_start_times[slice_index]/target_sr
            current_slice_end_time = current_slice_start_time + audio_length

            time, beat_number = re.findall(r"[/\d+\.?\d*/]+", line.strip('\n'))
            time = float(time)
            beat_number = int(beat_number)

            relative_time = round(time - current_slice_start_time, 4)
            is_downbeat = 1 if beat_number == 1 else 0

            # 오디오 슬라이드 간에 겹치는 부분이 있으므로 다음 비트의 첫 비트 인덱스를 미리 저장함
            if relative_time > audio_length - slice_overlap and next_line_index == 0:
                next_line_index = line_index

            if relative_time <= audio_length:
                slice_annotations.append([relative_time, is_downbeat])
                line_index += 1

            reached_end_of_file = line_index + 1 == len(lines)
            if relative_time > audio_length or reached_end_of_file:
                # slice annotation을 전체 annotation 리스트에 추가하여 다음 슬라이드로 넘어가게 함
                annotations.append(slice_annotations[:])
                slice_annotations.clear()

                line_index = next_line_index
                next_line_index = 0

                if reached_end_of_file:
                    break
                
                slice_index += 1

    return annotations

def slice_audio(loaded_audio, loaded_audio_length, audio_length, target_sr):
    audio_slices = []

    slice_count = math.ceil(loaded_audio_length / audio_length)
    slice_remainder = loaded_audio_length % audio_length
    slice_overlap = (audio_length - slice_remainder)/(slice_count - 1)

    slice_start_times = []

    # audio slice processing
    for slice_index in range(slice_count):
        slice_start = int((audio_length - slice_overlap)*slice_index*target_sr)
        slice_length = int(audio_length*target_sr)
        audio_slices.append(loaded_audio.narrow(1, slice_start, slice_length))
        slice_start_times.append(slice_start)

    return audio_slices, slice_start_times, slice_overlap

def get_slices(audio_file_path, label_file_path, audio_length, target_sr):
    audio_slices = []
    annotations = []

    loaded_audio, loaded_audio_sr = torchaudio.load(audio_file_path)
    loaded_audio_length = loaded_audio.size(dim=1) / loaded_audio_sr
    target_audio_length = int(audio_length*target_sr)

    # sampling control
    if loaded_audio_sr != target_sr:
        loaded_audio = julius.resample_frac(loaded_audio, loaded_audio_sr, target_sr)

    if loaded_audio.size(dim=1) < target_audio_length:
        loaded_audio, attention_mask = pad(loaded_audio)
        audio_slices.append(loaded_audio)
    elif loaded_audio.size(dim=1) > target_audio_length:
        attention_mask = None

        audio_slices, slice_start_times, slice_overlap = slice_audio(
            loaded_audio,
            loaded_audio_length,
            audio_length,
            target_sr
        )

        if label_file_path != None:
            annotations = slice_label(
                label_file_path,
                slice_start_times,
                audio_length,
                target_sr,
                slice_overlap
            )

    return audio_slices, annotations

def process_pretrain_data(audio_file_paths, audio_length, sr):
    audio_slices = []

    for audio_file_path in audio_file_paths:
        new_audio_slices = get_slices(audio_file_path, None, audio_length, sr)
        audio_slices += new_audio_slices

    return audio_slices

def process_training_data(audio_file_paths, audio_length, sr):
    audio_slices = []
    annotations = []

    '''
    --datapath
        -- dataname
            -- data
                -- *.wav
            -- label
                -- *.txt
                -- *.beats
    '''
    for audio_file_path in audio_file_paths:
        if audio_file_path.find(".wav"):
            label_file_path = audio_file_path.replace(".wav", ".beats")
        elif audio_file_path.find(".mp3"):
            label_file_path = audio_file_path.replace(".mp3", ".beats")

        if label_file_path:
            label_file_path = label_file_path.replace("/data/", "/label/")
            new_audio_slices, new_annotations = get_slices(
                audio_file_path,
                label_file_path,
                audio_length,
                sr
            )

            audio_slices += new_audio_slices
            annotations += new_annotations

    return audio_slices, annotations

class BeatDataset():
    def __init__(self, path, audio_length=12.8, sr=22050):
        self.audio_slices = []
        self.annotations = []

        with open(os.path.join(path, 'new_data.txt'), 'r') as fp:
            audio_file_paths = [line.strip('\n') for line in fp.readlines()]
            self.audio_slices, self.annotations = process_training_data(audio_file_paths, audio_length, sr)

    def __len__(self):
        return len(self.audio_slices)

    def __getitem__(self, idx):
        return self.audio_slices[idx], self.annotations[idx]
        
    # def random_crop(self,item):
    #     crop_size = int(self.sequence_len * self.sr)
    #     start = int(random.random() * (item.shape[0] - crop_size))
    #     return item[start:(start+crop_size)]

class SelfSupervisedDataset(Dataset):
    def __init__(self, path, audio_length=12.8, sr=22050):
        self.audio_slices = process_pretrain_data(
            list(glob(os.path.join(path, 'data', '*.wav'))) + list(glob(os.path.join(path, 'data', '*.mp3'))),
            audio_length,
            sr
        )

    def __len__(self):
        return len(self.audio_slices)

    def __getitem__(self, idx):
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
