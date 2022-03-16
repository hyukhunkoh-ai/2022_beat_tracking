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

class BeatDataset():
    def __init__(self, path, audio_length=12.8, sr=44100):
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

        self.label = filter_data(self.label)
        self.data = [(label[:-6] + ".wav").replace("label", "data") for label in self.label]
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        num_audio_samples = int(self.audio_length*self.sr)
        audio, sr = torchaudio.load(self.data[idx])
        target = torch.zeros(int(self.audio_length*100))

        audio = audio.float()
        audio /= audio.abs().max() # normalize
        anot = ''
        '''
        ToDo
        random crop 적용
        anot 가공
        '''

        if audio.size(dim=1) < num_audio_samples:
            audio = torch.nn.ConstantPad1d((0, num_audio_samples - audio.size(dim=1)), 0)(audio)
        elif audio.size(dim=1) > num_audio_samples:
            audio = audio.narrow(1, 0, num_audio_samples)

        # sampling control
        if sr != self.sr:
            audio = julius.resample_frac(audio, sr, self.sr)
        
        beat_indices = []
        normalized_beat_times = []
        beats_by_type = []

        filename = self.label[idx]
        start_time = 0

        with open(filename, 'r') as fp:
            for line in fp.readlines():
                time_in_seconds, beat_number = line.strip('\n').replace('\t', ' ').split(' ')
                time_in_seconds = float(time_in_seconds)
                beat_number = int(beat_number)

                offset_time = time_in_seconds - start_time

                # 지금 한 오디오 파일로부터 12.8초 짜리만 쓴다.
                if offset_time < 0:
                    continue

                if offset_time > self.audio_length:
                    break

                beat_index = int(offset_time*100) - 1
                normalized_beat_time = offset_time/self.audio_length
                beat_type = 1 if beat_number == 1 else 0

                beat_indices.append(beat_index)
                normalized_beat_times.append(normalized_beat_time)
                beats_by_type.append(beat_type)

                target[beat_index] = 1

        annotations = {
            "beat_indices": beat_indices,
            "normalized_beat_times": normalized_beat_times,
            "beats_by_type": beats_by_type,
            "filename": filename
        }

        return audio, target, annotations
        
    def random_crop(self,item):
        crop_size = int(self.sequence_len * self.sr)
        start = int(random.random() * (item.shape[0] - crop_size))
        return item[start:(start+crop_size)]

def filter_data(paths):
    new_paths = []
    for path in paths:
        with open(path, 'r') as fp:
            first_line = fp.readlines()[0]
            time_in_seconds, beat_number = first_line.strip('\n').replace('\t', ' ').split(' ')
            time_in_seconds = float(time_in_seconds)
            if time_in_seconds < 2:
                new_paths.append(path)

    return new_paths


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
