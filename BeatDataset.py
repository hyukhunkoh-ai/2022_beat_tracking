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
    def __init__(self,datapath,sr=44100,downbeat=False):
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
        self.label = []
        self.sr = sr
        self.downbeat= downbeat
        
        
        with os.scandir(datapath) as fs:
            for f in fs:
                if f.is_dir():
                    data = os.path.join(f.path,'data','*.wav')
                    label = os.path.join(f.path,'label','*.txt')
                    beats = os.path.join(f.path,'label','*.beats')
                    self.data += list(glob(data))
                    self.label += list(glob(label))
                    self.label += list(glob(beats))
    
    
    
    def __len__(self):
        return len(self.label)
    
    
    def __getitem__(self,idx):
        audio, sr = torchaudio.load(self.data[idx])
        audio = audio.float()
        audio /= audio.abs().max() # normalize
        
        # sampling control
        if sr != self.sr:
            audio = julius.resample_frac(audio, sr, self.sr)
        
        with open(self.label[idx],'r',encoding='utf-8') as f:
            beats = f.read().strip().split('\n')
            if '' in beats:
                beats.remove('')
        if self.downbeat:
            beat_downbeat = list(map(str.split(),beats))
            downbeats = [float(beat) for beat,order in beat_downbeat if int(order) == 1]
            beats = [float(beat) for beat,_ in beat_downbeat]
            return audio,beats,downbeats
        else:
            return audio,beats


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



    def __getitem__(self, item,transform=False):
        y, sr = librosa.load(item)
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
        crop_size = self.sequence_len * self.sr
        start = int(random.random() * (item.shape[0] - crop_size))
        return item[start:(start+crop_size)]

    def augment(self,item):
        return self.transform(item)