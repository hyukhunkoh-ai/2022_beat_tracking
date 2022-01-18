import torch
import os
import librosa
from torch.utils.data import Dataset
from glob import glob
class BeatDataset(Dataset):
    def __init__(self,datapath,sr=None,downbeat=False):
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
        wav,_ = librosa.load(self.data[idx],sr=self.sr)
        
        with open(self.label[idx],'r',encoding='utf-8') as f:
            beats = f.read().strip().split('\n')
            if '' in beats:
                beats.remove('')
        if self.downbeat:
            beat_downbeat = list(map(str.split(),beats))
            downbeats = [float(beat) for beat,order in beat_downbeat if int(order) == 1]
            beats = [float(beat) for beat,_ in beat_downbeat]
            return wav,beats,downbeats
        else:
            return wav,beats

                
                
            
