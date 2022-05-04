import os
import torch
import torchaudio
import librosa
from glob import glob

path = '/beat_tracking/unlabel'
#dataset_types = ["60_excerpts_30", "extended_ballroom_30", "acm_mirum_tempo_30_60", "fma_30", "openmic_10"]
dataset_types = ["fma_30"]

for dataset_type in dataset_types:
    audio_file_paths = list(glob(os.path.join(path, dataset_type, '*.wav'))) + list(glob(os.path.join(path, dataset_type, '*.mp3')))

    for i, audio_file_path in enumerate(audio_file_paths):
        loaded_audio, loaded_audio_sr = torchaudio.load(audio_file_path)
        is_nan = torch.isnan(loaded_audio).any()
        num_zeros = torch.count_nonzero(loaded_audio)
        if is_nan or num_zeros == 0:
            os.rename(audio_file_path, audio_file_path.replace(dataset_type, "broken"))
            print(i, len(audio_file_paths), is_nan, audio_file_path, num_zeros)