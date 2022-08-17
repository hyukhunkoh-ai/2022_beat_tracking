import os
import librosa
from glob import glob

path = '/beat_tracking/unlabel'
dataset_types = ["60_excerpts_30", "extended_ballroom_30", "acm_mirum_tempo_30_60", "fma_30", "openmic_10"]

for dataset_type in dataset_types:
    audio_file_paths = list(glob(os.path.join(path, dataset_type, '*.wav'))) + list(glob(os.path.join(path, dataset_type, '*.mp3')))

    for audio_file_path in audio_file_paths:
        size = os.path.getsize(audio_file_path)
        if size < 5000:
            os.rename(audio_file_path, audio_file_path.replace(dataset_type, "broken"))