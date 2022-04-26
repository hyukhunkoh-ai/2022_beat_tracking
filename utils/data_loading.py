import torch
import torchaudio
import julius
import re

def load_audio(audio_file_path, target_sr):
    loaded_audio, loaded_audio_sr = torchaudio.load(audio_file_path)
    loaded_audio_length = loaded_audio.size(dim=1) / loaded_audio_sr

    # sampling control
    if loaded_audio_sr != target_sr:
        loaded_audio = julius.resample_frac(loaded_audio, loaded_audio_sr, target_sr)

    # convert to mono
    if len(loaded_audio) == 2:
        loaded_audio = torch.mean(loaded_audio, dim=0).unsqueeze(0)

    return loaded_audio, loaded_audio_length

def load_annotation(label_file_path):
    annotation = []

    with open(label_file_path, 'r') as fp:
        for line in fp.readlines():
            time, beat_number = re.findall(r"[/\d+\.?\d*/]+", line.strip('\n'))
            time = round(float(time), 4)
            beat_number = int(beat_number)
            is_downbeat = 1 if beat_number == 1 else 0
            annotation.append([time, is_downbeat])

    return annotation
