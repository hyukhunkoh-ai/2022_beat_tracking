import torch
import torchaudio
import julius

def load_audio(audio_file_path, target_sr)
    loaded_audio, loaded_audio_sr = torchaudio.load(audio_file_path)
    loaded_audio_length = loaded_audio.size(dim=1) / loaded_audio_sr
    target_audio_length = int(audio_length*target_sr)

    # sampling control
    if loaded_audio_sr != target_sr:
        loaded_audio = julius.resample_frac(loaded_audio, loaded_audio_sr, target_sr)

    # convert to mono
    if len(loaded_audio) == 2:
        loaded_audio = torch.mean(loaded_audio, dim=0).unsqueeze(0)

    return loaded_audio, loaded_audio_length
