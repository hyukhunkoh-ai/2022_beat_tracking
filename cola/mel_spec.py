import librosa
import numpy as np
import torchaudio.transforms as at
# 보통 자연어 음성 조각 자를 때 25ms, 건너 뛸 때 10ms
# 22050 기준 552, 160
# n_fft는 window length보다 크거나 같아야 함
# win_length가 커질 수록 주파수 성분의 해상도가 높아지고, win_length가 작을 때는 시간 성분에 대한 해상력이 높아짐
## 푸리에 변환 : 음성 신호에 진동수가(frequency분포) 얼마나 있는 지 파악하여 저음과 고음이 얼마나 들어있는지 판단한다.
## STFT : 시간에 따른 진동수 변화를 파악하기 위해 시간을 잘게 쪼개서 푸리에 변환을 한다.
## mel : mel-scale은 사람의 귀가 인지하는 범위에 맞춰 자연의 존재하는 주파수를 사람이 인지하는 형태의 주파수를 가공하는 방법이다.

def mel_spectrogram(
        audio,
        n_mel=64,
        sr=16000,
        n_fft=1024):
    fmax = sr / 2.0
    fmin = 60.0 # default 0
    win_len = int(np.ceil(0.025*sr))
    hop_len = int(np.ceil(0.01*sr))
    spec = at.MelSpectrogram(sample_rate=sr,n_fft=n_fft,win_length=win_len,hop_length=hop_len,n_mels=n_mel,f_max=fmax,f_min=fmin)(audio)
    log_spec = at.AmplitudeToDB()(spec)
    # 주파수,시간축 return -> 뒤바꾸어야 함
    return log_spec.T


if __name__ == "__main__":
    pass