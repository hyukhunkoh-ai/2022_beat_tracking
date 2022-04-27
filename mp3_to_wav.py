import pydub
from glob import glob
import os

audio_files = list(glob(os.path.join('/beat_tracking/unlabel/fma_30/', '*.mp3')))
for index, audio_file_path in enumerate(audio_files):
    print(index, len(audio_files))
    try:
        audio = pydub.AudioSegment.from_mp3(audio_file_path)
        audio.export(audio_file_path.replace("fma_30", "fma_30_wav").replace(".mp3", ".wav"), format="wav")
    except pydub.exceptions.CouldntDecodeError:
        pass