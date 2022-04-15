from utils.padding import pad
import torch
import math
import julius
import re
import torchaudio

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

def get_slices(audio_file_path, label_file_path, audio_length, target_sr):
    audio_slices = []
    annotations = []
    attention_mask = None

    loaded_audio, loaded_audio_sr = torchaudio.load(audio_file_path)
    loaded_audio_length = loaded_audio.size(dim=1) / loaded_audio_sr
    target_audio_length = int(audio_length*target_sr)

    # sampling control
    if loaded_audio_sr != target_sr:
        loaded_audio = julius.resample_frac(loaded_audio, loaded_audio_sr, target_sr)

    # convert to mono
    if len(loaded_audio) == 2:
        loaded_audio = torch.mean(loaded_audio, dim=0).unsqueeze(0)

    if loaded_audio.size(dim=1) < target_audio_length:
        loaded_audio, attention_mask = pad(loaded_audio, audio_length, target_sr)
        audio_slices.append(loaded_audio)
    elif loaded_audio.size(dim=1) > target_audio_length:
        attention_mask = torch.ones(size=(1, int(audio_length*target_sr)))

        audio_slices, slice_start_times, slice_overlap = slice_audio(
            loaded_audio,
            loaded_audio_length,
            audio_length,
            target_sr
        )

        slice_count = len(audio_slices)

        if label_file_path != None:
            annotations = slice_label(
                label_file_path,
                slice_start_times,
                audio_length,
                target_sr,
                slice_overlap
            )

    return audio_slices, annotations, attention_mask
