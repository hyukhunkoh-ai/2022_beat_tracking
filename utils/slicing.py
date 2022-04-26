import torch
import math
from utils.data_loading import load_audio, load_annotation
from utils.augmentation import apply_augmentations
from utils.padding import pad

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

def slice_annotation(annotation, slice_start_times, audio_length, target_sr, slice_overlap):
    total_annotations = []
    sliced_annotations = []

    slice_index = 0
    annotation_index = 0
    next_annotation_index = 0

    while annotation_index < len(annotation):
        annotation = annotation[annotation_index]

        current_slice_start_time = slice_start_times[slice_index]/target_sr

        time, is_downbeat = annotation
        relative_time = round(time - current_slice_start_time, 4)

        # 오디오 슬라이드 간에 겹치는 부분이 있으므로 다음 비트의 첫 비트 인덱스를 미리 저장함
        if relative_time > audio_length - slice_overlap and next_annotation_index == 0:
            next_annotation_index = annotation_index

        if relative_time <= audio_length:
            sliced_annotations.append([relative_time, is_downbeat])
            annotation_index += 1

        reached_end = annotation_index + 1 == len(annotation)
        if relative_time > audio_length or reached_end:
            # slice annotation을 전체 annotation 리스트에 추가하여 다음 슬라이드로 넘어가게 함
            total_annotations.append(sliced_annotations[:])
            sliced_annotations.clear()

            annotation_index = next_annotation_index
            next_annotation_index = 0

            if reached_end:
                break
            
            slice_index += 1

    return total_annotations

def get_slices(audio_file_path, label_file_path, audio_length, target_sr, augment):
    audio_slices = []
    annotation_slices = []
    attention_mask = None

    loaded_audio, loaded_audio_sr = load_audio(audio_file_path, target_sr)
    loaded_annotation = load_annotation(label_file_path)

    if loaded_audio.size(dim=1) < target_audio_length:
        loaded_audio, attention_mask = pad(loaded_audio, audio_length, target_sr)
        audio_slices.append(loaded_audio)
        annotation_slices.append(loaded_annotation)
    elif loaded_audio.size(dim=1) > target_audio_length:
        attention_mask = torch.ones(size=(1, int(audio_length*target_sr)))

        audio_slices, slice_start_times, slice_overlap = slice_audio(
            loaded_audio,
            loaded_audio_length,
            audio_length,
            target_sr
        )

        if label_file_path != None:
            annotation_slices = slice_annotation(
                loaded_annotation,
                slice_start_times,
                audio_length,
                target_sr,
                slice_overlap
            )

    if label_file_path != None:
        assert len(audio_slices) == len(annotation_slices), "audio and annotation slice counts not equal"

    for index, _ in enumerate(audio_slices):
        audio_slices[index], annotation_slices[index] = apply_augmentations(
            audio_slices[index],
            annotation_slices[index],
            audio_length,
            target_sr
        )

    return audio_slices, annotation_slices, attention_mask
