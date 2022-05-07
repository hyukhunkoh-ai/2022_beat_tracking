import torch
import math
from utils.data_loading import load_audio, load_annotation
from utils.augmentation import apply_augmentations
from utils.padding import pad

def get_slice_count(audio_length, desired_audio_length):
    slice_count = math.ceil(audio_length / desired_audio_length)
    slice_remainder = audio_length % desired_audio_length
    slice_overlap = (desired_audio_length - slice_remainder)/(slice_count - 1) if slice_count > 1 else 1

    return slice_count, slice_overlap

def slice_audio(loaded_audio, loaded_audio_length, audio_length, target_sr):
    audio_slices = []

    slice_count, slice_overlap = get_slice_count(loaded_audio_length, audio_length)

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

def get_slice(audio_file_path, label_file_path, audio_length, sr, augment, audio_start_time):
    loaded_audio, loaded_audio_length = load_audio(audio_file_path, sr)
    loaded_annotation = None
    attention_mask = None

    if label_file_path is not None:
        loaded_annotation = load_annotation(label_file_path)

    if augment:
        loaded_audio, loaded_annotation = apply_augmentations(
            loaded_audio,
            loaded_annotation,
            audio_length,
            sr
        )

    target_audio_length = int(audio_length*sr)
    if loaded_audio.size(dim=1) <= target_audio_length:
        loaded_audio, attention_mask = pad(loaded_audio, audio_length, sr)
        #print(attention_mask.shape)
        #attention_mask = torch.ones(int(audio_length*sr))
        #print(attention_mask.shape)

        return loaded_audio, loaded_annotation, attention_mask
    elif loaded_audio.size(dim=1) > target_audio_length:
        audio_slices, slice_start_times, slice_overlap = slice_audio(
            loaded_audio,
            loaded_audio_length,
            audio_length,
            sr
        )

        attention_mask = torch.ones(int(audio_length*sr))

        # if label_file_path is not None:
        #     # todo: fix for labeled dataset
        #     annotation_slices = slice_annotation(
        #         loaded_annotation,
        #         slice_start_times,
        #         audio_length,
        #         sr,
        #         slice_overlap
        #     )

        slice_index = math.floor(sr * audio_start_time / target_audio_length)
        loaded_audio = audio_slices[slice_index]

        # loaded_annotation이 None일 경우 두번째 반환값은 None이 됨
        if augment:
            loaded_annotation = None

            if label_file_path is not None:
                loaded_annotation = annotation_slices[slice_index]

    return loaded_audio, loaded_annotation, attention_mask

def get_slices(audio_file_path, label_file_path, audio_length, sr, augment):
    audio_slices = []
    annotation_slices = []
    attention_masks = None

    loaded_audio, loaded_audio_length = load_audio(audio_file_path, sr)
    loaded_annotation = None

    if label_file_path is not None:
        loaded_annotation = load_annotation(label_file_path)

    if augment:
        loaded_audio, loaded_annotation = apply_pre_augmentations(
            loaded_audio,
            loaded_annotation,
            audio_length,
            sr
        )

    target_audio_length = int(audio_length*sr)
    if loaded_audio.size(dim=1) < target_audio_length:
        loaded_audio, attention_mask = pad(loaded_audio, audio_length, sr)

        if augment:
            loaded_audio, loaded_annotation = apply_post_augmentations(
                loaded_audio,
                loaded_annotation,
                sr
            )

        audio_slices.append(loaded_audio)
        annotation_slices.append(loaded_annotation)
        attention_masks = torch.ones(size=(1, int(audio_length*sr)))
    elif loaded_audio.size(dim=1) > target_audio_length:
        audio_slices, slice_start_times, slice_overlap = slice_audio(
            loaded_audio,
            loaded_audio_length,
            audio_length,
            sr
        )

        attention_masks = torch.ones(size=(len(audio_slices), int(audio_length*sr)))

        if label_file_path is not None:
            annotation_slices = slice_annotation(
                loaded_annotation,
                slice_start_times,
                audio_length,
                sr,
                slice_overlap
            )

        # loaded_annotation이 None일 경우 두번째 반환값은 None이 됨
        if augment:
            for slice_index, audio_slice in enumerate(audio_slices):
                annotation_slice = None
                if label_file_path is not None:
                    annotation_slice = annotation_slices[slice_index]

                augmented_audio, augmented_annotation = apply_post_augmentations(
                    audio_slice,
                    annotation_slice,
                    sr
                )

                audio_slices[slice_index] = augmented_audio

                if label_file_path is not None:
                    annotation_slices[slice_index] = augmented_annotation

    return audio_slices, annotation_slices, attention_masks