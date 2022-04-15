from utils.audio_slicing import get_slices

def process_pretrain_data(audio_file_paths, audio_length, sr):
    audio_slices = []
    attention_masks = []

    for audio_file_path in audio_file_paths:
        new_audio_slices, _, attention_mask = get_slices(audio_file_path, None, audio_length, sr)
        audio_slices += new_audio_slices
        attention_masks += [attention_mask]*len(new_audio_slices)

    return audio_slices, attention_masks

def process_training_data(audio_file_paths, audio_length, sr):
    audio_slices = []
    annotations = []

    '''
    --datapath
        -- dataname
            -- data
                -- *.wav
            -- label
                -- *.txt
                -- *.beats
    '''
    for audio_file_path in audio_file_paths:
        if audio_file_path.find(".wav"):
            label_file_path = audio_file_path.replace(".wav", ".beats")
        elif audio_file_path.find(".mp3"):
            label_file_path = audio_file_path.replace(".mp3", ".beats")

        if label_file_path:
            label_file_path = label_file_path.replace("/data/", "/label/")
            new_audio_slices, new_annotations, _ = get_slices(
                audio_file_path,
                label_file_path,
                audio_length,
                sr
            )

            audio_slices += new_audio_slices
            annotations += new_annotations

    return audio_slices, annotations
