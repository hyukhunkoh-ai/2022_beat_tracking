import tensorflow as tf
import librosa
def extract_log_mel_spectrogram(waveform,
                                sample_rate=16000,
                                frame_length=400,
                                frame_step=160,
                                fft_length=1024,
                                n_mels=64,
                                fmin=60.0,
                                fmax=7800.0):
    """Extract frames of log mel spectrogram from a raw waveform."""

    stfts = tf.signal.stft(
      waveform,
      frame_length=frame_length,
      frame_step=frame_step,
      fft_length=fft_length)
    print(stfts.shape)
    spectrograms = tf.abs(stfts)

    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    print(linear_to_mel_weight_matrix.shape)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

    mel_spectrograms = tf.clip_by_value(
      mel_spectrograms,
      clip_value_min=1e-5,
      clip_value_max=1e8)
    print(mel_spectrograms.shape)
    log_mel_spectrograms = tf.math.log(mel_spectrograms)

    return log_mel_spectrograms


def extract_window(waveform, seg_length=16000):
    """Extracts a random segment from a waveform."""
    padding = tf.maximum(seg_length - tf.shape(waveform)[0], 0)
    left_pad = padding // 2
    right_pad = padding - left_pad
    padded_waveform = tf.pad(waveform, paddings=[[left_pad, right_pad]])
    return tf.image.random_crop(padded_waveform, [seg_length])

if __name__ == "__main__":
    data, _ = librosa.load("../datapath/simac/data/001_A love_supreme_part_1__acknowledgement.wav")
    print(data.shape)
    x = tf.math.l2_normalize(data, epsilon=1e-9)
    print(x.shape)
    waveform_a = extract_window(x)
    print(waveform_a.shape)
    mels_a = extract_log_mel_spectrogram(waveform_a)
    print(mels_a.shape)
    frames_anchors = mels_a[Ellipsis, tf.newaxis]

    waveform_p = extract_window(x)
    waveform_p = waveform_p + (
        0.001 * tf.random.normal(tf.shape(waveform_p)))
    mels_p = extract_log_mel_spectrogram(waveform_p)
    frames_positives = mels_p[Ellipsis, tf.newaxis]
    print(frames_anchors.shape)
    print(frames_positives.shape)
