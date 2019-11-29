import tensorflow as tf
import numpy as np
from utils import play
from data_processing.feature_extractor import FeatureExtractor

train_tfrecords_filenames = './data_processing/test_0.tfrecords'

def tf_record_parser(record):
    keys_to_features = {
        "noise_stft_phase": tf.io.FixedLenFeature((), tf.string, default_value=""),
        'noise_stft_mag_features': tf.io.FixedLenFeature([], tf.string),
        "clean_stft_magnitude": tf.io.FixedLenFeature((), tf.string)
    }

    features = tf.io.parse_single_example(record, keys_to_features)

    noise_stft_mag_features = tf.io.decode_raw(features['noise_stft_mag_features'], tf.float32)
    clean_stft_magnitude = tf.io.decode_raw(features['clean_stft_magnitude'], tf.float32)
    noise_stft_phase = tf.io.decode_raw(features['noise_stft_phase'], tf.float32)

    n_features = 129
    # reshape input and annotation images
    noise_stft_mag_features = tf.reshape(noise_stft_mag_features, (n_features, 8, 1), name="noise_stft_mag_features")
    clean_stft_magnitude = tf.reshape(clean_stft_magnitude, (n_features, 1, 1), name="clean_stft_magnitude")
    noise_stft_phase = tf.reshape(noise_stft_phase, (n_features,), name="noise_stft_phase")

    return noise_stft_mag_features, clean_stft_magnitude, noise_stft_phase

train_dataset = tf.data.TFRecordDataset([train_tfrecords_filenames])
train_dataset = train_dataset.map(tf_record_parser)
train_dataset = train_dataset.repeat(1)
train_dataset = train_dataset.batch(1000)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

window_length=256
overlap=64
sr = 16000

feature_extractor = FeatureExtractor(None, windowLength=window_length, overlap=overlap, sample_rate=sr)


def revert_features_to_audio(features, phase, cleanMean=None, cleanStd=None):
    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    phase = np.transpose(phase, (1, 0))
    features = np.squeeze(features)

    # features = librosa.db_to_amplitude(features)
    # features = librosa.db_to_power(features)
    features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

    features = np.transpose(features, (1, 0))
    return feature_extractor.get_audio_from_stft_spectrogram(features)

for pred, target, phase in train_dataset:

    # pred = np.transpose(pred, (1, 0))
    # target = np.transpose(target, (1, 0))
    print("Min:", np.min(pred), "Max:", np.max(pred))
    print("Min:", np.min(target), "Max:", np.max(target))
    print("Min:", np.min(phase), "Max:", np.max(phase))

    phase = np.transpose(phase.numpy(), (1, 0))
    print("Pred:", pred.shape)
    print("Phase:", phase.shape)
    print("target:", target.shape)
    audio = revert_features_to_audio(target.numpy(), phase)
    break

print("Audio length:", len(audio))
play(audio, sample_rate=16000)

# Min: -0.5883574 Max: 10.728247
# Min: -4.8901606 Max: 7.3664904
# Min: -3.1415927 Max: 3.1415927
# Phase: (129, 201)
# target: (201, 129, 1, 1)
# Audio length: 12800