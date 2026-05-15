import tensorflow as tf
import numpy as np
import librosa
import os
import super_utils as sutils

num_features = 129
window_length = 256
hop_length = 64
sample_rate = 16000

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def serialize_example(noisy_mag_segments, clean_magnitude, noisy_phase=None):
    feature = {
        'noisy_mag_features': _float_feature(noisy_mag_segments),
        'clean_magnitude': _float_feature(clean_magnitude)
    }
    if noisy_phase is not None:
        feature['noisy_phase'] = _float_feature(noisy_phase)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def save_to_tfrecord(clean_folder_path, noisy_folder_path, output_tfrecord, num_features=num_features):
    clean_files = sorted([f for f in os.listdir(clean_folder_path) if f.endswith('.wav')])
    noisy_files = sorted([f for f in os.listdir(noisy_folder_path) if f.endswith('.wav')])

    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        for clean_filename in clean_files:
            clean_file_path = os.path.join(clean_folder_path, clean_filename)
            clean_audio, sr = librosa.load(clean_file_path, sr=sample_rate)
            clean_magnitude_og, clean_phase = sutils.preprocess(clean_audio, sr)

            # Find corresponding noisy samples
            corresponding_noisy_files = [f for f in noisy_files if clean_filename in f]

            for noisy_filename in corresponding_noisy_files:
                noisy_file_path = os.path.join(noisy_folder_path, noisy_filename)
                noisy_audio, sr = librosa.load(noisy_file_path, sr=sample_rate)
                noisy_magnitude, noisy_phase = sutils.preprocess(noisy_audio, sr)

                # phase aware scaling
                clean_magnitude = sutils.phase_aware_scaling(clean_magnitude_og, clean_phase, noisy_phase)

                # Log-magnitude scaling (adding small constant to avoid log(0))
                eps = 1e-10
                noisy_magnitude = np.maximum(noisy_magnitude, eps)
                clean_magnitude = np.maximum(clean_magnitude, eps)

                noisy_magnitude = np.log(noisy_magnitude + eps)
                clean_magnitude = np.log(clean_magnitude + eps)

                # Normalize the features
                noisy_magnitude = (noisy_magnitude - noisy_magnitude.min()) / (noisy_magnitude.max() - noisy_magnitude.min())
                clean_magnitude = (clean_magnitude - clean_magnitude.min()) / (clean_magnitude.max() - clean_magnitude.min())

                # Reshape the features
                noisy_mag_features = sutils.prepare_input_features(noisy_magnitude, 8, num_features)

                # (num_segments, num_features, num_time_steps)
                noisy_mag_features = np.transpose(noisy_mag_features, (2, 0, 1))
                # (num_time_steps, num_features)
                clean_magnitude = np.transpose(clean_magnitude, (1, 0))

                # (num_segments, num_features, num_time_steps, 1)
                noisy_mag_features = np.expand_dims(noisy_mag_features, axis=3)
                # (num_time_steps, num_features, 1)
                clean_magnitude = np.expand_dims(clean_magnitude, axis=2)

                for x_, y_ in zip(noisy_mag_features, clean_magnitude):
                    y_ = np.expand_dims(y_, 2)
                    example = serialize_example(x_, y_)
                    writer.write(example)

def tf_record_parser(record):
    keys_to_features = {
        'noisy_mag_features': tf.io.FixedLenFeature([num_features * 8 * 1], tf.float32),
        'clean_magnitude': tf.io.FixedLenFeature([num_features * 1 * 1], tf.float32)
    }

    features = tf.io.parse_single_example(record, keys_to_features)

    noisy_mag_features = features['noisy_mag_features']
    clean_magnitude = features['clean_magnitude']

    # print(f"Deserialized shapes: noisy_mag_features={noisy_mag_features.shape}, clean_magnitude={clean_magnitude.shape}, noisy_phase={noisy_phase.shape}")
    # print(f"Number of elements: noisy_mag_features={tf.size(noisy_mag_features)}, clean_magnitude={tf.size(clean_magnitude)}, noisy_phase={tf.size(noisy_phase)}")

    n_features = 129
    n_segments = 8

    # reshape input and annotation images
    noisy_mag_features = tf.reshape(noisy_mag_features, (n_features, n_segments, 1), name="noisy_mag_features")
    clean_magnitude = tf.reshape(clean_magnitude, (n_features, 1, 1), name="clean_magnitude")

    return noisy_mag_features, clean_magnitude


def load_tfrecord_dataset(tfrecord_file, batch_size=512):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(tf_record_parser)
    return parsed_dataset.batch(batch_size)

def validate_dataset_scaling(tfrecord_file, num_samples=100):
    """Validate scaling and normalization of the dataset"""
    dataset = load_tfrecord_dataset(tfrecord_file, batch_size=1)

    noisy_min, noisy_max = float('inf'), float('-inf')
    clean_min, clean_max = float('inf'), float('-inf')

    print("Checking dataset scaling...")
    for i, (noisy, clean) in enumerate(dataset):
        if i >= num_samples:
            break

        # Get min/max for this batch
        noisy_min = min(noisy_min, tf.reduce_min(noisy))
        noisy_max = max(noisy_max, tf.reduce_max(noisy))
        clean_min = min(clean_min, tf.reduce_min(clean))
        clean_max = max(clean_max, tf.reduce_max(clean))

        # Check if values are in expected range [0,1]
        if tf.reduce_min(noisy) < 0 or tf.reduce_max(noisy) > 1:
            print(f"WARNING: Noisy sample {i} has values outside [0,1] range")
            print(f"Range: [{tf.reduce_min(noisy):.4f}, {tf.reduce_max(noisy):.4f}]")

        if tf.reduce_min(clean) < 0 or tf.reduce_max(clean) > 1:
            print(f"WARNING: Clean sample {i} has values outside [0,1] range")
            print(f"Range: [{tf.reduce_min(clean):.4f}, {tf.reduce_max(clean):.4f}]")

    print("\nDataset Statistics:")
    print(f"Noisy samples range: [{noisy_min:.4f}, {noisy_max:.4f}]")
    print(f"Clean samples range: [{clean_min:.4f}, {clean_max:.4f}]")

    # Check distribution
    all_values = []
    for noisy, clean in dataset.take(num_samples):
        all_values.extend(noisy.numpy().flatten())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.hist(all_values, bins=50, density=True)
    plt.title('Distribution of Normalized Values')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig('dataset_distribution.png')
    plt.close()


# Example usage to reconstruct and save an audio file

# Example usage
clean_folder_path = 'dataset/CleanSpeech_training'
noisy_folder_path = 'dataset/NoisySpeech_training'
output_tfrecord = 'tfrecords/denoising8k.tfrecord'
# save_to_tfrecord(clean_folder_path, noisy_folder_path, output_tfrecord)

# dataset = load_tfrecord_dataset(output_tfrecord)
validate_dataset_scaling(output_tfrecord, num_samples=5000)  # 1/5th of the dataset
