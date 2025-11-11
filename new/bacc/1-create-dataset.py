import numpy as np
import os
import librosa
from glob import glob
import re
import pickle
from tqdm import tqdm
import tensorflow as tf

# PARAMETERS
SAMPLE_RATE = 8000  # Target sample rate (resampling from 16kHz)
N_FFT = 256  # FFT size (smaller for 8kHz audio)
HOP_LENGTH = 64  # Hop length
SEGMENT_DURATION = 1.0  # Duration of audio segments in seconds
SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLE_RATE)
CLEAN_DIR = "/home/ayanami/cnn_denoiser/dataset/CleanSpeech_training"
NOISY_DIR = "/home/ayanami/cnn_denoiser/dataset/NoisySpeech_training"
OUTPUT_DIR = "./dataset"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_complex_mask(noisy_spec, clean_spec):
    """Create complex ratio mask"""
    mask_real = (noisy_spec.real * clean_spec.real +
                 noisy_spec.imag * clean_spec.imag) / (np.abs(noisy_spec)**2 + 1e-8)
    mask_imag = (noisy_spec.real * clean_spec.imag -
                 noisy_spec.imag * clean_spec.real) / (np.abs(noisy_spec)**2 + 1e-8)
    return mask_real + 1j * mask_imag


def prepare_complex_specs(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convert audio to normalized complex spectrograms"""
    # Compute STFT
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # Split into real and imaginary components
    real = np.real(spec)
    imag = np.imag(spec)

    # Normalize both components to [-1, 1] range
    real_max = np.max(np.abs(real))
    imag_max = np.max(np.abs(imag))

    real_norm = real / (real_max + 1e-8)
    imag_norm = imag / (imag_max + 1e-8)

    return real_norm, imag_norm, (real_max, imag_max)


def prepare_mask_targets(noisy_spec, clean_spec):
    """Create normalized mask targets for training"""
    # Create complex mask
    mask = create_complex_mask(noisy_spec, clean_spec)

    # Split mask into real and imaginary components
    mask_real = np.real(mask)
    mask_imag = np.imag(mask)

    # Clip mask values to prevent extreme values
    mask_real = np.clip(mask_real, -1, 1)
    mask_imag = np.clip(mask_imag, -1, 1)

    return mask_real, mask_imag


def extract_snr_and_base_name(filename):
    """Extract SNR value and base name from noisy filename"""
    match = re.search(r'noisy\d+_SNRdb_(\d+\.\d+)_(.+)\.wav', filename)
    if match:
        snr = float(match.group(1))
        base_name = match.group(2)
        return snr, base_name
    return None, None


def segment_audio(audio, segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_SAMPLES // 2):
    """Segment audio into overlapping chunks"""
    segments = []
    for i in range(0, len(audio) - segment_samples + 1, hop_samples):
        segment = audio[i:i + segment_samples]
        segments.append(segment)
    return segments


def process_audio_pairs():
    """Process all audio pairs to create dataset"""
    # Find all clean files
    clean_files = glob(os.path.join(CLEAN_DIR, "*.wav"))

    # Data containers
    features = []  # Will store [noisy_real, noisy_imag]
    targets = []   # Will store [mask_real, mask_imag]
    metadata = []  # Will store [clean_file, noisy_file, snr, segment_idx]

    print(f"Found {len(clean_files)} clean files")

    for clean_file in tqdm(clean_files, desc="Processing files"):
        clean_basename = os.path.basename(clean_file)
        base_name = os.path.splitext(clean_basename)[0]

        # Find all corresponding noisy files
        noisy_pattern = os.path.join(NOISY_DIR, f"*_{base_name}.wav")
        noisy_files = glob(noisy_pattern)

        if not noisy_files:
            print(f"No noisy files found for {clean_basename}")
            continue

        # Load clean audio
        clean_audio, _ = librosa.load(clean_file, sr=SAMPLE_RATE)

        for noisy_file in noisy_files:
            # Extract SNR
            snr, _ = extract_snr_and_base_name(os.path.basename(noisy_file))
            if snr is None:
                continue

            # Load noisy audio
            noisy_audio, _ = librosa.load(noisy_file, sr=SAMPLE_RATE)

            # Ensure same length
            min_len = min(len(clean_audio), len(noisy_audio))
            clean_audio_trim = clean_audio[:min_len]
            noisy_audio_trim = noisy_audio[:min_len]

            # Segment audio into chunks
            clean_segments = segment_audio(clean_audio_trim)
            noisy_segments = segment_audio(noisy_audio_trim)

            for i, (clean_segment, noisy_segment) in enumerate(zip(clean_segments, noisy_segments)):
                # Convert to STFT
                noisy_spec = librosa.stft(
                    noisy_segment, n_fft=N_FFT, hop_length=HOP_LENGTH)
                clean_spec = librosa.stft(
                    clean_segment, n_fft=N_FFT, hop_length=HOP_LENGTH)

                # Prepare normalized spectrograms (CNN inputs)
                noisy_real, noisy_imag, _ = prepare_complex_specs(
                    noisy_segment)

                # Prepare mask targets (CNN outputs)
                mask_real, mask_imag = prepare_mask_targets(
                    noisy_spec, clean_spec)

                # Store features and targets
                features.append([noisy_real, noisy_imag])
                targets.append([mask_real, mask_imag])
                metadata.append({
                    'clean_file': clean_file,
                    'noisy_file': noisy_file,
                    'snr': snr,
                    'segment_idx': i
                })

    return np.array(features), np.array(targets), metadata


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def serialize_example(noisy_real, noisy_imag, mask_real, mask_imag, metadata_item):
    """Convert features and targets to a serialized tf.Example."""
    # Serialize numpy arrays
    feature = {
        'noisy_real': _float_feature(noisy_real),
        'noisy_imag': _float_feature(noisy_imag),
        'mask_real': _float_feature(mask_real),
        'mask_imag': _float_feature(mask_imag),
        'clean_file': _bytes_feature(metadata_item['clean_file'].encode()),
        'noisy_file': _bytes_feature(metadata_item['noisy_file'].encode()),
        'snr': tf.train.Feature(float_list=tf.train.FloatList(value=[metadata_item['snr']])),
        'segment_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[metadata_item['segment_idx']]))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def save_as_tfrecord(features, targets, metadata):
    """Save dataset as TFRecord files."""
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine how many examples per file
    n_examples = len(features)
    n_shards = max(1, n_examples // 1000)  # Create a new file every 1000 examples
    examples_per_shard = n_examples // n_shards + 1

    # Create TFRecord files
    for shard_idx in range(n_shards):
        output_filename = os.path.join(OUTPUT_DIR, f'denoiser-{shard_idx:05d}-of-{n_shards:05d}.tfrecord')

        start_idx = shard_idx * examples_per_shard
        end_idx = min((shard_idx + 1) * examples_per_shard, n_examples)

        with tf.io.TFRecordWriter(output_filename) as writer:
            for i in range(start_idx, end_idx):
                example = serialize_example(
                    features[i][0], features[i][1],
                    targets[i][0], targets[i][1],
                    metadata[i]
                )
                writer.write(example)

        print(f"Wrote {end_idx - start_idx} examples to {output_filename}")


def main():
    print("Creating audio denoising dataset...")
    features, targets, metadata = process_audio_pairs()

    # Save as TFRecord files
    save_as_tfrecord(features, targets, metadata)

    # Optionally, still save the numpy files for reference or non-TF use
    # np.save(os.path.join(OUTPUT_DIR, 'features.npy'), features)
    # np.save(os.path.join(OUTPUT_DIR, 'targets.npy'), targets)

    with open(os.path.join(OUTPUT_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # Print dataset stats
    print(f"Dataset created with {len(features)} examples")
    print(
        f"Feature shape: {features[0][0].shape} (real), {features[0][1].shape} (imag)")
    print(
        f"Target shape: {targets[0][0].shape} (mask_real), {targets[0][1].shape} (mask_imag)")
    print(f"Data saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
